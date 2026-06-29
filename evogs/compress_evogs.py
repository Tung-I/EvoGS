"""EvoGS compression PoC: 8-bit uniform scalar quantization + zstd (paper §5).

Two numbers are reported per level:

  1. Streamed storage = root params (L0 leaves) + per-level (ψ, α) residuals,
     each 8-bit uniformly quantized then zstd-compressed. This is the payload a
     client downloads to reconstruct level L (paper's "storage").

  2. Decode→render ΔPSNR: 8-bit quantize the *materialized* leaf params,
     dequantize, render the val set, and compare PSNR to the full-precision
     model. This measures the quality cost of quantization on the rendered
     output (render footprint = "memory").

Quantization mirrors gsplat/gsplat/compression/png_compression.py (per-channel
min/max → round(x·255)), applied to flat parameter / residual arrays.

Usage:
  python evogs/compress_evogs.py --data-dir /path/garden \
      --result-dir results/garden_evogs_asym --level 3 --eval-factor 1
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent.parent / "gsplat" / "examples"))

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.colmap import Dataset, Parser
from evogs.train_evogs import EvoGSConfig, _load_ply_as_splats, _eval_splats

_ZSTD_LEVEL = 19


# Per-parameter bit depth. `means` get 16 bits: at scene scale, 8-bit absolute
# positions give ~extent/256 error → catastrophic (−12 dB). Everything else
# 8-bit (paper §5 PoC). Residual ψ/α default to 8-bit too (small deltas).
BIT_DEPTH = {"means": 16, "quats": 8, "scales": 8, "opacities": 8,
             "sh0": 8, "shN": 8, "psi": 8, "alpha": 8}


def _quantize(x: np.ndarray, bits: int):
    """Per-column uniform quant of a [N, D] float array to `bits` bits."""
    x = x.reshape(x.shape[0], -1).astype(np.float32)
    mn = x.min(axis=0)
    scale = np.clip(x.max(axis=0) - mn, 1e-12, None)
    levels = float((1 << bits) - 1)
    dtype = np.uint16 if bits > 8 else np.uint8
    q = np.round((x - mn) / scale * levels).clip(0, levels).astype(dtype)
    return q, mn.astype(np.float32), scale.astype(np.float32)


def _dequantize(q, mn, scale, bits: int):
    return q.astype(np.float32) / float((1 << bits) - 1) * scale + mn


def _zstd_bytes(arr: np.ndarray) -> int:
    c = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
    return len(c.compress(np.ascontiguousarray(arr).tobytes()))


def _compress_param_dict(tensors: dict):
    """Per-param quant (BIT_DEPTH) + zstd. Returns (raw_b, q_b, zstd_b, deq)."""
    raw_b = q_b = zstd_b = 0
    deq = {}
    for k, v in tensors.items():
        a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        shape = a.shape
        bits = BIT_DEPTH.get(k, 8)
        q, mn, scale = _quantize(a, bits)
        raw_b += a.astype(np.float32).nbytes
        q_b += q.nbytes + mn.nbytes + scale.nbytes
        zstd_b += _zstd_bytes(q) + mn.nbytes + scale.nbytes
        deq[k] = torch.from_numpy(_dequantize(q, mn, scale, bits).reshape(shape))
    return raw_b, q_b, zstd_b, deq


@dataclass
class CompressConfig:
    data_dir: str = "data/garden"
    result_dir: str = "results/garden_evogs_asym"
    level: int = 3
    eval_factor: int = 1
    test_every: int = 8
    sh_degree: int = 3
    packed: bool = True


def main(ccfg: CompressConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EvoGSConfig(data_dir=ccfg.data_dir, result_dir=ccfg.result_dir,
                      test_every=ccfg.test_every, sh_degree=ccfg.sh_degree,
                      packed=ccfg.packed)
    level_dir = os.path.join(ccfg.result_dir, "levels")
    MB = 1e6

    # ── 1. Streamed storage: root (L0) + residuals (L1..level) ────────────
    print(f"\n{'='*70}\nStreamed storage (root + ψ/α residuals), 8-bit + zstd\n{'='*70}")
    root_ply = os.path.join(level_dir, "level_00_leaves.ply")
    root = _load_ply_as_splats(root_ply, cfg, "cpu")
    r_raw, r_q8, r_zstd, _ = _compress_param_dict(root)
    print(f"{'root (L0)':<22} raw {r_raw/MB:8.1f}  q8 {r_q8/MB:8.1f}  zstd {r_zstd/MB:8.1f} MB")
    tot_raw, tot_q8, tot_zstd = r_raw, r_q8, r_zstd

    for L in range(1, ccfg.level + 1):
        npz = os.path.join(level_dir, f"level_{L:02d}_tree", "residuals.npz")
        if not os.path.exists(npz):
            print(f"{'L%d residuals' % L:<22} (missing)")
            continue
        d = np.load(npz)
        res = {"psi": d["psi"], "alpha": d["alpha"]}
        raw, q8, z, _ = _compress_param_dict(res)
        tot_raw += raw; tot_q8 += q8; tot_zstd += z
        print(f"{'L%d ψ/α (%d pairs)' % (L, d['psi'].shape[0]):<22} "
              f"raw {raw/MB:8.1f}  q8 {q8/MB:8.1f}  zstd {z/MB:8.1f} MB")
    print("-" * 70)
    print(f"{'TOTAL streamed':<22} raw {tot_raw/MB:8.1f}  q8 {tot_q8/MB:8.1f}  "
          f"zstd {tot_zstd/MB:8.1f} MB   ({tot_raw/tot_zstd:.1f}× vs raw)")

    # ── 2. Decode→render ΔPSNR on the materialized leaves ─────────────────
    leaves_ply = os.path.join(level_dir, f"level_{ccfg.level:02d}_leaves.ply")
    leaves = _load_ply_as_splats(leaves_ply, cfg, "cpu")
    l_raw, l_q8, l_zstd, leaves_deq = _compress_param_dict(leaves)
    leaves_mb = os.path.getsize(leaves_ply) / MB
    print(f"\n{'='*70}\nRender footprint (materialized L{ccfg.level} leaves)\n{'='*70}")
    print(f"PLY on disk {leaves_mb:.1f} MB | raw {l_raw/MB:.1f} | "
          f"q8 {l_q8/MB:.1f} | q8+zstd {l_zstd/MB:.1f} MB  ({l_raw/l_zstd:.1f}×)")

    parser = Parser(data_dir=ccfg.data_dir, factor=ccfg.eval_factor,
                    normalize=True, test_every=ccfg.test_every)
    valset = Dataset(parser, split="val")
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    full = {k: v.to(device) for k, v in leaves.items()}
    s_full = _eval_splats(full, valset, device, cfg, psnr_m, ssim_m, lpips_m, tag="full")
    del full
    deq = {k: v.to(device) for k, v in leaves_deq.items()}
    s_q = _eval_splats(deq, valset, device, cfg, psnr_m, ssim_m, lpips_m, tag="q8")

    dpsnr = s_q["psnr"] - s_full["psnr"]
    print(f"\nΔPSNR (q8 − full) = {dpsnr:+.3f} dB  "
          f"(full {s_full['psnr']:.3f} → q8 {s_q['psnr']:.3f})")

    out = {
        "level": ccfg.level, "eval_factor": ccfg.eval_factor,
        "streamed_raw_mb": tot_raw / MB, "streamed_q8_mb": tot_q8 / MB,
        "streamed_zstd_mb": tot_zstd / MB,
        "leaves_ply_mb": leaves_mb, "leaves_q8zstd_mb": l_zstd / MB,
        "psnr_full": s_full["psnr"], "psnr_q8": s_q["psnr"], "dpsnr": dpsnr,
    }
    out_path = os.path.join(ccfg.result_dir, "eval", f"compress_L{ccfg.level}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main(tyro.cli(CompressConfig))
