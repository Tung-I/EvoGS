"""Standalone EvoGS evaluator.

Re-runs per-level PSNR/SSIM/LPIPS on saved checkpoints without re-training.
Also reports tree storage (topology.json + residuals.npz) vs active-leaves PLY.

Usage:
  python evogs/eval_evogs.py \\
      --data-dir /path/to/garden \\
      --result-dir results/garden_evogs
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent / "gsplat" / "examples"))

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.colmap import Dataset, Parser
from evogs.train_evogs import EvoGSConfig, _load_ply_as_splats, _eval_splats


@dataclass
class EvalConfig:
    data_dir: str = "data/garden"
    result_dir: str = "results/garden_evogs"
    test_every: int = 8
    normalize_world_space: bool = True
    data_factors: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    sh_degree: int = 3
    packed: bool = True
    near_plane: float = 0.01
    far_plane: float = 1e10
    levels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # If set, render EVERY level at this single factor instead of its native
    # per-level factor (the "fixed-resolution streaming ladder" eval, mirroring
    # LapisGS metrics_fixed_factor4.json). None = native per-level eval.
    eval_factor: Optional[int] = None
    # Alpha below this counts a leaf as "transparent" (ghost) for the redundancy %.
    transparent_alpha: float = 0.05


def main(ecfg: EvalConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build a minimal EvoGSConfig for helper functions
    cfg = EvoGSConfig(
        data_dir=ecfg.data_dir,
        result_dir=ecfg.result_dir,
        test_every=ecfg.test_every,
        normalize_world_space=ecfg.normalize_world_space,
        data_factors=ecfg.data_factors,
        sh_degree=ecfg.sh_degree,
        packed=ecfg.packed,
        near_plane=ecfg.near_plane,
        far_plane=ecfg.far_plane,
    )

    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    level_dir = os.path.join(ecfg.result_dir, "levels")
    results = []

    for level_idx in ecfg.levels:
        train_factor = ecfg.data_factors[level_idx]
        # native per-level eval, unless a fixed eval factor is requested
        factor     = ecfg.eval_factor if ecfg.eval_factor is not None else train_factor
        leaves_ply = os.path.join(level_dir, f"level_{level_idx:02d}_leaves.ply")

        if not os.path.exists(leaves_ply):
            print(f"[Eval] Level {level_idx}: missing {leaves_ply}, skipping")
            continue

        leaves_mb = os.path.getsize(leaves_ply) / 1e6
        tree_dir  = os.path.join(level_dir, f"level_{level_idx:02d}_tree")
        topo_mb   = os.path.getsize(os.path.join(tree_dir, "topology.json")) / 1e6 \
                    if os.path.exists(os.path.join(tree_dir, "topology.json")) else 0.0
        res_mb    = os.path.getsize(os.path.join(tree_dir, "residuals.npz")) / 1e6 \
                    if os.path.exists(os.path.join(tree_dir, "residuals.npz")) else 0.0

        raw = _load_ply_as_splats(leaves_ply, cfg, device)
        splats_eval = {k: v.to(device) for k, v in raw.items()}

        # Redundancy / "ghost" fraction: leaves whose alpha is near-zero. Opacities
        # are stored as logits, so sigmoid → alpha. (paper Table 3 mechanism)
        alpha = torch.sigmoid(splats_eval["opacities"].flatten())
        transparent_frac = float((alpha < ecfg.transparent_alpha).float().mean())

        parser = Parser(
            data_dir=ecfg.data_dir, factor=factor,
            normalize=ecfg.normalize_world_space, test_every=ecfg.test_every,
        )
        valset = Dataset(parser, split="val")
        print(f"\n[Eval L{level_idx}] train_f={train_factor} eval_f={factor}, "
              f"{splats_eval['means'].shape[0]:,} leaves, {len(valset)} test imgs, "
              f"transp={transparent_frac*100:.1f}%, leaves={leaves_mb:.1f}MB, tree={topo_mb+res_mb:.2f}MB")

        stats = _eval_splats(splats_eval, valset, device, cfg,
                             psnr_m, ssim_m, lpips_m, tag=f"L{level_idx}")
        stats.update({"level": level_idx, "factor": factor,
                      "train_factor": train_factor, "eval_factor": factor,
                      "transparent_frac": round(transparent_frac, 4),
                      "leaves_mb": round(leaves_mb, 2),
                      "tree_mb": round(topo_mb + res_mb, 2)})
        results.append(stats)
        del splats_eval

    fname = ("metrics_eval.json" if ecfg.eval_factor is None
             else f"metrics_fixed_factor{ecfg.eval_factor}.json")
    out_path = os.path.join(ecfg.result_dir, "eval", fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'─'*100}")
    print(f"{'Lvl':>3} {'evalF':>6} {'#Leaves':>10} {'PSNR':>7} {'SSIM':>7} "
          f"{'LPIPS':>7} {'Transp%':>8} {'Leaves(MB)':>11} {'Tree(MB)':>9}")
    print("─" * 100)
    for r in results:
        print(f"{r['level']:>3} {r['factor']:>6}x {r['num_GS']:>10,} "
              f"{r['psnr']:>7.3f} {r['ssim']:>7.4f} {r['lpips']:>7.3f} "
              f"{r['transparent_frac']*100:>7.1f}% {r['leaves_mb']:>11.1f} {r['tree_mb']:>9.2f}")
    print("─" * 100)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    ecfg = tyro.cli(EvalConfig)
    main(ecfg)
