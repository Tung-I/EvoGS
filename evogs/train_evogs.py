"""EvoGS trainer: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree.

Algorithm (coarse-to-fine image pyramid, 4 levels):
  Level 0: Train root model G0 from SFM init at 1/8 resolution, 30K steps, DefaultStrategy.
  Level 1: Load G0 leaves. Train evolution layer at 1/4 resolution, 30K steps.
           High-gradient leaves are split into two symmetric children at every 3K steps
           until step 15K (5 split events per level).
  Level 2: Load Level-1 leaves. Repeat at 1/2 resolution.
  Level 3: Load Level-2 leaves. Repeat at full resolution.

Key difference from LapisGS: refined parents are REMOVED from the active set; only
leaf nodes are rasterized. Children are initialized symmetrically (C1 = P + ψ,
C2 = P - ψ) and trained independently. Residuals (ψ, α) are extracted post-training.

Reference: EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution
Tree for Scalable 3D Streaming, arXiv:2606.07179.

Usage:
  python evogs/train_evogs.py \\
      --data-dir /path/to/garden --test-every 8 \\
      --result-dir results/garden_evogs \\
      --stages level0 level1 level2 level3 eval
"""

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, str(Path(__file__).parent.parent / "gsplat" / "examples"))

from gsplat.exporter import export_splats, load_ply_to_splats
from gsplat.losses import l1_loss, ssim_loss
from gsplat.rendering import rasterization
from gsplat.scene import GaussianScene
from gsplat.stage import Stage
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove as gs_remove

from datasets.colmap import Dataset, Parser
from utils import knn, rgb_to_sh, set_random_seed

from evogs.evolution_tree import EvolutionTree
from evogs.learned_repr import LearnedLeafSet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvoGSConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "data/garden"
    test_every: int = 8
    result_dir: str = "results/garden_evogs"
    normalize_world_space: bool = True

    # Resolution pyramid: L0 @ factor[0], L1 @ factor[1], ...
    data_factors: List[int] = field(default_factory=lambda: [8, 4, 2, 1])

    # ── Training ──────────────────────────────────────────────────────────
    steps_per_level: int = 30_000
    sh_degree: int = 3
    sh_degree_interval: int = 1_000
    ssim_lambda: float = 0.2
    batch_size: int = 1
    packed: bool = True
    near_plane: float = 0.01
    far_plane: float = 1e10

    # ── Learning rates ────────────────────────────────────────────────────
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20.0

    # ── Level-0 DefaultStrategy ───────────────────────────────────────────
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3_000
    grow_grad2d: float = 2e-4
    grow_scale3d: float = 0.01
    prune_opa: float = 0.005

    # ── Evolution splits (levels 1-3) ─────────────────────────────────────
    split_every: int = 3_000          # evolution split event every N steps
    split_until: int = 15_000         # stop splitting after this step
    split_frac: float = 0.01          # fraction of active leaves to split per event
    opacity_prune_threshold: float = 0.005

    # ── Learned representation (faithful EvoGS: trainable collinear ψ/α) ────
    learned_repr: bool = False        # use trainable (ψ, α) instead of free children
    asymmetric: bool = True           # learn per-attribute α (vs α≡1 symmetric)
    # "Ancestors frozen" = internal/split nodes (frozen automatically via detached
    # base snapshots). Frontier leaves (incl. inherited) stay trainable → False.
    freeze_inherited: bool = False
    log_alpha_lr: float = 1e-2        # LR for log-α

    # ── Stages to run ─────────────────────────────────────────────────────
    stages: List[str] = field(
        default_factory=lambda: ["level0", "level1", "level2", "level3", "eval"]
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def rasterize_splats(
    splats: dict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
    packed: bool = True,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    backgrounds: Optional[Tensor] = None,
):
    means     = splats["means"]
    quats     = splats["quats"]
    scales    = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors    = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    viewmats = torch.linalg.inv(camtoworlds)

    renders, alphas, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, packed=packed, near_plane=near_plane, far_plane=far_plane,
        backgrounds=backgrounds, rasterize_mode="classic",
    )
    return renders, alphas, info


def _make_optimizers(splats: torch.nn.ParameterDict, cfg: EvoGSConfig, scene_scale: float):
    BS = cfg.batch_size
    eps = 1e-15 / math.sqrt(BS)
    betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))

    def _adam(param, lr):
        return torch.optim.Adam(
            [{"params": param, "lr": lr * math.sqrt(BS)}],
            eps=eps, betas=betas, fused=True,
        )

    return {
        "means":     _adam(splats["means"],     cfg.means_lr * scene_scale),
        "scales":    _adam(splats["scales"],    cfg.scales_lr),
        "quats":     _adam(splats["quats"],     cfg.quats_lr),
        "opacities": _adam(splats["opacities"], cfg.opacities_lr),
        "sh0":       _adam(splats["sh0"],       cfg.sh0_lr),
        "shN":       _adam(splats["shN"],       cfg.shN_lr),
    }


def _init_splats_from_parser(parser, cfg: EvoGSConfig, scene_scale: float, device: str):
    points = torch.from_numpy(parser.points).float()
    rgbs   = torch.from_numpy(parser.points_rgb / 255.0).float()

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg  = torch.sqrt(dist2_avg)
    scales    = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats     = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), 0.1))

    K  = (cfg.sh_degree + 1) ** 2
    sh = torch.zeros((N, K, 3))
    sh[:, 0, :] = rgb_to_sh(rgbs)

    splats = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(points),
        "scales":    torch.nn.Parameter(scales),
        "quats":     torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0":       torch.nn.Parameter(sh[:, :1, :]),
        "shN":       torch.nn.Parameter(sh[:, 1:, :]),
    }).to(device)
    return splats


def _load_ply_as_splats(path: str, cfg: EvoGSConfig, device: str) -> dict:
    raw = load_ply_to_splats(path)
    K_want = (cfg.sh_degree + 1) ** 2 - 1
    shN = raw["shN"]
    if shN.shape[1] < K_want:
        pad = torch.zeros(shN.shape[0], K_want - shN.shape[1], 3)
        shN = torch.cat([shN, pad], dim=1)
    else:
        shN = shN[:, :K_want, :]
    return {
        "means":     raw["means"],
        "scales":    raw["scales"],
        "quats":     raw["quats"],
        "opacities": raw["opacities"],
        "sh0":       raw["sh0"],
        "shN":       shN,
    }


def _splats_dict_to_paramdict(raw: dict, device: str) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.clone().to(device), requires_grad=True)
        for k, v in raw.items()
    })


def _save_full_ply(splats: dict, path: str):
    export_splats(
        means=splats["means"], scales=splats["scales"],
        quats=splats["quats"], opacities=splats["opacities"],
        sh0=splats["sh0"], shN=splats["shN"],
        format="ply", save_to=path,
    )


@torch.no_grad()
def _eval_splats(splats_dict: dict, valset, device: str, cfg: EvoGSConfig,
                 psnr_m, ssim_m, lpips_m, tag: str = "val") -> dict:
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    metrics = defaultdict(list)

    for data in valloader:
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        renders, _, _ = rasterize_splats(
            splats_dict, camtoworlds, Ks, width, height,
            sh_degree=cfg.sh_degree, packed=cfg.packed,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        pred   = renders[..., :3].clamp(0, 1)
        pred_p = pred.permute(0, 3, 1, 2)
        gt_p   = pixels.permute(0, 3, 1, 2)

        metrics["psnr"].append(psnr_m(pred_p, gt_p))
        metrics["ssim"].append(ssim_m(pred_p, gt_p))
        metrics["lpips"].append(lpips_m(pred_p, gt_p))

    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats["num_GS"] = len(splats_dict["means"])
    print(f"[Eval {tag}] PSNR={stats['psnr']:.3f} SSIM={stats['ssim']:.4f} "
          f"LPIPS={stats['lpips']:.3f} #GS={stats['num_GS']:,}")
    return stats


def _accumulate_grad2d(evo_state: dict, info: dict, packed: bool = True):
    """Accumulate per-Gaussian 2D gradient norms into evo_state (in-place)."""
    if info.get("means2d") is None or info["means2d"].grad is None:
        return

    grads = info["means2d"].grad.clone()
    grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
    grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

    if packed:
        gs_ids = info["gaussian_ids"]
    else:
        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        grads = grads[sel]

    evo_state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
    evo_state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Level 0: standard 3DGS base model at 1/8 resolution
# ---------------------------------------------------------------------------

def run_level0(cfg: EvoGSConfig, device: str) -> str:
    """Train root model G0 at 1/8 resolution. Returns path to saved PLY."""
    level_dir = os.path.join(cfg.result_dir, "levels")
    os.makedirs(level_dir, exist_ok=True)
    leaves_ply = os.path.join(level_dir, "level_00_leaves.ply")

    if os.path.exists(leaves_ply):
        print(f"[Level 0] Already exists: {leaves_ply}")
        return leaves_ply

    factor = cfg.data_factors[0]
    print(f"\n{'─'*70}")
    print(f"[Level 0] Training base model at factor={factor} ({cfg.steps_per_level} steps)")
    print(f"{'─'*70}")

    parser = Parser(
        data_dir=cfg.data_dir, factor=factor,
        normalize=cfg.normalize_world_space, test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")
    valset   = Dataset(parser, split="val")
    print(f"[Level 0] Train: {len(trainset)} imgs, Val: {len(valset)} imgs @ factor={factor}")

    scene_scale = parser.scene_scale * 1.1
    with open(os.path.join(cfg.result_dir, "scene_scale.json"), "w") as _f:
        json.dump({"scene_scale": float(scene_scale)}, _f)

    splats = _init_splats_from_parser(parser, cfg, scene_scale, device)
    optimizers = _make_optimizers(splats, cfg, scene_scale)

    strategy = DefaultStrategy(
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
        grow_grad2d=cfg.grow_grad2d,
        grow_scale3d=cfg.grow_scale3d,
        prune_opa=cfg.prune_opa,
        absgrad=False,
        verbose=True,
    )
    strategy.check_sanity(splats, optimizers)

    def _render_fn(camtoworlds, Ks, width, height, sh_degree, splats=None, **kwargs):
        sp = splats if splats is not None else _render_fn._splats
        return rasterize_splats(sp, camtoworlds, Ks, width, height,
                                sh_degree=sh_degree, packed=cfg.packed,
                                near_plane=cfg.near_plane, far_plane=cfg.far_plane)

    _render_fn._splats = splats

    scene = GaussianScene.from_splats(splats, id="level0")
    splats = scene.splats
    stage  = Stage()
    stage.add_scene(scene, _render_fn)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps_per_level)
    )

    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    writer  = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb", "level0"))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    pbar = tqdm.tqdm(range(cfg.steps_per_level), desc="Level 0 (base, factor=8)")
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        renders, alphas, info = stage.render(
            scene.id,
            camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
            sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        colors = renders[..., :3]

        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        l1   = l1_loss(colors, pixels).mean()
        ssim = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = torch.lerp(l1, ssim, cfg.ssim_lambda)
        loss.backward()

        pbar.set_description(f"L0 loss={loss.item():.3f} #GS={len(splats['means']):,}")

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state,
            step=step, info=info, packed=cfg.packed, scene=scene,
        )
        info.pop("isect_ids", None)
        info.pop("flatten_ids", None)

        if step % 500 == 0:
            writer.add_scalar("L0/loss", loss.item(), step)
            writer.add_scalar("L0/num_GS", len(splats["means"]), step)

        if step in (6999, cfg.steps_per_level - 1):
            sp = {k: splats[k].detach() for k in splats}
            _eval_splats(sp, valset, device, cfg, psnr_m, ssim_m, lpips_m, tag=f"L0@{step}")
            del sp

    _save_full_ply({k: splats[k].detach().cpu() for k in splats}, leaves_ply)
    ply_mb = os.path.getsize(leaves_ply) / 1e6
    print(f"[Level 0] Saved {len(splats['means']):,} Gaussians → {leaves_ply} ({ply_mb:.1f} MB)")
    writer.close()
    return leaves_ply


# ---------------------------------------------------------------------------
# Levels 1-3: evolution training
# ---------------------------------------------------------------------------

def run_levelN(cfg: EvoGSConfig, level_idx: int, device: str) -> str:
    """Train evolution level `level_idx` (1, 2, or 3).

    Loads prior level's active leaves as starting splats. High-gradient leaves
    are split every `split_every` steps until `split_until`. Parents are removed
    from the active set; only leaf nodes are rasterized.

    Returns path to saved active-leaves PLY.
    """
    assert level_idx in (1, 2, 3), f"level_idx must be 1-3, got {level_idx}"

    level_dir = os.path.join(cfg.result_dir, "levels")
    os.makedirs(level_dir, exist_ok=True)
    leaves_ply = os.path.join(level_dir, f"level_{level_idx:02d}_leaves.ply")
    tree_dir   = os.path.join(level_dir, f"level_{level_idx:02d}_tree")

    if os.path.exists(leaves_ply):
        print(f"[Level {level_idx}] Already exists: {leaves_ply}")
        return leaves_ply

    factor   = cfg.data_factors[level_idx]
    prior_ply = os.path.join(level_dir, f"level_{level_idx-1:02d}_leaves.ply")
    print(f"\n{'─'*70}")
    print(f"[Level {level_idx}] Evolution training at factor={factor} ({cfg.steps_per_level} steps)")
    print(f"[Level {level_idx}] Starting from: {prior_ply}")
    print(f"{'─'*70}")

    # ── Load scene scale ──────────────────────────────────────────────────
    ss_path = os.path.join(cfg.result_dir, "scene_scale.json")
    if os.path.exists(ss_path):
        with open(ss_path) as _f:
            scene_scale = json.load(_f)["scene_scale"]
    else:
        p_tmp = Parser(cfg.data_dir, factor=cfg.data_factors[0], normalize=True, test_every=1)
        scene_scale = float(p_tmp.scene_scale) * 1.1

    # ── Load dataset ──────────────────────────────────────────────────────
    parser = Parser(
        data_dir=cfg.data_dir, factor=factor,
        normalize=cfg.normalize_world_space, test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")
    valset   = Dataset(parser, split="val")
    print(f"[Level {level_idx}] Train: {len(trainset)} imgs, Val: {len(valset)} imgs @ factor={factor}")

    # ── Load prior level leaves as starting splats ────────────────────────
    raw = _load_ply_as_splats(prior_ply, cfg, device)
    splats = _splats_dict_to_paramdict(raw, device)
    N_start = len(splats["means"])
    print(f"[Level {level_idx}] Starting leaves: {N_start:,}")

    # ── Optimizers ────────────────────────────────────────────────────────
    optimizers = _make_optimizers(splats, cfg, scene_scale)

    # ── LR scheduler (means position LR decays to 1% over training) ──────
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps_per_level)
    )

    # ── Evolution tree ────────────────────────────────────────────────────
    tree = EvolutionTree()
    leaf_ids = tree.register_roots(N_start).to(device)

    # ── Grad accumulation state ───────────────────────────────────────────
    evo_state = {
        "grad2d": torch.zeros(N_start, device=device),
        "count":  torch.zeros(N_start, device=device),
    }

    # ── Metrics + logging ─────────────────────────────────────────────────
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    writer  = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb", f"level{level_idx}"))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(cfg.steps_per_level),
                     desc=f"Level {level_idx} (evo, factor={factor})")
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # ── Forward: rasterize ONLY active leaves ─────────────────────────
        renders, alphas, info = rasterize_splats(
            splats, camtoworlds, Ks, width, height,
            sh_degree=sh_degree_to_use, packed=cfg.packed,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        colors = renders[..., :3]

        # Register retain_grad for 2D-gradient accumulation
        info["means2d"].retain_grad()

        # ── Loss ─────────────────────────────────────────────────────────
        l1   = l1_loss(colors, pixels).mean()
        ssim = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = torch.lerp(l1, ssim, cfg.ssim_lambda)
        loss.backward()

        # ── Accumulate 2D gradient norms (after backward) ─────────────────
        _accumulate_grad2d(evo_state, info, packed=cfg.packed)

        pbar.set_description(
            f"L{level_idx} loss={loss.item():.3f} #leaves={len(splats['means']):,}"
        )

        # ── Parameter update ──────────────────────────────────────────────
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        info.pop("isect_ids", None)
        info.pop("flatten_ids", None)

        # ── Evolution split events ─────────────────────────────────────────
        if step > 0 and step % cfg.split_every == 0 and step <= cfg.split_until:
            n_active = len(splats["means"])

            # Select top split_frac of leaves by average 2D gradient norm
            avg_grad = evo_state["grad2d"] / evo_state["count"].clamp_min(1)
            n_split = max(1, int(n_active * cfg.split_frac))
            n_split = min(n_split, n_active - 1)  # keep at least 1 leaf
            _, top_idx = avg_grad.topk(n_split)
            split_mask = torch.zeros(n_active, dtype=torch.bool, device=device)
            split_mask[top_idx] = True

            # EvoGS evolution split (symmetric: C1 = P + ψ, C2 = P - ψ)
            leaf_ids = tree.evo_split(split_mask, splats, optimizers, leaf_ids, level_idx)

            # Prune leaves that have become transparent
            is_prune = torch.sigmoid(splats["opacities"].flatten()) < cfg.opacity_prune_threshold
            if is_prune.any():
                gs_remove(splats, optimizers, {}, is_prune)
                leaf_ids = tree.update_ids_after_prune(~is_prune, leaf_ids)

            # Reset grad state for new leaf count
            n_new = len(splats["means"])
            evo_state["grad2d"] = torch.zeros(n_new, device=device)
            evo_state["count"]  = torch.zeros(n_new, device=device)

            n_pruned = is_prune.sum().item() if is_prune.any() else 0
            print(
                f"\n[Level {level_idx}] Step {step}: "
                f"split {split_mask.sum()} | pruned {n_pruned} | "
                f"active leaves: {n_new:,}"
            )
            writer.add_scalar(f"L{level_idx}/num_leaves", n_new, step)
            torch.cuda.empty_cache()

        if step % 500 == 0:
            writer.add_scalar(f"L{level_idx}/loss", loss.item(), step)

        if step in (6999, cfg.steps_per_level - 1):
            sp = {k: splats[k].detach() for k in splats}
            _eval_splats(sp, valset, device, cfg, psnr_m, ssim_m, lpips_m,
                         tag=f"L{level_idx}@{step}")
            del sp

    # ── Save active leaves PLY ────────────────────────────────────────────
    leaves_dict = {k: splats[k].detach().cpu() for k in splats}
    _save_full_ply(leaves_dict, leaves_ply)
    ply_mb = os.path.getsize(leaves_ply) / 1e6
    print(f"[Level {level_idx}] Active leaves: {len(splats['means']):,} → {leaves_ply} ({ply_mb:.1f} MB)")

    # ── Save tree topology + residuals ───────────────────────────────────
    tree.save(tree_dir, splats=splats, leaf_ids=leaf_ids.cpu())
    topo_mb = os.path.getsize(os.path.join(tree_dir, "topology.json")) / 1e6
    res_path = os.path.join(tree_dir, "residuals.npz")
    res_mb = os.path.getsize(res_path) / 1e6 if os.path.exists(res_path) else 0.0
    print(f"[Level {level_idx}] Tree: {len(tree.splits)} splits | "
          f"topology {topo_mb:.2f} MB | residuals {res_mb:.2f} MB")

    writer.close()
    return leaves_ply


# ---------------------------------------------------------------------------
# Levels 1-3: evolution training with the LEARNED representation (faithful)
# ---------------------------------------------------------------------------

def run_levelN_learned(cfg: EvoGSConfig, level_idx: int, device: str) -> str:
    """Evolution level with trainable collinear (ψ, α) children (faithful EvoGS).

    Differs from run_levelN only in the leaf parametrization: instead of two
    free children + post-hoc ψ/α, split pairs share a trainable ψ and learn a
    per-attribute α against a frozen parent snapshot. See evogs/learned_repr.py.
    """
    assert level_idx in (1, 2, 3), f"level_idx must be 1-3, got {level_idx}"

    level_dir = os.path.join(cfg.result_dir, "levels")
    os.makedirs(level_dir, exist_ok=True)
    leaves_ply = os.path.join(level_dir, f"level_{level_idx:02d}_leaves.ply")
    tree_dir   = os.path.join(level_dir, f"level_{level_idx:02d}_tree")

    if os.path.exists(leaves_ply):
        print(f"[Level {level_idx}] Already exists: {leaves_ply}")
        return leaves_ply

    factor    = cfg.data_factors[level_idx]
    prior_ply = os.path.join(level_dir, f"level_{level_idx-1:02d}_leaves.ply")
    mode = "asymmetric" if cfg.asymmetric else "symmetric"
    print(f"\n{'─'*70}")
    print(f"[Level {level_idx}] LEARNED-REPR ({mode}, freeze_inherited="
          f"{cfg.freeze_inherited}) at factor={factor} ({cfg.steps_per_level} steps)")
    print(f"[Level {level_idx}] Starting from: {prior_ply}")
    print(f"{'─'*70}")

    # ── Scene scale ───────────────────────────────────────────────────────
    ss_path = os.path.join(cfg.result_dir, "scene_scale.json")
    if os.path.exists(ss_path):
        with open(ss_path) as _f:
            scene_scale = json.load(_f)["scene_scale"]
    else:
        p_tmp = Parser(cfg.data_dir, factor=cfg.data_factors[0], normalize=True, test_every=1)
        scene_scale = float(p_tmp.scene_scale) * 1.1

    # ── Dataset ───────────────────────────────────────────────────────────
    parser = Parser(
        data_dir=cfg.data_dir, factor=factor,
        normalize=cfg.normalize_world_space, test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")
    valset   = Dataset(parser, split="val")
    print(f"[Level {level_idx}] Train: {len(trainset)} imgs, Val: {len(valset)} imgs @ factor={factor}")

    # ── Load prior leaves; build learned leaf set ─────────────────────────
    raw = _load_ply_as_splats(prior_ply, cfg, device)
    raw = {k: v.to(device) for k, v in raw.items()}
    lls = LearnedLeafSet(raw, device, asymmetric=cfg.asymmetric,
                         freeze_inherited=cfg.freeze_inherited)
    print(f"[Level {level_idx}] Starting leaves: {lls.N:,}")

    lls.set_lr_spec({
        "means":     cfg.means_lr * scene_scale,
        "scales":    cfg.scales_lr,
        "quats":     cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh0":       cfg.sh0_lr,
        "shN":       cfg.shN_lr,
        "log_alpha": cfg.log_alpha_lr,
    })

    evo_state = {
        "grad2d": torch.zeros(lls.N, device=device),
        "count":  torch.zeros(lls.N, device=device),
    }

    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    writer  = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb", f"level{level_idx}"))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    pbar = tqdm.tqdm(range(cfg.steps_per_level),
                     desc=f"Level {level_idx} (learned, factor={factor})")
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # ── Reconstruct active leaves, then rasterize ─────────────────────
        recon = lls.reconstruct()
        renders, alphas, info = rasterize_splats(
            recon, camtoworlds, Ks, width, height,
            sh_degree=sh_degree_to_use, packed=cfg.packed,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        colors = renders[..., :3]
        info["means2d"].retain_grad()

        l1   = l1_loss(colors, pixels).mean()
        ssim = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = torch.lerp(l1, ssim, cfg.ssim_lambda)
        loss.backward()

        _accumulate_grad2d(evo_state, info, packed=cfg.packed)

        pbar.set_description(
            f"L{level_idx} loss={loss.item():.3f} #leaves={lls.N:,}"
        )

        lls.step()
        lls.scale_means_lr(0.01 ** (step / cfg.steps_per_level))

        info.pop("isect_ids", None)
        info.pop("flatten_ids", None)

        # ── Evolution split events ────────────────────────────────────────
        if step > 0 and step % cfg.split_every == 0 and step <= cfg.split_until:
            n_active = lls.N
            avg_grad = evo_state["grad2d"] / evo_state["count"].clamp_min(1)
            n_split = max(1, int(n_active * cfg.split_frac))
            n_split = min(n_split, n_active - 1)
            _, top_idx = avg_grad.topk(n_split)
            split_mask = torch.zeros(n_active, dtype=torch.bool, device=device)
            split_mask[top_idx] = True

            lls.split(split_mask, level_idx)

            with torch.no_grad():
                opa = torch.sigmoid(lls.reconstruct()["opacities"].flatten())
            prune_mask = opa < cfg.opacity_prune_threshold
            n_pruned = lls.prune(prune_mask)

            evo_state["grad2d"] = torch.zeros(lls.N, device=device)
            evo_state["count"]  = torch.zeros(lls.N, device=device)
            print(f"\n[Level {level_idx}] Step {step}: split {int(split_mask.sum())} | "
                  f"pruned {n_pruned} | active leaves: {lls.N:,}")
            writer.add_scalar(f"L{level_idx}/num_leaves", lls.N, step)
            torch.cuda.empty_cache()

        if step % 500 == 0:
            writer.add_scalar(f"L{level_idx}/loss", loss.item(), step)
            if cfg.asymmetric:
                writer.add_scalar(f"L{level_idx}/alpha_mean",
                                  lls.alpha().mean().item(), step)

        if step in (6999, cfg.steps_per_level - 1):
            _eval_splats(lls.materialize(), valset, device, cfg,
                         psnr_m, ssim_m, lpips_m, tag=f"L{level_idx}@{step}")

    # ── Save materialized leaves + exact tree residuals ───────────────────
    leaves_dict = {k: v.cpu() for k, v in lls.materialize().items()}
    _save_full_ply(leaves_dict, leaves_ply)
    ply_mb = os.path.getsize(leaves_ply) / 1e6
    print(f"[Level {level_idx}] Active leaves: {lls.N:,} → {leaves_ply} ({ply_mb:.1f} MB)")

    info_tree = lls.save(tree_dir)
    topo_mb = os.path.getsize(os.path.join(tree_dir, "topology.json")) / 1e6
    res_path = os.path.join(tree_dir, "residuals.npz")
    res_mb = os.path.getsize(res_path) / 1e6 if os.path.exists(res_path) else 0.0
    print(f"[Level {level_idx}] Tree: {len(lls.splits)} splits | "
          f"{info_tree.get('n_residual_pairs', 0)} residual pairs | "
          f"topology {topo_mb:.2f} MB | residuals {res_mb:.2f} MB")

    writer.close()
    return leaves_ply


# ---------------------------------------------------------------------------
# Evaluation: per-level metrics at native resolution
# ---------------------------------------------------------------------------

def run_eval(cfg: EvoGSConfig, device: str):
    eval_dir = os.path.join(cfg.result_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    metrics_path = os.path.join(eval_dir, "metrics.json")
    level_dir    = os.path.join(cfg.result_dir, "levels")

    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    results = []
    print(f"\n{'─'*70}")
    print("[Eval] Per-level evaluation at native resolution")
    print(f"{'─'*70}")

    for level_idx in range(len(cfg.data_factors)):
        factor     = cfg.data_factors[level_idx]
        leaves_ply = os.path.join(level_dir, f"level_{level_idx:02d}_leaves.ply")

        if not os.path.exists(leaves_ply):
            print(f"[Eval] Level {level_idx}: PLY not found, skipping ({leaves_ply})")
            continue

        leaves_mb = os.path.getsize(leaves_ply) / 1e6

        # Tree storage cost (topology + residuals = what gets streamed)
        tree_dir  = os.path.join(level_dir, f"level_{level_idx:02d}_tree")
        topo_path = os.path.join(tree_dir, "topology.json")
        res_path  = os.path.join(tree_dir, "residuals.npz")
        tree_mb = 0.0
        if os.path.exists(topo_path):
            tree_mb += os.path.getsize(topo_path) / 1e6
        if os.path.exists(res_path):
            tree_mb += os.path.getsize(res_path) / 1e6

        raw = _load_ply_as_splats(leaves_ply, cfg, device)
        splats_eval = {k: v.to(device) for k, v in raw.items()}
        N_gs = splats_eval["means"].shape[0]

        parser = Parser(
            data_dir=cfg.data_dir, factor=factor,
            normalize=cfg.normalize_world_space, test_every=cfg.test_every,
        )
        valset = Dataset(parser, split="val")
        print(f"[Eval L{level_idx}] factor={factor}, {N_gs:,} active leaves, {len(valset)} test imgs")

        stats = _eval_splats(splats_eval, valset, device, cfg,
                             psnr_m, ssim_m, lpips_m, tag=f"L{level_idx}@factor{factor}")
        stats["level"]     = level_idx
        stats["factor"]    = factor
        stats["leaves_mb"] = round(leaves_mb, 2)
        stats["tree_mb"]   = round(tree_mb, 2)
        results.append(stats)

        del splats_eval

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Metrics saved → {metrics_path}")

    print(f"\n{'─'*90}")
    print(f"{'Level':>5} {'Factor':>6} {'#Leaves':>10} {'PSNR':>7} {'SSIM':>7} "
          f"{'LPIPS':>7} {'Leaves(MB)':>11} {'Tree(MB)':>9}")
    print("─" * 90)
    for r in results:
        print(f"{r['level']:>5} {r['factor']:>6}x {r['num_GS']:>10,} "
              f"{r['psnr']:>7.3f} {r['ssim']:>7.4f} {r['lpips']:>7.3f} "
              f"{r['leaves_mb']:>11.1f} {r['tree_mb']:>9.2f}")
    print("─" * 90)

    print("\nPaper reference (Garden, Mip-NeRF360, EvoGS vs LapisGS):")
    print("  EvoGS improves avg PSNR over LapisGS by ~0.31 dB at finest level")
    print("  EvoGS uncompressed total: ~347 MB vs LapisGS ~802 MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: EvoGSConfig):
    set_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.result_dir, "levels"), exist_ok=True)

    stages = set(cfg.stages)
    print(f"[EvoGS] Stages: {sorted(stages)}")
    print(f"[EvoGS] Data: {cfg.data_dir}")
    print(f"[EvoGS] Results: {cfg.result_dir}")
    print(f"[EvoGS] Resolution pyramid: {cfg.data_factors}")
    print(f"[EvoGS] Steps per level: {cfg.steps_per_level}")
    print(f"[EvoGS] Evolution splits: every {cfg.split_every} steps until {cfg.split_until}")
    print(f"[EvoGS] Split fraction: {cfg.split_frac:.1%} of active leaves per event")

    level_fn = run_levelN_learned if cfg.learned_repr else run_levelN
    if cfg.learned_repr:
        print(f"[EvoGS] Learned representation: asymmetric={cfg.asymmetric}, "
              f"freeze_inherited={cfg.freeze_inherited}")

    if "level0" in stages:
        run_level0(cfg, device)

    if "level1" in stages:
        level_fn(cfg, level_idx=1, device=device)

    if "level2" in stages:
        level_fn(cfg, level_idx=2, device=device)

    if "level3" in stages:
        level_fn(cfg, level_idx=3, device=device)

    if "eval" in stages:
        run_eval(cfg, device)


if __name__ == "__main__":
    cfg = tyro.cli(EvoGSConfig)
    main(cfg)
