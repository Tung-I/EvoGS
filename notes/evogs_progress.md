# EvoGS Implementation Progress

**Date:** 2026-06-26  
**Scene:** Garden (Mip-NeRF 360)  
**Status:** STOPPED — level 0 interrupted mid-run (~step 7K/30K); no completed checkpoints yet

---

## What is EvoGS

EvoGS (arXiv:2606.07179) is a scalable 3DGS streaming method that uses a binary **Evolution Tree** instead of additive layers (like LapisGS). The key distinction:

- **LapisGS:** `G_i = G_{i-1} ∪ ΔG_i` — each level adds new Gaussians, old ones remain.
- **EvoGS:** `F_i = (F_{i-1} \ S_i) ∪ C(S_i)` — selected leaves are **replaced** by two children.

Children are initialized as: `C1 = P + ψ`, `C2 = P - α⊙ψ` where ψ is a refinement direction and α is a per-group asymmetry scalar. Only **active leaves** are rasterized (parents removed). Residuals (ψ, α) are compact and compressible.

**Paper reported results (Garden, Mip-NeRF360):** EvoGS improves avg PSNR by ~0.31 dB over LapisGS at finest level; storage 347 MB uncompressed vs. LapisGS 802 MB.

---

## Implementation Location

All EvoGS code lives in `/home/tungichen_umass_edu/lctvgs-copy/evogs/`:

```
lctvgs-copy/
├── evogs/
│   ├── __init__.py              # package marker
│   ├── evolution_tree.py        # EvolutionTree class (splits, residuals, serialization)
│   ├── train_evogs.py           # main trainer (entry point)
│   └── eval_evogs.py            # standalone evaluator
└── train_garden_evogs.sh        # run script (sets env, launches training)
```

**Key paths:**
- Dataset: `/work/pi_rsitaram_umass_edu/tungi/datasets/garden`
- Results: `/home/tungichen_umass_edu/lctvgs-copy/results/garden_evogs/`
- Log: `/home/tungichen_umass_edu/lctvgs-copy/logs/garden_evogs.log`

---

## Training Pipeline

4 levels, each 30K steps, image pyramid 1/8 → 1/4 → 1/2 → full resolution:

| Level | Factor | Type | Notes |
|-------|--------|------|-------|
| 0 | 1/8 | Standard 3DGS (DefaultStrategy) | SFM init, builds roots |
| 1 | 1/4 | EvoGS evolution | Splits every 3K steps until step 15K |
| 2 | 1/2 | EvoGS evolution | Same schedule |
| 3 | 1× | EvoGS evolution | Full resolution, final model |

**Evolution split schedule:** 5 events per level at steps 3K/6K/9K/12K/15K. Each event:
1. Select top 1% of active leaves by avg 2D gradient norm
2. Split each: C1 = P + ψ, C2 = P - ψ (symmetric symmetric init, α=1)
3. Prune leaves with opacity < 0.005
4. Reset gradient accumulators

No opacity reset during evolution levels (`reset_every` effectively disabled).

---

## File Contents

### `evolution_tree.py`

**`EvolutionTree` class:**
- `register_roots(n)` → `leaf_ids` tensor — assigns IDs to initial leaves
- `evo_split(mask, splats, optimizers, leaf_ids, level)` → new `leaf_ids` — performs symmetric split, records topology, stores parent snapshots for residual extraction. Calls `gsplat.strategy.ops._update_param_with_optimizer` for in-place optimizer management.
- `update_ids_after_prune(keep_mask, leaf_ids)` → new `leaf_ids`
- `extract_residuals(splats, leaf_ids)` → list of dicts with `psi`, `alpha` per surviving split
- `save(tree_dir, splats, leaf_ids)` → writes `topology.json` + `residuals.npz`

**Split initialization** (in `evo_split`):
```python
# Rotation-aligned perturbation (similar to 3DGS split but symmetric)
psi_means = R @ (scales * unit_dir)   # one direction per parent
C1.means = P.means + psi_means
C2.means = P.means - psi_means        # symmetric (α=1 init)
C1.scales = C2.scales = log(exp(P.scales) / 1.6)   # shrink
C1.opacities = C2.opacities = P.opacities - log(2)  # halve
C1.sh* = C2.sh* = P.sh*              # copy appearance
```

**Post-hoc residual extraction:** After training, for each split where both children survived:
```
ψ_k = C1_k - P_k   (element-wise, per param group)
α_k = -(C2_k - P_k) / ψ_k   (clamped to [0.01, 10])
```
Saved as `residuals.npz` with `psi` shape `[n_splits, D]` and `alpha` shape `[n_splits, 6]`.

### `train_evogs.py`

Entry point. Config: `EvoGSConfig` dataclass (tyro CLI).

Key functions:
- `run_level0(cfg, device)` — exact copy of LapisGS `run_layer0` (uses `GaussianScene`+`Stage`+`DefaultStrategy`), saves `level_00_leaves.ply`
- `run_levelN(cfg, level_idx, device)` — evolution training: loads prior PLY, runs evo splits, saves `level_{N:02d}_leaves.ply` + `level_{N:02d}_tree/` (topology + residuals)
- `run_eval(cfg, device)` — per-level PSNR/SSIM/LPIPS at native resolution + storage sizes
- `_accumulate_grad2d(evo_state, info, packed)` — standalone gradient accumulator (mirrors DefaultStrategy._update_state)

Shared helpers copied verbatim from `lapisgs/train_lapisgs.py`:
`rasterize_splats`, `_make_optimizers`, `_init_splats_from_parser`, `_load_ply_as_splats`, `_save_full_ply`, `_eval_splats`

### `eval_evogs.py`

Standalone evaluator. Run to re-evaluate saved checkpoints without re-training:
```bash
python evogs/eval_evogs.py \
    --data-dir /work/pi_rsitaram_umass_edu/tungi/datasets/garden \
    --result-dir results/garden_evogs \
    --levels 0 1 2 3
```

---

## Training Status — COMPLETE (2026-06-26)

All 4 levels trained and evaluated successfully. PID 689932, interactive L40S session.

| Level | Factor | #GS | PSNR | SSIM | LPIPS | Leaves MB | Tree MB | Status |
|-------|--------|-----|------|------|-------|-----------|---------|--------|
| 0 | 8× | 2.84M | 29.63 | 0.922 | 0.032 | 669.7 | — | ✅ |
| 1 | 4× | 2.63M | 27.83 | 0.868 | 0.075 | 620.0 | 35.0 | ✅ |
| 2 | 2× | 2.61M | 26.74 | 0.813 | 0.152 | 615.4 | 34.9 | ✅ |
| 3 | 1× | 2.61M | 26.43 | 0.795 | 0.238 | 615.5 | 34.9 | ✅ |
| eval | — | — | — | — | — | — | — | ✅ |

**Fix applied before run:** `evogs/scripts/train_garden_evogs.sh` had `cd "$WORK_DIR/lctvgs-copy"` (directory did not exist); changed to `cd "$WORK_DIR/lctvgs"`.

**Split event pruning pattern (converges cleanly each level):**
- L1: 262K → 29K → 22K → 16K → 12K pruned per event
- L2: 71K → 30K → 24K → 15K → 8K
- L3: 37K → 36K → 29K → 17K → 8K

---

## How to Resume / Re-Launch

The trainer is **idempotent** — if a level's PLY already exists it skips that level:

```bash
cd /home/tungichen_umass_edu/lctvgs-copy
bash train_garden_evogs.sh          # runs all missing levels
```

Or run specific stages:
```bash
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lctvgs/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/tungichen_umass_edu/lctvgs-copy
python evogs/train_evogs.py \
    --data-dir "$WORK_DIR/datasets/garden" \
    --result-dir results/garden_evogs \
    --stages level2 level3 eval        # only missing stages
```

---

## Bugs Fixed During Development

1. **numpy scalar in residual extraction:** `splats["opacities"][i]` yields a 0-D numpy array after `.detach().cpu().numpy()` indexing — not an `np.ndarray`. Fix: store parent snapshots with `np.atleast_1d(...).astype(np.float32)` in `evo_split()`, and `.flatten()` all tensors in `extract_residuals()` before arithmetic.

---

## Design Choices (Underspecified in Paper)

| Decision | Choice |
|---|---|
| Training parameterization | Full leaf params as dense tensors; residuals extracted post-hoc |
| Scale/opacity domains | Log / logit (same as gsplat standard) |
| Quaternion after split | Normalized with `F.normalize` |
| α initialization | 1.0 (symmetric); paper's symmetric ablation as baseline |
| α post-training bounds | Clamped to [0.01, 10] in `extract_residuals` |
| Split fraction per event | 1% of active leaves |
| Pruning | Opacity < 0.005 at each split event (same as DefaultStrategy) |
| Opacity reset | Disabled (reset_every=100_000) |

---

## Expected Results

From paper (Garden, Mip-NeRF360):
- Level 3 (full res): PSNR ~0.31 dB above LapisGS L3
- Active leaves PLY: smaller than LapisGS cumulative (parents removed)
- Residuals NPZ: very compact (sparse ψ relative to parent)

LapisGS Garden results for comparison: `results/garden_lapisgs/eval/metrics.json`
