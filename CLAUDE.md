# CLAUDE.md — EvoGS

This file guides Claude Code when working in the **EvoGS** repo. These
instructions OVERRIDE default behavior; follow them exactly.

## Project Goal

Implement & reproduce **EvoGS: Constructing Continuous-Layered Gaussian
Splatting with Evolution Tree for Scalable 3D Streaming** (arXiv 2606.07179),
using **gsplat** as the 3DGS backend.

EvoGS is a *continuous-layering* streaming representation: instead of LapisGS's
discrete additive layers, it organizes splats into a binary **Evolution Tree**
where a parent leaf `P` is *replaced* by two children along a learned
wavelet-like refinement direction `ψ`:
```
C1 = P + ψ ,   C2 = P − α ⊙ ψ        (α = per-attribute asymmetry, A=5 groups)
leaf params  S = P_root + Σ_k s_k ⊙ ψ_k ,  s_k ∈ {+1, −α_k}
```
Only **active leaves** are rasterized (parents removed) → far less redundancy
(~25% transparent splats vs LapisGS ~66%) and smaller storage (paper: L3 347 MB
vs LapisGS 802 MB; 91.75 MB after zstd, ~11.5× smaller than LapisGS).

**This work was already started in the sibling project `../lctvgs` and is
SEPARATED here.** Read these in order before doing anything:
1. `notes/evogs_progress.md` — full implementation log + completed Garden results
2. `notes/evogs_paper_summary.md` — clean method extraction from the paper
3. `notes/INHERITED_KNOWLEDGE.md` — transferable lessons from the LapisGS repro
4. `progress_notes.txt` — running log (start of this repo's history)

## Current status (carried over from ../lctvgs, 2026-06-26..28)
- **Code complete & runnable**: `evogs/{evolution_tree,train_evogs,eval_evogs}.py`.
- **Garden (Mip-NeRF360) DONE** — 4 levels trained + evaluated, native-factor:
  L0 29.63/0.922, L1 27.83/0.868, L2 26.74/0.813, L3 26.43/0.795 (PSNR/SSIM).
  Beats our LapisGS by avg +0.82 dB on L1–L3. Artifacts in
  `results/garden_evogs/` (level PLYs + per-level `*_tree/{topology.json,
  residuals.npz}` + `eval/metrics.json`).
- **KNOWN GAP vs paper (clear next step):** our implementation uses the
  **symmetric** split (α=1 init, α extracted post-hoc). The paper's headline
  result uses **learned asymmetric α during training**, which recovers ≈+1 dB at
  every level (Table 2). Implementing trainable α is the highest-value next task.
- Other next steps: more scenes (Blender / Tanks&Temples / Deep Blending),
  and the zstd+8-bit-quant compression pipeline for the storage numbers.

## Environment Setup (SLURM, L40S GPUs — same cluster as ../lctvgs)
EvoGS runs in the shared **gsplat** env (it IS a gsplat-backend method; unlike
the ProGS sibling repo, no extra Scaffold-GS/HAC extensions are needed). Use
this env block at the top of run scripts:
```bash
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/EvoGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Submit with `sbatch`; check `squeue -u tungichen_umass_edu`. Long runs in
`../lctvgs` were often done in interactive L40S sessions with `nohup ... &`.

## How to run / resume
The trainer is **idempotent** (skips a level whose `level_NN_leaves.ply` exists):
```bash
bash evogs/scripts/train_garden_evogs.sh            # all missing levels + eval
# or specific stages:
python evogs/train_evogs.py --data-dir $WORK_DIR/datasets/garden \
    --result-dir results/garden_evogs --stages level2 level3 eval
# re-evaluate saved checkpoints without retraining:
python evogs/eval_evogs.py --data-dir $WORK_DIR/datasets/garden \
    --result-dir results/garden_evogs --levels 0 1 2 3
```

## Key Paths
- **Repo**: `/work/pi_rsitaram_umass_edu/tungi/EvoGS/`
- **gsplat submodule**: `gsplat/` pinned to `fca0873` (Tung-I/gsplat-lctvgs) —
  the training backend (`gsplat/examples/simple_trainer.py` for vanilla baseline)
- **Sibling knowledge source**: `../lctvgs/` (LapisGS/EvoGS/CityGS); its
  `lapisgs/train_lapisgs.py` is the parent of `evogs/train_evogs.py` (shared
  helpers were copied verbatim).
- **Datasets**: `/work/pi_rsitaram_umass_edu/tungi/datasets/` (`garden`, `counter`
  staged; Blender / T&T / Deep Blending need staging)
- **Results**: `results/{scene}_evogs/`

## Paper Targets (averaged across 4 datasets, L3, asymmetric)
PSNR 27.73 / SSIM 0.964 / LPIPS 0.047 / Storage 347.36 MB / Mem 215.43 MB;
vs LapisGS 27.54 / 0.962 / 0.050 / 801.80 / 801.80. Eval each level at its
**native pyramid resolution** (L0 8×, L1 4×, L2 2×, L3 full).

## Working Conventions (inherited — see INHERITED_KNOWLEDGE.md)
1. Establish a ground-truth reference BEFORE optimizing (cost LapisGS 4 iters).
2. Match the eval protocol exactly (native per-level resolution; PSNR is the
   robust cross-impl metric; LPIPS net differs Alex vs VGG).
3. Commit hygiene with the gsplat submodule — see `notes/git_commit_workflow.txt`
   (stray top-level PLYs + submodule-SHA-must-be-pushed traps).
4. Keep `progress_notes.txt` and `notes/` current so the next session resumes cold.

## Memory
Persist non-obvious project facts to file-based memory; keep notes current.
