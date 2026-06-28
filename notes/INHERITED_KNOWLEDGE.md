# Inherited Knowledge — from the LapisGS reproduction (`../lctvgs`)

EvoGS was developed *inside* `../lctvgs` alongside LapisGS and shares its gsplat
backend and many helpers (`evogs/train_evogs.py` copied `rasterize_splats`,
`_make_optimizers`, `_init_splats_from_parser`, `_load_ply_as_splats`,
`_save_full_ply`, `_eval_splats` from `lapisgs/train_lapisgs.py`). So the LapisGS
lessons transfer *directly* here — more so than for the ProGS sibling repo.

## 0. The single most important lesson
**Establish a ground-truth reference BEFORE you optimize.** LapisGS burned four
iterations (v1–v4) chasing a misattributed "paper target" curve at the wrong
resolution (~10× too many pixels) before running the official code to get true
per-scene, native-resolution numbers. For EvoGS: the apples-to-apples baseline
is **our own LapisGS** (same gsplat backend, same pyramid), at
`../lctvgs/results/garden_lapisgs_v5/`. Compare against that, not against
absolute paper numbers from a different backend.

## 1. Evaluation protocol must match EXACTLY
- **Native per-level resolution.** EvoGS (like LapisGS) evaluates each level at
  its training pyramid resolution (L0 8× / L1 4× / L2 2× / L3 full). SSIM
  *decreases* L0→L3 because finer images are harder — that is correct, not a bug.
- **Resolution caveat:** the LapisGS v5 fix was to use the Mip-NeRF360 cap (finest
  = 1600px, then ÷2/4/8 = 200/400/800/1600px) via dedicated `*_lapis` datasets,
  NOT the raw full÷{8,4,2,1} pyramid (648/1297/2594/5187px). EvoGS results here
  were produced on `datasets/garden` (the raw pyramid). **If you compare EvoGS
  to the LapisGS v5 numbers, re-run EvoGS on the matched `garden_lapis` dataset**
  or you are comparing different resolutions. (Our current EvoGS-vs-LapisGS
  comparison used the older raw-pyramid LapisGS, so it is internally consistent.)
- Test split = every 8th image (llffhold=8). PSNR is the robust cross-impl
  metric; LPIPS net differs (Alex vs VGG) → same-impl only.
- **In-training PSNR can be bogus** (LapisGS official printed ~14 dB for enh
  layers while saved models rendered fine). Trust the standalone eval, not the log.

## 2. Densification / split mechanics (shared gsplat backend)
- EvoGS splits selected leaves and **removes the parent** (`ℱ_{i+1} =
  (ℱ_i \ S_i) ∪ C(S_i)`). This is DIFFERENT from LapisGS where anchors are kept.
  The known LapisGS bug — spawned children parked at a near-prune-floor opacity
  so they never prune yet stay invisible — does not apply identically, but watch
  the analogous failure: children initialized so faint/redundant they don't help.
  Our split halves opacity (`P.opacity − log 2`) and shrinks scale (÷1.6), like
  standard 3DGS split. Pruning at opacity < 0.005 per event converges cleanly.
- gsplat in packed mode counts the densification gradient per (camera,Gaussian)
  pair = per frame; matches inria-3DGS per-frame. EvoGS's split criterion uses an
  accumulated 2D-gradient threshold — `_accumulate_grad2d` mirrors
  `DefaultStrategy._update_state`.
- **Opacity reset is disabled** during evolution levels (`reset_every=100_000`),
  same convention as LapisGS block/anchor training.

## 3. Known bug already fixed (do NOT reintroduce)
**numpy scalar in residual extraction:** indexing `splats[...][i]` after
`.detach().cpu().numpy()` yields a 0-D array. Fix: store parent snapshots with
`np.atleast_1d(...).astype(np.float32)` in `evo_split()`, and `.flatten()` all
tensors in `extract_residuals()` before arithmetic. (See `evogs_progress.md`.)

## 4. Cluster / ops notes (same environment)
- SLURM, L40S (46 GB). EvoGS runs in the shared `$WORK/conda/envs/gsplat` env —
  no extra extensions (it's a pure gsplat method). Env block in `CLAUDE.md`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids OOM fragmentation.
- Datasets in `$WORK/datasets/`; `garden`/`counter` staged. EvoGS paper also uses
  Blender / Tanks&Temples / Deep Blending (need staging).
- SfM init is critical (empty points3D → catastrophic quality); verify non-empty.
- COLMAP binary does not support `--Mapper.min_num_inliers`.

## 5. Repo / git hygiene (gsplat submodule)
See `notes/git_commit_workflow.txt`. Two traps that bit us in lctvgs:
1. Stray top-level PLYs (`merged.ply`, etc.) escape `results/**/ply/`; the
   `.gitignore` here adds `results/**/*.ply` + `levels/` + `*.npz` catch-alls.
   Size-check untracked files before `git add -A`.
2. The submodule SHA must be pushed to its own remote (`gsplat-lctvgs`, the
   `lctvgs` remote inside the submodule — NOT `origin`=nerfstudio) before pinning.

## 6. Cross-references in `../lctvgs`
- `documentation/lapisgs_reproduction.md` — LapisGS-on-gsplat mapping (parent code)
- `notes/lapisgs_official_reference.md` — official ground-truth protocol
- `results/garden_lapisgs_v5/eval/metrics.json` — LapisGS comparison numbers
- `notes/status_2026_06_26.md` — multi-method status snapshot (incl. EvoGS table)
