# EvoGS Paper Summary (method extraction)

**Paper:** EvoGS: Constructing Continuous-Layered Gaussian Splatting with
Evolution Tree for Scalable 3D Streaming, arXiv 2606.07179.

> Distilled from the arXiv HTML. Verify exact equations/constants against the PDF
> before relying on them. Our implementation notes are in `evogs_progress.md`.

## 1. Core idea: continuous vs discrete layering
- **LapisGS (discrete):** independent splat sets per LoD; lower levels frozen
  while training higher → accumulated geometric error masked by redundant
  "ghost" splats (~66% transparent at L3).
- **EvoGS (continuous):** parent→child *lineage* tree; children structurally
  **refine/correct** ancestors via learned parameters → redundancy "from over
  65% to under 25%".

## 2. Evolution Tree representation (asymmetric collinear, "Option D")
Parent `P ∈ ℝ^D` (pos, rot, scale, opacity, SH) split into two children:
```
C1 = P + ψ
C2 = P − α ⊙ ψ
(C1 + C2)/2 = P + ½(1 − α)ψ          # α=1 ⇒ symmetric, mean preserved
```
- `ψ ∈ ℝ^D` learned refinement direction (Haar/JPEG2000-biorthogonal inspired);
  sparse: "<20% of coefficients carry >90% of energy".
- `α ∈ ℝ^A`, A=5 per-attribute asymmetry factors (one per param type).
- **Leaf reconstruction** (root→leaf accumulation):
  `S = P_root + Σ_{k=1..ℓ} s_k ⊙ ψ_k`, with `s_k ∈ {+1, −α_k}` = branch taken.
- **Active rendering:** client reconstructs current leaf set `ℱ_i`, rasterizes
  with the standard tile rasterizer. Truncating the tree at any depth = valid scene.

## 3. Training procedure
- N+1 levels (experiments use 4: L0..L3). Image pyramid by 8×/4×/2×/1×.
- **L0:** standard 3DGS on 8× images, 30k iters → roots (ψ=0, α=1) → leaf set ℱ0.
- **Li (i≥1):** train on less-downsampled images, 30k iters. Only frontier-leaf
  `ψ`/`α` trainable; ancestors frozen. Densify every 3k up to 15k.
- **Split criterion:** leaves whose accumulated 2D positional gradient exceeds a
  threshold split into two children → unbalanced, spatially-adaptive tree.
  `ℱ_{i+1} = (ℱ_i \ S_i) ∪ C(S_i)`. After a level converges, its frontier
  ψ/α are frozen.

## 4. Loss & hyperparameters
- Per level: `L_i = (1−λ) Σ_m L1 + λ Σ_m L_DSSIM`, **λ = 0.2**.
- 30k iters/level; densify every 3k up to 15k; otherwise "default 3DGS settings".
- A collinearity constraint keeps children along the ψ axis.

## 5. Storage & streaming
- Base layer: transmit root params `P_root`. Enhancement layer i: transmit only
  `ψ ∈ ℝ^D` + `α ∈ ℝ^5` per split leaf (D+5 params/split). ψ sparse → smooth
  progressive refinement, no ghost splats.
- **Storage ≠ rendering memory:** storage = cumulative payload incl. internal
  nodes; render footprint = materialized leaves only (e.g. DB L3 405 vs 215 MB).
- **Compression PoC:** 8-bit uniform scalar quant + zstd → L3 347.36 → 91.75 MB
  ("11.5% of LapisGS", −0.10 dB PSNR).

## 6. Datasets & eval protocol
- Blender, Mip-NeRF360, Tanks&Temples, Deep Blending.
- Pyramid L0 8× / L1 4× / L2 2× / L3 full. **Each level evaluated at its NATIVE
  pyramid resolution.** Metrics: PSNR, SSIM, LPIPS, Storage(MB), Mem(MB).

## 7. Key results
Averaged over 4 datasets at L3:
| Method | PSNR | SSIM | LPIPS | Storage | Mem |
|--------|-----:|-----:|------:|--------:|----:|
| EvoGS  | 27.73 | 0.964 | 0.047 | 347.36 | 215.43 |
| LapisGS| 27.54 | 0.962 | 0.050 | 801.80 | 801.80 |
| L3GS   | 27.64 | 0.958 | 0.044 | — | — |

Per-dataset L3 Δ over best baseline: Blender +0.19 dB / −53% stor / −82% mem;
Mip360 +0.03 / −10.7% / −43%; T&T +0.00 / −18% / −66%; DeepBlending +0.13 / −39% / −68%.

**Symmetric vs Asymmetric (Table 2, avg):** asymmetric α recovers ≈+1 dB at every
level (L0 +0.82, L1 +0.91, L2 +0.99, L3 +1.01) for +0.83% storage. → Learned α is
worth implementing.

**Redundancy (Table 3, % transparent):** LapisGS L0→L3 15.9/46.9/55.6/65.8;
EvoGS-asym 12.1/17.0/21.0/24.9.

## 8. Implications for our implementation
Our code (`evogs/`) reproduces the **symmetric** variant and already beats our
LapisGS baseline. The biggest open item is **trainable asymmetric α** (the paper's
headline, +1 dB). Then: zstd+8-bit compression pipeline, and more datasets.
See `evogs_progress.md` for the existing implementation + design choices.
