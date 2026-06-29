# Learned Representation — Findings (asymmetric α vs symmetric vs skeleton)

**Date:** 2026-06-29
**Scene:** Garden (Mip-NeRF360), raw pyramid (L0 8× … L3 full = 5187px).
**Code:** `evogs/learned_repr.py` (faithful collinear ψ + trainable α) wired into
`evogs/train_evogs.py` via `--learned-repr/--asymmetric/--freeze-inherited`.

## TL;DR
The faithful learned representation **matches** the free-training skeleton at a
**smaller footprint with exact residuals**, but in our **relaxed** training
regime the learned per-attribute **α is a no-op** — it does *not* reproduce the
paper's headline asym-over-sym gain (Table 2: +0.8…+1.0 dB). This is the
"doesn't beat the old one" case; causes + the follow-up (strict regime) below.

## What we built (faithful vs skeleton)
- Skeleton: split → two *free* children; ψ/α back-solved post-hoc.
- Faithful: split children share a **trainable ψ** and learn **α∈ℝ⁵** against a
  **frozen parent snapshot** (`C1=base+ψ`, `C2=base−α⊙ψ`), exact collinearity;
  residuals are stored params (exact, not least-squares).
- "Ancestors frozen" = internal/split nodes (auto via detached `base`). Frontier
  leaves trainable. `freeze_inherited=False` = **relaxed** (all frontier leaves
  adapt); `True` = **strict** (refinement only through split ψ).

## Result 1 — relaxed regime A/B (native per-level resolution)
| Level | asym PSNR | sym PSNR | skeleton PSNR | asym−sym |
|------:|----------:|---------:|--------------:|---------:|
| L1 f4 | 27.787 | 27.796 | 27.83 | −0.009 |
| L2 f2 | 26.761 | 26.752 | 26.74 | +0.009 |
| L3 f1 | 26.486 | 26.466 | 26.46 | +0.020 |

- asym ≈ sym ≈ skeleton at **every** level (|Δ| ≤ 0.02 dB).
- Footprint: asym uses ~4% fewer Gaussians (L3 2.498M vs 2.607M) and a smaller
  model (589.6 vs 615.5 MB) than the skeleton, with **exact** 25 MB residuals.

**Interpretation (why α doesn't help here):** with `freeze_inherited=False`,
all frontier leaves are free to adapt, so the model already reaches the
free-training quality ceiling. The split asymmetry α then has **no headroom** —
both asym and sym converge to the same optimum that free training finds. α only
changes the *parametrization* of an already-saturated fit, not the attainable
quality.

## Result 2 — compression PoC (asym L3, 8-bit + zstd, means 16-bit)
- Render footprint 589.6 → **100.7 MB** (5.9×) at **ΔPSNR −0.22 dB** — matches
  the paper's shape (L3 ~91.75 MB, −0.10 dB).
- Pitfall found & fixed: 8-bit-quantizing *absolute* means at scene scale gave
  −11.95 dB (≈extent/256 positional error). Fix: 16-bit means (negligible size).

## Hypothesis for the missing α advantage → strict regime
The paper's asym-over-sym gain most likely requires the **constrained** setting
where refinement happens *only* through split children (ancestors AND inherited
frontier leaves frozen), so the children's asymmetry is the dominant degree of
freedom. Under that constraint sym should *underfit* and asym should *recover*
~1 dB. We are testing this with `--freeze-inherited` (asym_strict vs sym_strict).
A secondary suspect is the **split budget**: 1% top-k per event refines too few
leaves for the strict regime to fit; the paper uses a gradient *threshold*
(potentially many more splits). [strict results appended below once available.]

## Strict regime A/B (freeze_inherited=True)
[pending — appended after runs complete]
