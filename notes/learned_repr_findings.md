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

## Strict regime A/B (freeze_inherited=True) — RESULT (2026-06-29)
Inherited frontier leaves frozen; refinement flows ONLY through split children,
so α is the dominant refinement DOF. Garden, native per-level resolution.

| Level | factor | asym PSNR | sym PSNR | asym−sym | asym SSIM | sym SSIM |
|------:|:------:|----------:|---------:|---------:|----------:|---------:|
| L1 | 4× | 25.793 | 25.675 | **+0.118** | 0.7745 | 0.7709 |
| L2 | 2× | 23.975 | 23.842 | **+0.134** | 0.6372 | 0.6310 |
| L3 | 1× | 23.584 | 23.425 | **+0.159** | 0.6412 | 0.6358 |

**Two findings:**
1. **Sign confirmed** — under strict freezing, asym beats sym at *every* level and
   the gap **grows with depth** (+0.12→+0.16 dB), exactly as predicted (deeper
   levels have more split children whose asymmetry is the only DOF). This is the
   qualitative confirmation the relaxed regime couldn't give (α flat no-op there).
2. **Magnitude ~6× too small** (+0.16 vs paper's +0.8…1.0 dB) and **both strict
   variants underfit badly** — L3 23.4–23.6 vs relaxed/skeleton ~26.5 (~3 dB
   deficit). Strict freezing *starves* the model of refinement capacity.

**Conclusion:** α is real and correctly signed, but the binding constraint on
reproducing the paper's headline is the **split mechanism, not the α
parametrization**. Our split budget (1% top-k per event) creates far too few
split children vs the paper's gradient *threshold*; α can only help to the extent
children exist to be made asymmetric. Relaxed regime saturates (α no-op); strict
regime starves (α helps a little but everything underfits). Highest-value next
step to close the gap: replace the top-k split budget with a gradient-threshold
split criterion (likely many more children per event).

### Compression — strict L3 (8-bit + zstd, means 16-bit)
| Variant | render footprint | →q8+zstd | ratio | ΔPSNR | streamed raw→zstd |
|--------:|-----------------:|---------:|------:|------:|------------------:|
| asym_strict | 710.0 MB | 86.8 MB | 8.2× | −0.36 dB | 742.5 → 151.1 MB |
| sym_strict  | 709.4 MB | 135.4 MB | 5.2× | −0.02 dB | 739.1 → 149.8 MB |

asym's leaves compress markedly better (86.8 vs 135.4 MB) at a larger ΔPSNR
(−0.36 vs −0.02); both match the paper's compression *shape*. Streamed storage
(root + ψ/α residuals) is ~150 MB either way. Files: each dir's
`eval/compress_L3.json`.

## Gradient-threshold split criterion (2026-06-30) — RESULT
Tested the prior section's "highest-value next step": replace the 1% top-k
per-event split budget with a gradient *threshold* (paper-style densification).
Code: `--split-mode threshold --split-grad-threshold τ --split-max-frac` driving
`_select_split_mask()` in `train_evogs.py` (topk path unchanged → all prior
results reproduce). τ=2e-4 (standard 3DGS `grow_grad2d`) calibrated on garden L1
step-3000 grad quantiles (between p95 1.6e-4 and p99 2.9e-4). Relaxed asymmetric,
garden native res. Arm: `results/garden_evogs_asym_thr/` vs top-k baseline
`results/garden_evogs_asym/`.

**Split behaviour:** threshold makes far more children than top-k, decaying per
event as the fit improves. L3 split counts 142.5k→99.2k→64.0k→41.1k→26.4k
(steps 3k–15k); the `--split-max-frac 0.05` cap **binds at the first L3 event**
(142.5k ≈ 5% of 2.87M) — i.e. at deep levels the cap, not τ, governs. Final leaf
counts vs top-k: L1 2.686M vs 2.559M (+5%), L2 2.869M, **L3 3.053M vs 2.498M
(+22%)**.

**Quality (native per-level PSNR), threshold vs top-k:**
| Level | top-k | threshold | Δ |
|------:|------:|----------:|--:|
| L1 | 27.787 | 27.814 | +0.027 |
| L2 | 26.761 | 26.819 | +0.058 |
| L3 | 26.486 | 26.477 | **−0.009** |

**Rate-distortion (8-bit+zstd, 16-bit means):**
| Level | arm | leaves | PSNR | streamed MB | render q8+zstd MB | decode ΔPSNR |
|------:|:----|-------:|-----:|------------:|------------------:|-------------:|
| L1 | top-k | 2.56M | 27.787 | 143.1 | 108.4 | −0.127 |
| L1 | thr   | 2.69M | 27.814 | 148.5 | 103.0 | −0.234 |
| L2 | top-k | 2.79M | 26.761 | 148.3 | 101.1 | −0.247 |
| L2 | thr   | 2.87M | 26.819 | 163.7 | 106.8 | −0.266 |
| L3 | top-k | 2.50M | 26.486 | **153.4** | 100.7 | **−0.222** |
| L3 | thr   | 3.05M | 26.477 | **178.2** | 114.8 | **−1.312** |

**Verdict — threshold is strictly dominated by top-k on the RD frontier in the
relaxed regime.** At L3 the threshold spends +22% leaves and +16% streamed
storage (178 vs 153 MB) for −0.009 dB of PSNR, and the extra marginal children
also quantize much worse (decode ΔPSNR −1.31 vs −0.22 dB; suspected attribute-
range outliers from aggressive splitting widening the 8-bit per-channel buckets —
worth a per-channel breakdown but does not change the conclusion). This is the
direct confirmation the relaxed-vs-strict analysis predicted: **relaxed training
is saturated, so adding children buys no quality and only hurts rate-distortion.**
The split criterion alone is therefore NOT the lever that recovers the paper's
+0.8…1.0 dB headline — the missing ingredient is the constrained (strict)
refinement setting in which those extra children's asymmetry becomes the dominant
DOF. The likely path to the paper number is **threshold split *combined with*
strict freezing** (more children AND headroom for α), not either alone. Files:
`results/garden_evogs_asym_thr/eval/{metrics,compress_L1,compress_L2,compress_L3}.json`.

## garden_lapis apples-to-apples vs LapisGS v5 (2026-06-30) — RESULT
**This reverses the raw-garden threshold no-op above.** All prior EvoGS numbers
were on the raw `garden` pyramid (648/1297/2594/5187px); our LapisGS v5 reference
used the Mip-NeRF360-capped `garden_lapis` pyramid (200/400/800/1600px). Re-ran
EvoGS on `garden_lapis` (relaxed asym) in two arms — top-k and threshold
(τ=2e-4, cap 0.05) — to compare at **matched resolution**. LapisGS v5 reference:
`../lctvgs/results/garden_lapisgs_v5/eval/metrics.json`.

| Level | top-k PSNR | thr PSNR | LapisGS PSNR | top-k #GS | thr #GS | LapisGS #GS |
|------:|-----------:|---------:|-------------:|----------:|--------:|------------:|
| L0 | 32.242 | 32.298 | 32.256 | 813k | 816k | 808k |
| L1 | 30.845 | 30.936 | 30.806 | 669k | 771k | 2.14M |
| L2 | 28.682 | 28.885 | 29.024 | 666k | 920k | 4.66M |
| L3 | 26.495 | 26.891 | 26.971 | 683k | 1.13M | 7.7M |

**thr − top-k:** L1 +0.091, L2 +0.203, **L3 +0.396** — the threshold gain *grows
with depth*, the opposite of raw garden (no-op). **thr − LapisGS:** L1 **+0.130**,
L2 −0.139, **L3 −0.080** — threshold EvoGS essentially matches LapisGS at L3 while
using **6.8× fewer Gaussians** (1.13M vs 7.7M), and beats it at L1 at 2.8× fewer.

**Reconciliation (key insight):** the lever is not resolution per se but whether
the model is **starved vs saturated**. On raw garden the top-k baseline already
held ~2.5M leaves at full res (near-saturated) → extra children bought nothing and
hurt RD. On the matched pyramid the top-k baseline is **under-densified** — leaf
count is nearly *flat* across levels (669k→666k→683k) because the 1% top-k budget
can't keep up with the LapisGS-style aggressive pyramid — so the threshold's extra
children (which *do* grow with depth: 771k→920k→1.13M) have real headroom and
recover +0.2…0.4 dB at L2/L3. **Takeaway:** the gradient-threshold split is the
right criterion when the budget would otherwise starve the model; on already-
saturated fits it only worsens RD. Files:
`results/garden_lapis_evogs_{asym,asym_thr}/eval/metrics.json`.

### Storage — garden_lapis L3 (8-bit + zstd, 16-bit means)
| Arm | #GS | PSNR full→q8 | ΔPSNR | streamed zstd | leaves PLY→q8+zstd |
|----:|----:|:------------:|------:|--------------:|-------------------:|
| top-k | 683k | 26.495→26.161 | −0.334 | 43.4 MB | 161.2 → 28.7 MB |
| threshold | 1.13M | 26.891→26.425 | −0.466 | 62.5 MB | 267.2 → 43.8 MB |
| LapisGS v5 | 7.7M | 26.971 (uncompressed) | — | — | delta 669.8 / full 1817.6 MB |

Even after compression, threshold L3 (26.425 @ 62.5 MB streamed) **beats** top-k
(26.161 @ 43.4 MB) by +0.264 dB at +44% storage, and both are an order of
magnitude smaller than LapisGS (whose v5 numbers are *uncompressed* PLY: per-level
delta 669.8 MB, cumulative 1817.6 MB at 7.7M GS). Caveat: the LapisGS reference is
uncompressed, so the fair size comparison is EvoGS uncompressed leaves PLY (top-k
161 MB / thr 267 MB) vs LapisGS L3 — still ~7–11× smaller, tracking the GS-count
ratio; 8-bit+zstd then takes EvoGS to 29–44 MB. Files:
`results/garden_lapis_evogs_{asym,asym_thr}/eval/compress_L3.json`.
