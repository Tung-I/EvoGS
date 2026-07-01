# City-scale Rubble — EvoGS vs LapisGS error-accumulation test (2026-07-01)

**Question.** LapisGS *failed* on city-scale Rubble: adding layers on top of the
frozen L0 base did not improve quality (L3@f4 22.61 dB @ 15.1M GS vs vanilla 25.29
@ 6.9M — more Gaussians, worse). Hypothesis: the cause is **error accumulation
from the frozen, inaccurate base**. EvoGS claims to fix this by *replacing* a
parent with two children and reconstructing every leaf as `S = P_root + Σ ψ_k`,
so descendants can **correct** an ancestor through learned ψ.

**Decisive test — STRICT regime** (`--freeze-inherited`): ancestors stay frozen
exactly like LapisGS, so the *only* path to fixing a bad base is the split
children's ψ. Relaxed mode is out of scope (it lets inherited leaves move freely →
conflates "correction" with "re-training the base").

**Setup.** `results/rubble_evogs_strict_thr/`. Dataset
`../lctvgs/datasets/rubble_sfm_v2` (1449 train / 208 val, pyramid [32,16,8,4] =
matches LapisGS Rubble). Config: `--learned-repr --asymmetric --freeze-inherited
--split-mode threshold --split-grad-threshold 1e-4 --split-max-frac 0.08`, 30k
steps/level. τ/cap calibrated on a Rubble smoke (L1 step-3000 avg_grad p95=1.2e-4,
p99=2.6e-4; cap 0.08 chosen so τ governs at ~7%/event and the strict regime is not
starved of children). Refs: LapisGS `../lctvgs/results/rubble_lapisgs/eval/`,
vanilla `../lctvgs/results/rubble_vanilla_sfm/` (25.29@f4 60k / 24.21@f4 30k).

## Result 1 — native per-level (each level at its training factor)
| Level | f | EvoGS PSNR / SSIM / LPIPS | EvoGS #GS | LapisGS PSNR | LapisGS #GS | ΔPSNR | GS ratio |
|------:|--:|:--|--:|--:|--:|--:|--:|
| L0 | 32 | 26.864 / 0.903 / 0.062 | 2.47M | 26.78 | 2.49M | +0.08 | ~1× |
| L1 | 16 | 26.405 / 0.859 / 0.109 | 2.45M | 25.63 | 5.50M | **+0.78** | 2.2× fewer |
| L2 | 8  | 24.977 / 0.759 / 0.229 | 3.46M | 24.17 | 9.86M | **+0.81** | 2.9× fewer |
| L3 | 4  | 23.558 / 0.643 / 0.381 | 4.89M | 22.61 | 15.10M | **+0.95** | 3.1× fewer |

## Result 2 — fixed factor-4 streaming ladder (all levels rendered @ f4) — DECISIVE
| Level | EvoGS PSNR | LapisGS PSNR | Δ | EvoGS #GS | LapisGS #GS |
|------:|-----------:|-------------:|--:|----------:|------------:|
| L0 | 19.345 | 19.252 | +0.093 | 2.47M | 2.49M |
| L1 | 20.928 | 20.372 | +0.556 | 2.45M | 5.50M |
| L2 | 22.884 | 21.987 | +0.897 | 3.46M | 9.86M |
| L3 | **23.558** | **22.605** | **+0.953** | **4.89M** | **15.10M** |
| vanilla @f4 | — | 24.21 (30k) / 25.29 (60k) | — | — | 6.91M |

The EvoGS ladder is monotonic (19.35→23.56) and **beats LapisGS at every level;
the gap GROWS with depth (+0.09 → +0.95)** — the direct signature of the
error-accumulation fix. LapisGS accumulates error as layers deepen; EvoGS's ψ
lineage corrects it, at **3.1× fewer Gaussians** at L3.

## Result 3 — redundancy / "ghost" fraction (alpha < 0.05, identical threshold)
| Level | LapisGS transp% (N) | EvoGS transp% (N) |
|------:|--------------------:|------------------:|
| L0 | 65.7% (2.49M) | 65.7% (2.47M) |
| L1 | 63.1% (5.50M) | 38.8% (2.45M) |
| L2 | 57.7% (9.86M) | 30.5% (3.46M) |
| L3 | **52.9% (15.10M)** | **24.2% (4.89M)** |

Both start identical at L0 (same vanilla base). Then they diverge completely:
LapisGS stays bloated → **8.0M transparent ghost splats at L3** (52.9% of 15.1M);
EvoGS leans out → **1.2M** (24.2% of 4.89M). LapisGS carries ~6.8× more dead
weight. EvoGS even reaches higher PSNR with fewer *active* splats (≈3.7M vs 7.1M).
EvoGS L3 24.2% matches the paper's EvoGS redundancy claim (~24.9%, Table 3).

## Result 4 — storage (L3, 8-bit + zstd, 16-bit means)
- EvoGS render footprint (L3 leaves): 1153.2 MB → q8+zstd **169.1 MB** (6.8×);
  ΔPSNR −0.46 dB (23.56 → 23.10).
- EvoGS streamed (root + ψ/α residuals L1..L3): 1153.6 MB raw → q8+zstd **224.2 MB**.
- LapisGS L3: cumulative full **3563.65 MB** (uncompressed), per-level delta 889.85.
- EvoGS streamed 224 MB is ~16× smaller than LapisGS cumulative; even uncompressed
  EvoGS leaves (1153 MB) are ~3× smaller (tracking the 3.1× GS ratio).
  Files: `results/rubble_evogs_strict_thr/eval/{metrics_eval,metrics_fixed_factor4,compress_L3}.json`.

## Verdict — HYPOTHESIS CONFIRMED (with an honest nuance)
The four numbers that tell the whole story (all @ factor 4, L3):

| model | PSNR | note |
|------:|-----:|:-----|
| LapisGS frozen | 22.61 | freeze-and-add, 15.1M GS |
| LapisGS unfrozen + finetuned | 23.67 | (from LapisGS diagnosis) unfreeze all, no densify |
| **EvoGS strict (ancestors frozen)** | **23.56** | replace + ψ, 4.9M GS |
| vanilla 30k / 60k | 24.21 / 25.29 | 6.9M GS |

1. **Error accumulation is real and EvoGS fixes it.** Strict EvoGS — with ancestors
   *frozen exactly like LapisGS* — beats frozen LapisGS by +0.95 dB at L3 (gap
   growing with depth) at 3× fewer Gaussians, and **essentially matches
   unfrozen-finetuned LapisGS (23.56 ≈ 23.67) without unfreezing anything.** The
   ≈+1 dB that LapisGS could only recover by unfreezing, EvoGS recovers through the
   ψ-lineage while keeping the freeze constraint. This isolates the *architecture*
   (replace + collinear ψ correction), not "let the base move", as the fix.
2. **The redundancy mechanism is confirmed directly** (Result 3): LapisGS masks the
   uncorrectable frozen-base error with ~8M transparent ghosts; EvoGS corrects it
   and stays lean (~1.2M), reproducing the paper's Table-3 shape on a real
   city-scale scene.
3. **Nuance (honest):** strict EvoGS L3 (23.56) is still 0.65 dB below 30k-vanilla
   (24.21) and 1.73 below 60k-vanilla (25.29). So error accumulation was the
   *major* component of LapisGS's Rubble failure (EvoGS closes ~64% of the
   frozen→60k-vanilla gap: −2.68 → −1.73 dB), but a residual capacity/optimization
   gap remains that is *not* attributable to the freeze — consistent with unfrozen
   LapisGS also stalling at 23.67. Closing that last gap would need more
   capacity/steps (or the relaxed regime), which is a separate axis from the
   accumulation question this study answered.

## Reproduce / files
- Train: `RESULT=results/rubble_evogs_strict_thr TAU=1e-4 CAP=0.08 STEPS=30000 \
  STAGES="level0 level1 level2 level3 eval" bash evogs/scripts/run_rubble_evogs.sh`
  (idempotent). Log `logs/rubble_strict_thr.log`.
- Eval: `python evogs/eval_evogs.py --data-dir <rubble> --result-dir <R>
  --data-factors 32 16 8 4 [--eval-factor 4]` (fixed factor + transparency %).
- Open follow-ups: relaxed-regime contrast (upper bound / how much the freeze still
  costs); threshold cap sweep for the residual capacity gap; 2nd city scene.
