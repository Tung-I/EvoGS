#!/bin/bash
# EvoGS on city-scale Rubble — STRICT regime (ancestors frozen, like LapisGS),
# learned asymmetric alpha, gradient-THRESHOLD split. Tests whether the psi-lineage
# (S = root + sum psi_k) corrects an inaccurate frozen base where LapisGS's
# freeze-and-add plateaued (L3 f4 22.61 dB vs vanilla 25.29). Idempotent/resumable.
#
# Usage:
#   TAU=1e-4 CAP=0.05 STEPS=30000 RESULT=results/rubble_evogs_strict_thr \
#     STAGES="level0 level1 level2 level3 eval" bash evogs/scripts/run_rubble_evogs.sh
set -u
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"; CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORK_DIR/EvoGS:$WORK_DIR/EvoGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$WORK_DIR/EvoGS"

RUBBLE="$WORK_DIR/lctvgs/datasets/rubble_sfm_v2"
RESULT="${RESULT:-results/rubble_evogs_strict_thr}"
TAU="${TAU:-1e-4}"                     # split-grad threshold (calibrate from smoke)
CAP="${CAP:-0.05}"                     # per-event max frac split (OOM safety)
STEPS="${STEPS:-30000}"
STAGES="${STAGES:-level0 level1 level2 level3 eval}"

echo "[rubble-evogs] $(date) RESULT=$RESULT TAU=$TAU CAP=$CAP STEPS=$STEPS STAGES=[$STAGES]"
python evogs/train_evogs.py \
  --data-dir "$RUBBLE" \
  --result-dir "$RESULT" \
  --data-factors 32 16 8 4 \
  --learned-repr --asymmetric --freeze-inherited \
  --split-mode threshold --split-grad-threshold "$TAU" --split-max-frac "$CAP" \
  --steps-per-level "$STEPS" \
  --stages $STAGES
echo "[rubble-evogs] $(date) EXIT=$?"
