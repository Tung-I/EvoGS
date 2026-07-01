#!/bin/bash
# garden_lapis apples-to-apples: EvoGS on the Mip-NeRF360-capped pyramid
# (200/400/800/1600px) used by our LapisGS v5 reference. Two arms, sequential:
#   (A) top-k asym  -> results/garden_lapis_evogs_asym
#   (B) threshold asym -> results/garden_lapis_evogs_asym_thr
# Trainer is idempotent (skips levels whose level_NN_leaves.ply exists).
set -u
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"; CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"; export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORK_DIR/EvoGS:$WORK_DIR/EvoGS/gsplat:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$WORK_DIR/EvoGS"
GL="$WORK_DIR/datasets/garden_lapis"

echo "[gl] $(date) host=$(hostname) === (A) top-k asym ==="
python evogs/train_evogs.py --data-dir "$GL" \
  --result-dir results/garden_lapis_evogs_asym \
  --learned-repr --asymmetric \
  --stages level0 level1 level2 level3 eval || echo "[gl] (A) FAILED"

echo "[gl] $(date) === (B) threshold asym ==="
python evogs/train_evogs.py --data-dir "$GL" \
  --result-dir results/garden_lapis_evogs_asym_thr \
  --learned-repr --asymmetric \
  --split-mode threshold --split-grad-threshold 2e-4 --split-max-frac 0.05 \
  --stages level0 level1 level2 level3 eval || echo "[gl] (B) FAILED"

echo "[gl] $(date) DONE EXIT=$?"
