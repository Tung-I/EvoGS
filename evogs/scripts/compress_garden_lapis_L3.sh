#!/bin/bash
# Compress L3 of both garden_lapis arms (top-k + threshold) for the storage
# comparison vs LapisGS v5 (matched 200/400/800/1600px pyramid).
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

for arm in garden_lapis_evogs_asym garden_lapis_evogs_asym_thr; do
  echo "[gl-compress] $(date) $arm L3 ..."
  python evogs/compress_evogs.py --result-dir "results/$arm" \
    --data-dir "$GL" --level 3 --eval-factor 1 || echo "[gl-compress] FAILED $arm"
done
echo "[gl-compress] $(date) DONE EXIT=$?"
