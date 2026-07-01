#!/bin/bash
# Resume the gradient-threshold split experiment: L3 + eval only (L0-L2 idempotent-skip).
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
GARDEN="$WORK_DIR/datasets/garden"

echo "[asym_thr] $(date) host=$(hostname) resume L3+eval ..."
python evogs/train_evogs.py --data-dir "$GARDEN" \
  --result-dir results/garden_evogs_asym_thr \
  --learned-repr --asymmetric \
  --split-mode threshold --split-grad-threshold 2e-4 --split-max-frac 0.05 \
  --stages level3 eval
echo "[asym_thr] $(date) EXIT=$?"
