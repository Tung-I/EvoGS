#!/bin/bash
# RD-curve compression batch: threshold arm L1/L2/L3 + top-k baseline L1/L2 backfill.
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

run() {  # result_dir level factor
  echo "[compress] $(date) $1 L$2 f$3 ..."
  python evogs/compress_evogs.py --result-dir "$1" --data-dir "$GARDEN" \
    --level "$2" --eval-factor "$3" || echo "[compress] FAILED $1 L$2"
}

# Threshold arm (new): all three refinement levels
run results/garden_evogs_asym_thr 1 4
run results/garden_evogs_asym_thr 2 2
run results/garden_evogs_asym_thr 3 1
# Top-k baseline backfill (L3 already present)
run results/garden_evogs_asym 1 4
run results/garden_evogs_asym 2 2

echo "[compress] $(date) DONE EXIT=$?"
