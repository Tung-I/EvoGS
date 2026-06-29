#!/bin/bash
# Controlled A/B: learned-repr asymmetric vs symmetric on raw garden.
# Reuses the existing skeleton L0 (standard 3DGS, identical) in each result dir.
set -e
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"; CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/EvoGS:$WORK_DIR/EvoGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1   # live log output (no stdout block-buffering under nohup)

cd "$WORK_DIR/EvoGS"
VARIANT="${1:-asym}"    # asym | sym
if [ "$VARIANT" = "asym" ]; then
  FLAGS="--learned-repr --asymmetric"
  RDIR="results/garden_evogs_asym"
else
  FLAGS="--learned-repr --no-asymmetric"
  RDIR="results/garden_evogs_sym"
fi

echo "[A/B] variant=$VARIANT  result-dir=$RDIR"
python evogs/train_evogs.py \
  --data-dir "$WORK_DIR/datasets/garden" \
  --result-dir "$RDIR" \
  $FLAGS \
  --stages level1 level2 level3 eval
