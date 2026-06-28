#!/bin/bash
# EvoGS training script for Garden (Mip-NeRF360).
# Run directly in the interactive session: bash train_garden_evogs.sh
# All 4 levels (~6-8h on L40S).

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/EvoGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WORK_DIR/EvoGS"

python evogs/train_evogs.py \
    --data-dir "$WORK_DIR/datasets/garden" \
    --result-dir results/garden_evogs \
    --stages level0 level1 level2 level3 eval
