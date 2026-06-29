#!/bin/bash
# Unattended chain: waits for the running asym garden run, then sym garden,
# then compression on both, then a 2nd Mip360 scene (bicycle) asym from scratch.
# Each stage logs separately; a failure in one stage does not abort the chain.
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"; CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/EvoGS:$WORK_DIR/EvoGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK_DIR/EvoGS"
GARDEN="$WORK_DIR/datasets/garden"

echo "[chain] $(date) waiting for asym garden to finish..."
until [ -f results/garden_evogs_asym/eval/metrics.json ]; do sleep 60; done
echo "[chain] $(date) asym done. compressing asym L3..."
python evogs/compress_evogs.py --data-dir "$GARDEN" \
  --result-dir results/garden_evogs_asym --level 3 --eval-factor 1 \
  > logs/compress_asym.log 2>&1 || echo "[chain] compress asym FAILED"

echo "[chain] $(date) training sym garden..."
bash evogs/scripts/run_ab_garden.sh sym > logs/garden_sym.log 2>&1 || echo "[chain] sym FAILED"
echo "[chain] $(date) compressing sym L3..."
python evogs/compress_evogs.py --data-dir "$GARDEN" \
  --result-dir results/garden_evogs_sym --level 3 --eval-factor 1 \
  > logs/compress_sym.log 2>&1 || echo "[chain] compress sym FAILED"

echo "[chain] $(date) training bicycle asym (full L0-L3 from scratch)..."
python evogs/train_evogs.py --data-dir "$WORK_DIR/datasets/bicycle" \
  --result-dir results/bicycle_evogs_asym \
  --learned-repr --asymmetric \
  --stages level0 level1 level2 level3 eval > logs/bicycle_asym.log 2>&1 \
  || echo "[chain] bicycle FAILED"

echo "[chain] $(date) ALL DONE"
