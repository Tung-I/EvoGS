#!/bin/bash
# Strict-regime A/B: freeze inherited frontier leaves (ancestors-frozen reading),
# so refinement happens ONLY via split ψ. This is where the paper's asymmetric-α
# advantage over symmetric should appear (Table 2). Runs asym then sym.
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"; CUDA_DIR="$WORK_DIR/cuda-13.0"
unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/EvoGS:$WORK_DIR/EvoGS/gsplat:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$WORK_DIR/EvoGS"
GARDEN="$WORK_DIR/datasets/garden"

echo "[strict] $(date) asym-strict ..."
python evogs/train_evogs.py --data-dir "$GARDEN" \
  --result-dir results/garden_evogs_asym_strict \
  --learned-repr --asymmetric --freeze-inherited \
  --stages level1 level2 level3 eval || echo "[strict] asym FAILED"

echo "[strict] $(date) sym-strict ..."
python evogs/train_evogs.py --data-dir "$GARDEN" \
  --result-dir results/garden_evogs_sym_strict \
  --learned-repr --no-asymmetric --freeze-inherited \
  --stages level1 level2 level3 eval || echo "[strict] sym FAILED"

echo "[strict] $(date) DONE"
