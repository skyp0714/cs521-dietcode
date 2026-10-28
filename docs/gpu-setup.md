# GPU setup and running

**Environment:** one H100 NVL; driver 580.105.08; Python 3.12;
PyTorch 2.5.1+cu124; Triton 3.1.0. The existing GPU environment is on `/storage`,
linked as `.venv-gpu`; `.venv` is the separate CPU environment.

**Input:** `configs/workloads.json` and the GPU environment.
**Output:** `results/gpu/` with baseline JSON, environment and source/workload records.
Each baseline uses FP16, 10 warmup calls and 30 measured calls per shape.
All three baselines passed eight shapes; the GPU environment passed 19 tests.

## Run in the existing server checkout

From the repository root:

```bash
export DIETCODE_GPU_ROOT="$(dirname "$(readlink -f .venv-gpu)")"
bash scripts/run_gpu_baselines.sh
```

## Set up a fresh checkout

From the repository root, using a storage directory with enough space:

```bash
export DIETCODE_GPU_ROOT="/storage/$USER/cs521-dietcode"
mkdir -p "$DIETCODE_GPU_ROOT/tmp"
python3 -m venv "$DIETCODE_GPU_ROOT/venv"
ln -s "$DIETCODE_GPU_ROOT/venv" .venv-gpu
export TMPDIR="$DIETCODE_GPU_ROOT/tmp"
.venv-gpu/bin/python -m pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
.venv-gpu/bin/python -m pip install -e . --no-cache-dir
bash scripts/run_gpu_baselines.sh
```

## TODO

- [ ] Choose a new `DIETCODE_RESULTS_DIR` for each independent trial; the default output files are overwritten.
- [ ] Connect team kernels/model to `benchmark.py`, then compare using [the evaluation plan](evaluation.md).

The runner creates fresh compiler caches under `DIETCODE_GPU_ROOT` for each backend.
Generated environments/results stay local and are excluded from Git.
