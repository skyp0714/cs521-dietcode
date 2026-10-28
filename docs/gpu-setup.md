# GPU development on the project server

The selected server has one NVIDIA H100 NVL with 95,830 MiB reported memory,
compute capability 9.0, and driver 580.105.08. The project GPU environment uses
Python 3.12, PyTorch 2.5.1+cu124, and its matching Triton 3.1.0.
The driver reports CUDA 13.0 support; the PyTorch wheel carries its own CUDA 12.4
runtime. The system CUDA toolkit version and the wheel runtime need not match.

Validation on October 27, 2026: all 19 tests passed in the GPU environment,
including the CUDA benchmark/device-selection regression check. Eager, default
Inductor, and max-autotune each passed all eight float16 GEMM/BMM workloads
with 10 warmup and 30 measured calls. Max-autotune emitted actual candidate
benchmarking logs. These are initial correctness/setup checks, not evidence
of speedup from the team's pending implementation.

The repository is public: <https://github.com/skyp0714/cs521-dietcode>.
Reading/cloning requires no invitation. Push access still requires a collaborator
invitation or a fork and pull request.

## Existing server checkout

A separate `.venv-gpu` link points to a GPU environment on `/storage`, where
there is room for the CUDA packages and compiler caches. The original `.venv`
continues to provide CPU development. Both environments are excluded from Git.

From the repository root:

```bash
source .venv-gpu/bin/activate
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
export DIETCODE_GPU_ROOT="$(dirname "$(readlink -f .venv-gpu)")"
bash scripts/run_gpu_baselines.sh
```

This runs the three baselines serially on the visible GPU, with float16 inputs,
10 warmup calls and 30 measured calls for each of eight shapes. It uses a fresh
compiler/Triton cache directory for each backend and keeps the environment,
GPU driver information, source commit and workload snapshot with the results.
The reference matmul and CUDA context are initialized before first-call timing;
these runs are not complete cold-process startup measurements.

Results are saved to `results/gpu/` and are not automatically committed.
`DIETCODE_RESULTS_DIR` can select a different output directory. Each invocation
replaces files in the selected results directory; choose a new directory when
preserving multiple independent trials. Compiler caches remain under
`DIETCODE_GPU_ROOT` and can be reused for diagnosis.

## Fresh checkout / another team account

Create an environment on a disk with sufficient space; the following uses
`/storage/$USER` as an example. Run from the cloned repository root:

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

If using an activated environment with another name, set `DIETCODE_PYTHON=python`.
`CUDA_VISIBLE_DEVICES` can select a GPU; this server currently exposes only one.
These are small setup/correctness checks. Final evaluation still needs larger
workloads, independent repeats, actual team kernels/model, and tuning-time
instrumentation described in [the evaluation plan](evaluation.md).
