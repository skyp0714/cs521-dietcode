# CS 521 — DietCode-inspired Dynamic Tensor Optimization

Team: Yi-Shiun Chou, Ambika Sharan, Haneul Park. Fall 2026.

We aim to reimplement the key ideas of [DietCode (MLSys 2022)](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html)
for GEMM and batch matrix multiplication in PyTorch/TorchInductor: a shape-generic
kernel search space, a shared performance model with joint tuning, and runtime
selection across dynamic shapes. This repository is an independent course project.

## Current status

The initial scaffold includes GEMM/BMM workloads, correctness-checked PyTorch
baselines (`eager`, `inductor`, `inductor-max-autotune`), JSON measurements,
a Python dispatcher that accepts kernel candidates and a predictor, and CPU CI.
Train and unseen shape sets are disjoint. The eight small workloads are for
smoke testing; extend them for final performance experiments.

Local validation (October 27, 2026): Python 3.12.3, PyTorch 2.5.1+cpu for CPU
checks, and PyTorch 2.5.1+cu124 / Triton 3.1.0 on an NVIDIA H100 NVL for GPU
checks. All 19 tests passed in the GPU environment. All eight shapes passed
correctness checks with each of the three baseline modes on both CPU (float32)
and GPU (float16). These small smoke runs do not establish performance gains.

**Pending:** Triton kernels and per-shape autotuning, a learned cost model,
joint autotuning, and integration of selected kernels into TorchInductor.
The Python dispatcher currently falls back to `torch.matmul` if no candidates
are registered. It does not yet provide an optimization over PyTorch.

## Setup

Use Linux and Python 3.10–3.12. CPU checks use Python 3.12 and PyTorch 2.5.1.
The example CUDA environment also starts from 2.5.1; validate the GPU/driver
first and agree on one version before implementing private Inductor APIs.

```bash
git clone https://github.com/skyp0714/cs521-dietcode.git
cd cs521-dietcode
python3 -m venv .venv
source .venv/bin/activate

# CPU development / smoke testing
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
dietcode-bench --device cpu --backend eager --warmup 1 --repeats 3 --output results/cpu-smoke.json
```

For a compatible NVIDIA GPU, use a separate environment or replace the CPU
build with the explicitly selected CUDA build:

```bash
python -m pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Let the PyTorch CUDA wheel install its matching Triton dependency. See
[official version-specific installation commands](https://docs.pytorch.org/get-started/previous-versions/)
if your GPU needs a different build. The GPU path must be tested on the team's
target machine; CPU smoke tests do not validate GPU performance. For the project
server's H100 and its separate `.venv-gpu` environment, see
[GPU setup and the one-command baseline runner](docs/gpu-setup.md).

To inspect workloads without installing PyTorch:

```bash
python -m pip install -e .
dietcode-bench --dry-run
```

Run commands from the repository root, or pass an absolute `--workloads` path.
`python -m dietcode` is equivalent to `dietcode-bench` after installation.

## Baseline experiments

Run each baseline in a separate process on the same otherwise idle GPU:

```bash
dietcode-bench --backend eager --device cuda --dtype float16 --output results/eager.json
dietcode-bench --backend inductor --device cuda --dtype float16 --output results/inductor.json
dietcode-bench --backend inductor-max-autotune --device cuda --dtype float16 --output results/inductor-max-autotune.json
```

`torch.compile` uses `dynamic=True` and `fullgraph=True`; each operation's
callable is reused across the workload sequence. The max-autotune baseline
uses PyTorch's actual `max-autotune` mode, including its default CUDA graph
behavior. See the [torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile).
Recompilation can still occur, and these modes must be validated with logs
on the selected PyTorch version. Use `TORCH_LOGS=recompiles,dynamic` when diagnosing it.

Every shape is checked against `torch.matmul` before measuring warm latency.
JSON records the environment, shape/split, first-call wall time, callable setup
time, raw warm samples, median/min/max latency, and TFLOPS. CUDA latency uses
CUDA events; CPU latency uses a wall clock and includes Python overhead.
Float32 TF32 is disabled. Dtype-dependent correctness tolerances are in the harness.

`first_call_ms` includes execution and any lazy compilation/autotuning, and may
reuse caches from previous shapes or runs. **It is not isolated autotuning time.**
For cold-start experiments, use fresh processes and fresh `TORCHINDUCTOR_CACHE_DIR`
and `TRITON_CACHE_DIR` directories for each run. Record the driver version,
GPU sharing/power settings, dependency snapshot (`python -m pip freeze`), source
commit, workload file, and full command alongside results. See
[the evaluation plan](docs/evaluation.md) for the remaining instrumentation.

Use `--split train` for model/tuner measurements. Do not train or select
configurations using unseen results. An `--split all` baseline run only executes
the listed inputs; it does not train a model. Unseen-only cold runs and a dynamic
sequence with train shapes followed by unseen shapes answer different questions.

## Team ownership

| Role | Main responsibility | Initial entry point |
| --- | --- | --- |
| Person 1 | Triton GEMM/BMM candidates, masks/padding, kernel profiling | `KernelConfig`, `KernelCandidate` in `src/dietcode/contracts.py` |
| Person 2 | Shared cost model, measurement collection, joint autotuning | `Predictor.predict_us(problem, config)` in `src/dietcode/contracts.py` |
| Person 3 — Haneul Park | Baselines, runtime dispatch, Inductor integration, evaluation/report | `src/dietcode/benchmark.py`, `src/dietcode/dispatch.py` |

Read [shared interfaces](docs/interfaces.md), [the team plan](docs/team-plan.md),
and [Person 3's Korean checklist](docs/person3-plan.ko.md) before implementing.

Use feature branches and pull requests into `main`. Keep generated measurements
under `results/` (ignored by Git); commit selected final data explicitly once
reviewed. Original course/team documents are kept local and summarized in `docs/`.
The repository is public, so teammates can clone it immediately. Direct push
access requires adding their GitHub accounts as collaborators.

## Acknowledgment

OpenAI Codex assisted with the initial repository scaffold, baseline harness,
dispatcher scaffold, tests, and planning documents. Team members are responsible
for reviewing the implementation and interpreting experimental results. Update
this acknowledgment with subsequent AI assistance, as required by the course.
