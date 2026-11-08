# CS 521 — DietCode-inspired Dynamic Tensor Optimization

Team: Yi-Shiun Chou, Ambika Sharan, Haneul Park.

Apply DietCode's shared kernel search space, cost model, and runtime selection
ideas to dynamic GEMM/BMM in PyTorch/TorchInductor.

- **Input:** matrices `A`, `B` with varying shapes; benchmark shapes and dtype.
- **Output:** `C = A @ B`; JSON results with correctness, latency, and environment.
- **Current status:** three PyTorch baselines validated on H100; dispatcher scaffold
  ready. Initial FP16 Triton GEMM/BMM candidates support irregular shapes; GPU
  validation is pending. Shared model, joint tuning, and Inductor integration are pending.

## Files and data flow

```text
configs/workloads.json → workloads.py → benchmark.py → results/*.json
```

| File | Input → output / role |
| --- | --- |
| [configs/workloads.json](configs/workloads.json) | GEMM/BMM shapes, separated into train and unseen sets. |
| [workloads.py](src/dietcode/workloads.py) | Workload settings → validated problems. |
| [contracts.py](src/dietcode/contracts.py) | Shared definitions for problems, kernel candidates, and the predictor interface. |
| [kernels/](src/dietcode/kernels/) | Configurable FP16 Triton GEMM/BMM candidates with masked edge handling. |
| [benchmark.py](src/dietcode/benchmark.py) | Problems → generated tensors → baseline execution, correctness checks, timing JSON. |
| [dispatch.py](src/dietcode/dispatch.py) | Tensors + candidates + predictor → selected kernel → result tensor. Currently falls back to PyTorch without candidates. |
| [run_gpu_baselines.sh](scripts/run_gpu_baselines.sh) | Runs all three GPU baselines and saves results plus environment records. |

## TODO — connect the components

- [x] **Person 1:** implement initial Triton GEMM/BMM candidates with irregular-shape
  support through `contracts.py`. See [kernel usage and scope](docs/kernels.md).
- [ ] **Person 1:** run all candidate correctness tests on H100; GPU validation
  is required before treating the Week 1–2 deliverables as validated.
- [ ] **Person 2:** collect train-shape candidate timings through an extended
  `benchmark.py`; use them to build the shared predictor and joint tuner.
- [ ] **Person 3:** connect those candidates and the predictor to `dispatch.py`,
  then add the dispatched method to `benchmark.py` alongside existing baselines.
- [ ] **Person 3:** integrate kernel selection with TorchInductor and verify the
  selected team kernels actually execute in compiled graphs.
- [ ] **Team:** expand `workloads.json`; compare tuning cost and latency against
  the baselines and a per-shape Triton tuner on frozen unseen shapes, then report results.

[GPU setup / running](docs/gpu-setup.md) · [Interface details](docs/interfaces.md) ·
[Evaluation plan](docs/evaluation.md) · [Week 1–2 kernels](docs/kernels.md)

AI assistance: OpenAI Codex helped create the initial scaffold, tests, and documentation.
