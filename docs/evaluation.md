# Evaluation

**Input:** fixed GEMM/BMM workloads, a GPU environment, and a method to compare.
**Output:** correctness, tuning cost, warm latency, and performance on unseen shapes.

Current baselines: PyTorch eager, Inductor default, and max-autotune.
All three passed the initial eight FP16 workloads on H100; final workloads are pending.

| Measurement | Source / purpose |
| --- | --- |
| Correctness and warm latency | `benchmark.py` compares outputs and records CUDA-event samples. |
| Environment and raw results | `run_gpu_baselines.sh` saves versions, workload/source records, and JSON files. |
| Tuning cost | TODO: record total tuning time and candidate measurement count. |
| Dispatch overhead | TODO: separate kernel selection cost from execution time. |

`first_call_ms` includes execution and possible compilation/autotuning; it is not
isolated tuning time. Current latency measures operator calls, not isolated kernels.

## TODO

- [ ] Expand `configs/workloads.json`; keep train and unseen shapes disjoint.
- [ ] Connect team kernels/model through `dispatch.py` to `benchmark.py`.
- [ ] Add per-shape Triton tuning and compare it with joint tuning using the same candidates and budget.
- [ ] Freeze the shared model before unseen evaluation; report per-shape tuning cost separately.
- [ ] Repeat runs on the same GPU with consistent precision and controlled caches; report correctness, tuning cost, and latency together.
