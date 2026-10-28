# Evaluation protocol

## Baselines and current availability

| Method | Status | Purpose |
| --- | --- | --- |
| PyTorch eager | Implemented | Vendor-backed reference and correctness |
| TorchInductor default, dynamic shapes | Implemented entry point | Default compiler comparison |
| TorchInductor max-autotune, dynamic shapes | Implemented entry point | PyTorch autotuning comparison |
| Per-shape Triton autotuning | Pending Person 1/2 integration | Independent search cost and latency |
| Shared-model joint tuning + dispatcher | Pending actual kernels/model | Team method |
| Team method inside TorchInductor | Pending Person 3 integration | Compiler integration result |

CPU execution validates the harness. The initial eight shapes also passed
correctness checks in all three implemented modes on the server's H100 NVL
with float16 inputs. Final-workload GPU correctness, timing and integration
still require validation before claiming performance results.

## Measurements

The scaffold records first invocation wall time, callable setup, warm execution
samples, correctness and environment metadata. CUDA samples use events with
warmup and synchronization; CPU samples use wall time. These are operator-call
measurements and can include allocation, launch gaps and CUDA graph behavior.
They are not profiler-isolated individual kernel measurements.

The following must be added with the actual tuner:

- Wall-clock total search/tuning time and number of candidate measurements.
- Compilation time separated from candidate execution/search time.
- Python dispatch selection/cache overhead and fallback frequency.
- Selected configuration, actual kernel evidence, recompilation count.
- Warm kernel-only profiler/event measurements where needed for interpretation.

Do not call `first_call_ms` isolated tuning time: it contains execution, possible
compilation/autotuning and cache effects. A previously unseen shape can still
reuse compilation products. Control persistent caches in addition to process state.

## Fair comparison

1. Record GPU model, driver, CUDA/PyTorch/Triton versions, exact dependency snapshot,
   source commit, workload file, commands, dtype and GPU sharing/power conditions.
2. Use the same fixed shapes, seeds, correctness tolerances and precision settings.
   Start with one GPU and contiguous inference inputs.
3. Use the same configuration space, GPU and measurement policy for per-shape
   versus joint tuning. Report both fixed-budget and comparable-quality results.
4. Train only on `train`. Freeze configurations/model before `unseen`. Do not
   tune on unseen shapes when evaluating transfer. The per-shape tuning baseline
   may tune there, but its added tuning cost must be reported separately.
5. Run each backend in a fresh process. Use distinct empty compiler/Triton cache
   directories for cold measurements, then separately measure warm performance.
6. Log recompilations and graph breaks. Confirm max-autotune and the intended
   kernel actually execute. Default max-autotune can use CUDA graphs; report this
   and consider an additional no-CUDA-graphs comparison for overhead analysis.
7. Run multiple independent trials, preserve raw samples, and summarize per-shape
   latency and geometric-mean speedup. Report failures rather than excluding them.

Expand the initial eight shapes to include larger, skinny, irregular and varying
batch cases before the final evaluation. Keep a frozen disjoint held-out set.
Compare both cold unseen-only runs and train-to-unseen runtime sequences, clearly
labeling their cache behavior. Do not infer GPU speedup from CPU smoke data.

## Report material

- End-to-end tuning time versus number of shapes, measurement counts, compile cost.
- Per-shape warm latency and aggregate speedup; distribution across independent runs.
- Unseen-shape generalization, model ranking error and fallback behavior.
- Ablations for shared model/search space, padding overhead, and runtime dispatch.
- Limitations: supported operators/layouts/dtypes, model/device specificity,
  version-sensitive private Inductor APIs, and scope relative to the original paper.

References: [DietCode](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html),
[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile),
[Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
