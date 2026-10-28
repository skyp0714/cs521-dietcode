# Shared implementation interfaces

These are initial contracts for team review, not a claim that the kernel/model
components have been implemented. Source of truth: `src/dietcode/contracts.py`.

## Workload scope

- GEMM: `(M, K) @ (K, N)`; BMM: `(B, M, K) @ (B, K, N)`.
- Positive dimensions, matching input dtype and device, no batch broadcasting.
- float16, bfloat16, float32; inference only for the initial dispatcher.
- Start with contiguous inputs. Every candidate must explicitly reject layouts
  it cannot handle; the dispatcher key includes both actual input strides.
- Masked loads/stores or local padding must handle irregular dimensions.
- `Problem` contains operation, M/N/K, batch, and dtype; no tensors or timing data.

## Person 1: kernel candidates

Expose a collection of `KernelCandidate(name, config, supports, launch)`.
Names must be unique. A configuration contains block M/N/K, number of warps,
and number of stages; these describe a shape-generic candidate. A candidate
must verify its own device/layout/dtype restrictions in `supports`.

`supports(problem, device, a_strides, b_strides) -> bool` must be cheap and
side-effect-free. `launch(a, b)` returns the full correct result without tuning.
Do not time or search configurations inside launch. Supply an untuned explicit
candidate launch before adding per-shape Triton autotuning as a separate baseline.

## Person 2: prediction and joint tuning

Implement `predict_us(problem, config) -> float` on a predictor object. Predictions
must be finite and nonnegative. The model is specific to the target GPU and its
software/precision environment; create separate models and dispatchers for other
targets. Training and tuning use only `train` shapes. Freeze the model before
unseen evaluation.

Suggested measurement fields: problem, config, device/environment ID, compile
time, tuning time, warm latency, correctness status. Capture failed or unsupported
candidates separately rather than assigning fake latency. Agree on this table
before collecting training measurements.

## Person 3: runtime selection and integration

`RuntimeDispatcher(candidates, predictor)` filters supported candidates, minimizes
predicted execution time, and caches the choice by problem/device/strides.
Without a predictor it uses registration order. Without supported candidates it
falls back to `torch.matmul`. `clear_cache()` is required after changing model
weights; constructing a fresh dispatcher is preferable for frozen evaluation.

The initial implementation is a Python API. It is not an Inductor lowering,
training/autograd API, bounded serving cache, or final GPU launch interface.
Treat TorchInductor integration as a separate milestone and isolate private
version-specific APIs when adding it. Benchmark selection overhead separately
from kernel latency.
