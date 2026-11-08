# Person 1: Weeks 1 and 2

`dietcode.kernels.make_candidates()` exposes a shared set of tiled Triton GEMM
and BMM kernels through the existing `KernelCandidate` interface. There is no
internal autotuner: each candidate always launches its own configuration, so
Person 2 can measure it and Person 3 can select it explicitly.

Week 1 implementation: tiled GEMM with FP32 accumulation and FP16 output.
Week 2 implementation: masked loads for M/N/K edges, masked output stores, and
an independent batch grid dimension for BMM. Invalid input lanes contribute zero.
Both operations use the same kernel and configuration set.

## Supported inputs

- NVIDIA CUDA GPUs with compute capability 8.0 or newer (including H100).
- FP16, inference only, matching device/dtype, positive dimensions.
- Exact row-major contiguous strides; contiguous views with storage offsets work.
- Rank-2 GEMM or rank-3 BMM with equal batch sizes, at most 65,535 batches.
- No broadcasting, gradients, transposed/strided layouts, BF16, or FP32 yet.

`supports` returns false for unsupported inputs or missing GPU dependencies.
Direct `launch` rejects unsupported inputs instead of silently using PyTorch.
`RuntimeDispatcher` retains its normal PyTorch fallback. Importing the candidate
factory requires neither PyTorch nor Triton.

## Team handoff

```python
import torch
from dietcode.kernels import make_candidates
from dietcode.dispatch import RuntimeDispatcher

candidates = make_candidates()
a = torch.randn(3, 63, 65, device="cuda", dtype=torch.float16)
b = torch.randn(3, 65, 129, device="cuda", dtype=torch.float16)
c = candidates[0].launch(a, b)  # Explicit candidate, suitable for measurement.
dispatch = RuntimeDispatcher(candidates)  # Supply predictor=... when available.
c = dispatch(a, b)
torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
```

Default configurations are `(block_m, block_n, block_k, warps, stages)`:
`(32,32,32,4,2)`, `(32,64,32,4,2)`, `(64,32,32,4,2)`, `(64,64,32,4,3)`.
Custom `KernelConfig` sequences may use tile sizes 16/32/64, warps 4/8, and
stages 2/3/4. These are initial configurations, not performance-tuned winners.
GPU compilation is lazy; warm each candidate before measuring steady-state time.

## Validation on the GPU server

Use the environment in [GPU setup](gpu-setup.md), then run from the repo root:

```bash
.venv-gpu/bin/python -c 'import torch, triton; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 8'
.venv-gpu/bin/python -m unittest discover -s tests -p 'test_kernels.py' -v
.venv-gpu/bin/python -m unittest discover -s tests -v
```

The GPU tests directly execute every default candidate on all training shapes
and additional edge cases (including singleton dimensions, partial reduction
tiles, batch size one, and storage offsets), compare against an FP32 reference
rounded to FP16, and check inputs remain unchanged. They also exercise real
dispatch and unsupported-layout fallback. Frozen unseen shapes are excluded
from kernel development tests.

GPU tests skip when dependencies or a supported GPU are absent. A skipped run
does not validate GPU compilation or numerical correctness. H100 validation and
performance measurements remain required; no kernel speedup is claimed yet.
Benchmark backend/model wiring and Inductor integration remain Person 3's work.

Implementation reference: [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
