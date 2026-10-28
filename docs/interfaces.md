# Component interfaces

`contracts.py` defines how kernels, the cost model, and dispatch connect.

| Component | Input → output | Owner |
| --- | --- | --- |
| `Problem` | Operation, shape, dtype → one GEMM/BMM description | Shared |
| `KernelCandidate` | Input tensors + kernel configuration → result tensor | Person 1 |
| `Predictor.predict_us` | Problem + configuration → estimated latency in microseconds | Person 2 |
| `RuntimeDispatcher` | Tensors + candidates + predictor → selected kernel's result | Person 3 |

GEMM: `(M, K) @ (K, N)`. BMM: `(B, M, K) @ (B, K, N)`.
Start with inference inputs, matching dtype/device, and no batch broadcasting.
Candidates check supported shapes, devices and strides before launch; predictions
must be finite and nonnegative. Keep one trained model per target environment.

## TODO

- [ ] Person 1: expose correct GEMM/BMM candidates through `supports` and `launch`.
- [ ] Person 2: train the predictor on candidate timings from train shapes; freeze it before unseen evaluation.
- [ ] Person 3: connect both to `dispatch.py`, then benchmark the dispatched method in `benchmark.py`.
- [ ] Person 3: add TorchInductor integration after Python dispatch works.

Currently dispatch uses registration order without a predictor and PyTorch when
no candidate supports the input. Clear its cache after updating the model.
