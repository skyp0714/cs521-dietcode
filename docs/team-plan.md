# Team plan

**Goal:** connect shape-generic kernels, a shared cost model, and runtime selection
into one measured GEMM/BMM implementation.

| Owner | Input → output | Connection |
| --- | --- | --- |
| Person 1 | Problems + configurations → correct Triton kernels | Expose `KernelCandidate` from `contracts.py`. |
| Person 2 | Train-shape kernel timings → shared predictor + joint tuner | Supply `Predictor.predict_us`. |
| Person 3 | Kernels + predictor → dispatch, Inductor integration, comparisons | Connect `dispatch.py` and `benchmark.py`. |

## TODO — integration order

- [ ] Agree on `contracts.py`, GPU/software versions, and measurement format.
- [ ] Connect one GEMM candidate to dispatch and correctness/timing checks.
- [ ] Add the predictor, irregular shapes, and BMM.
- [ ] Add joint tuning and a per-shape tuner over the same candidates.
- [ ] Integrate selection with TorchInductor and verify actual kernel execution.
- [ ] Freeze unseen workloads and compare results for the final report.

Progress meeting: **November 13–19, 2026**, with a one-page report beforehand.
Final report/code: **December 9, 2026*** in the assignment; confirm course updates.
Submit working code, data, run instructions, and a report of up to five two-column
pages excluding references/appendix. Acknowledge AI assistance.
