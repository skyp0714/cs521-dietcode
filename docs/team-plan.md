# Team execution plan

Based on the proposal and four-week role table supplied with the project.
Person 3 is Haneul Park; assignments of Person 1 and Person 2 to the other names
should be confirmed by the team.

| Week | Person 1 | Person 2 | Person 3 |
| --- | --- | --- | --- |
| 1 | Implement Triton GEMM candidates | Design model and collect measurements | Set up benchmarking and baselines |
| 2 | Irregular dimensions and BMM | Shared performance predictor | Python runtime dispatcher |
| 3 | Optimize configurations and validate | Joint autotuning | Integrate selection with TorchInductor |
| 4 | Profiling and debugging | Tuning efficiency and unseen shapes | Full baseline comparisons and report |

Integrate at least weekly. Before the November 13–19 progress meeting, aim for
a small actual GEMM candidate connected to selection and correctness checks,
plus preliminary baseline results. Submit the one-page progress report before
the meeting. The final report/code date in the supplied assignment is December
9, 2026 (marked with an asterisk in that document; check course updates).

The final report is at most five two-column pages excluding references and
appendix. Submit working code, experimental data, and installation/run instructions.
Disclose AI-assisted development, as required by the supplied assignment.

## Integration sequence

1. Agree on `Problem`, candidate launch/support rules, latency units, device,
   dependency versions and training data format.
2. Connect one correct GEMM candidate to Python dispatch and baseline measurement.
3. Connect the shared predictor, irregular GEMM and BMM, then freeze train/unseen splits.
4. Add joint tuning and a per-shape tuner baseline over the same candidate space.
5. Add a version-specific Inductor integration; verify the actual selected kernel.
6. Freeze the implementation and experimental protocol before final comparisons.

Use pull requests, include correctness evidence, and update pending status in
README when a component actually runs. The course documents and original team
proposal remain local; this repository contains implementation-focused summaries.
