"""CPU interface checks plus GPU tests that launch every candidate directly."""

import unittest

from dietcode.contracts import KernelConfig, Problem
from dietcode.dispatch import RuntimeDispatcher
from dietcode.kernels import DEFAULT_CONFIGS, make_candidates
from dietcode.workloads import load_workloads


class CandidateTests(unittest.TestCase):
    def test_shared_configs_have_unique_stable_names(self):
        candidates = make_candidates()
        self.assertEqual(tuple(c.config for c in candidates), DEFAULT_CONFIGS)
        self.assertEqual(len({c.name for c in candidates}), len(DEFAULT_CONFIGS))
        self.assertEqual([c.name for c in candidates], [c.name for c in make_candidates()])

    def test_reject_invalid_or_duplicate_configs(self):
        for configs in [(KernelConfig(block_m=17),), (KernelConfig(num_warps=3),),
                        (KernelConfig(num_stages=5),), (DEFAULT_CONFIGS[0],) * 2]:
            with self.subTest(configs=configs), self.assertRaises(ValueError):
                make_candidates(configs)

    def test_unsupported_inputs_do_not_select_a_kernel(self):
        dispatcher = RuntimeDispatcher(make_candidates())
        p = Problem("gemm", 7, 5, 9, dtype="float16")
        for problem, device, sa, sb in [
            (p, "cpu", (9, 1), (5, 1)),
            (p, "cuda:0", (1, 7), (5, 1)),
            (p, "cuda:0", (9, 1), (10, 2)),
            (Problem("gemm", 7, 5, 9), "cuda:0", (9, 1), (5, 1)),
        ]:
            with self.subTest(device=device, strides=sa):
                self.assertIsNone(dispatcher.select(problem, device, sa, sb))


class GPUKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import triton
        except ImportError:
            raise unittest.SkipTest("PyTorch and Triton required for GPU kernel tests")
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise unittest.SkipTest("NVIDIA CUDA GPU required")
        if torch.cuda.get_device_capability()[0] < 8:
            raise unittest.SkipTest("SM80+ required")
        cls.torch = torch

    def test_every_candidate_on_train_and_edge_shapes(self):
        torch = self.torch
        torch.manual_seed(521)
        previous_precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous_precision)
        torch.set_float32_matmul_precision("highest")
        problems = [w.problem for w in load_workloads("configs/workloads.json", "float16", "train")]
        # Independent development cases; frozen unseen workloads are not used.
        problems += [Problem(op, m, n, k, batch=batch, dtype="float16")
                     for op, batch in (("gemm", 1), ("bmm", 1), ("bmm", 2))
                     for m, n, k in ((1, 1, 1), (1, 17, 33), (33, 1, 17),
                                     (31, 33, 65), (65, 63, 31))]
        with torch.inference_mode():
            for p in problems:
                a_shape, b_shape = p.input_shapes
                # Nonzero storage offsets exercise pointer handling too.
                a = torch.randn((2, *a_shape), device="cuda", dtype=torch.float16)[1]
                b = torch.randn((2, *b_shape), device="cuda", dtype=torch.float16)[1]
                before_a, before_b = a.clone(), b.clone()
                reference = (a.float() @ b.float()).half()
                for candidate in make_candidates():
                    with self.subTest(problem=p, candidate=candidate.name):
                        self.assertTrue(candidate.supports(p, str(a.device), a.stride(), b.stride()))
                        actual = candidate.launch(a, b)
                        torch.cuda.synchronize()
                        self.assertEqual(actual.dtype, a.dtype)
                        self.assertEqual(actual.device, a.device)
                        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(a, before_a, rtol=0, atol=0)
                torch.testing.assert_close(b, before_b, rtol=0, atol=0)

    def test_dispatch_selects_real_candidate_and_falls_back_for_layout(self):
        torch = self.torch
        dispatcher = RuntimeDispatcher(make_candidates())
        a = torch.randn(17, 33, device="cuda", dtype=torch.float16)
        b = torch.randn(33, 19, device="cuda", dtype=torch.float16)
        p = Problem("gemm", 17, 19, 33, dtype="float16")
        self.assertIsNotNone(dispatcher.select(p, str(a.device), a.stride(), b.stride()))
        torch.testing.assert_close(dispatcher(a, b), a @ b, rtol=1e-2, atol=1e-2)
        a = a.T.contiguous().T
        self.assertIsNone(dispatcher.select(p, str(a.device), a.stride(), b.stride()))
        torch.testing.assert_close(dispatcher(a, b), a @ b)
        with self.assertRaises(ValueError):
            make_candidates()[0].launch(a, b)

    def test_direct_launch_rejects_invalid_inputs(self):
        torch = self.torch
        launch = make_candidates()[0].launch
        a = torch.randn(3, 4, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 5, device="cuda", dtype=torch.float16)
        for x, y in [(a.cpu(), b.cpu()), (a.float(), b.float()), (a, b.double()),
                     (a, b[:3]), (a.unsqueeze(0).expand(2, -1, -1), b.unsqueeze(0)),
                     (a.clone().requires_grad_(), b)]:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                launch(x, y)


if __name__ == "__main__":
    unittest.main()
