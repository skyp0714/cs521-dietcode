import importlib.util
import unittest

from dietcode.benchmark import main
from dietcode.contracts import KernelCandidate, KernelConfig
from dietcode.dispatch import RuntimeDispatcher


@unittest.skipUnless(importlib.util.find_spec("torch"), "install PyTorch for tensor checks")
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(521)

    def test_fallback_gemm_bmm_and_transposed_inputs(self):
        torch = self.torch
        dispatcher = RuntimeDispatcher()
        for a, b in [
            (torch.randn(7, 9), torch.randn(9, 5)),
            (torch.randn(3, 7, 9), torch.randn(3, 9, 5)),
            (torch.randn(9, 7).T, torch.randn(5, 9).T),
        ]:
            with self.subTest(shape=tuple(a.shape)):
                torch.testing.assert_close(dispatcher(a, b), a @ b)

    def test_dispatch_launches_selected_candidate(self):
        torch = self.torch
        calls = []

        def launch(a, b):
            calls.append(True)
            return a @ b
        candidate = KernelCandidate("test", KernelConfig(), lambda *args: True, launch)
        a, b = torch.randn(7, 9), torch.randn(9, 5)
        torch.testing.assert_close(RuntimeDispatcher([candidate])(a, b), a @ b)
        self.assertEqual(len(calls), 1)

    def test_reject_broadcasting_and_bad_dimensions(self):
        torch = self.torch
        for a, b in [
            (torch.randn(2, 3, 4), torch.randn(1, 4, 5)),
            (torch.randn(3, 4), torch.randn(5, 6)),
            (torch.randn(4), torch.randn(4)),
        ]:
            with self.subTest(shape=tuple(a.shape)), self.assertRaises(ValueError):
                RuntimeDispatcher()(a, b)

    def test_reject_mixed_dtype_and_gradients(self):
        torch = self.torch
        a, b = torch.randn(3, 4), torch.randn(4, 5)
        with self.assertRaises(ValueError):
            RuntimeDispatcher()(a, b.double())
        with self.assertRaises(ValueError):
            RuntimeDispatcher()(a.requires_grad_(), b)

    def test_cpu_benchmark_checks_all_workloads_and_records_timings(self):
        import json
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            self.assertEqual(main(["--device", "cpu", "--warmup", "1", "--repeats", "3",
                                   "--output", str(output)]), 0)
            result = json.loads(output.read_text())
        self.assertEqual(len(result["results"]), 8)
        self.assertEqual(result["metadata"]["backend"], "eager")
        for row in result["results"]:
            self.assertTrue(row["correct"])
            self.assertGreater(row["latency_median_ms"], 0)
            self.assertEqual(len(row["samples_ms"]), 3)

    def test_cuda_benchmark_resolves_default_device_and_checks_results(self):
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA device required")
        import json
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cuda.json"
            self.assertEqual(main(["--device", "cuda", "--dtype", "float16", "--warmup", "1",
                                   "--repeats", "3", "--output", str(output)]), 0)
            result = json.loads(output.read_text())
        self.assertEqual(result["metadata"]["device"], f"cuda:{self.torch.cuda.current_device()}")
        self.assertEqual(result["metadata"]["latency_metric"], "cuda_event_elapsed")
        self.assertEqual(len(result["results"]), 8)
        self.assertTrue(all(row["correct"] for row in result["results"]))


if __name__ == "__main__":
    unittest.main()
