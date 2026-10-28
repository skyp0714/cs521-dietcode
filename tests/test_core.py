import json
import math
from pathlib import Path
import tempfile
import unittest

from dietcode.contracts import KernelCandidate, KernelConfig, Problem
from dietcode.dispatch import RuntimeDispatcher
from dietcode.workloads import load_workloads


class ProblemTests(unittest.TestCase):
    def test_shapes_and_flops(self):
        gemm = Problem("gemm", 3, 5, 7)
        self.assertEqual(gemm.input_shapes, ((3, 7), (7, 5)))
        self.assertEqual(gemm.flops, 210)
        bmm = Problem("bmm", 3, 5, 7, batch=2)
        self.assertEqual(bmm.input_shapes, ((2, 3, 7), (2, 7, 5)))
        self.assertEqual(bmm.flops, 420)

    def test_reject_invalid_dimensions(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Problem("gemm", value, 8, 8)

    def test_reject_invalid_operation_dtype_and_batch(self):
        for fields in ({"op": "conv"}, {"dtype": "int8"}, {"batch": 3}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Problem(**({"op": "gemm", "m": 8, "n": 8, "k": 8} | fields))


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("gemm", 16, 16, 16)
        self.small = KernelCandidate("small", KernelConfig(block_m=16), lambda *args: True, lambda a, b: None)
        self.large = KernelCandidate("large", KernelConfig(block_m=64), lambda *args: True, lambda a, b: None)

    def select(self, dispatcher, **kwargs):
        return dispatcher.select(kwargs.get("problem", self.problem), kwargs.get("device", "cuda:0"),
                                 kwargs.get("a_strides", (16, 1)), kwargs.get("b_strides", (16, 1)))

    def test_model_chooses_fastest_supported_candidate(self):
        class Model:
            def predict_us(self, problem, config):
                return 100 / config.block_m
        dispatcher = RuntimeDispatcher([self.small, self.large], Model())
        self.assertIs(self.select(dispatcher), self.large)

    def test_layout_restrictions_filter_candidates(self):
        restricted = KernelCandidate("contiguous", KernelConfig(),
                                     lambda p, d, a, b: a[-1] == b[-1] == 1, lambda a, b: None)
        dispatcher = RuntimeDispatcher([restricted])
        self.assertIsNone(self.select(dispatcher, a_strides=(1, 16)))
        self.assertIs(self.select(dispatcher), restricted)

    def test_no_supported_candidate_returns_fallback(self):
        self.assertIsNone(self.select(RuntimeDispatcher()))

    def test_default_selection_uses_registration_order(self):
        self.assertIs(self.select(RuntimeDispatcher([self.small, self.large])), self.small)

    def test_cache_separates_device_dtype_shape_and_strides(self):
        class Model:
            calls = 0

            def predict_us(self, problem, config):
                self.calls += 1
                return 1
        model = Model()
        dispatcher = RuntimeDispatcher([self.small], model)
        self.select(dispatcher)
        self.select(dispatcher)
        self.assertEqual(model.calls, 1)
        for change in ({"device": "cuda:1"}, {"a_strides": (1, 16)}, {"b_strides": (1, 16)},
                       {"problem": Problem("gemm", 17, 16, 16)},
                       {"problem": Problem("gemm", 16, 16, 16, dtype="float16")}):
            self.select(dispatcher, **change)
        self.assertEqual(model.calls, 6)
        dispatcher.clear_cache()
        self.select(dispatcher)
        self.assertEqual(model.calls, 7)

    def test_reject_invalid_predictions(self):
        for score in (math.nan, math.inf, -1):
            class Model:
                def predict_us(self, problem, config):
                    return score
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.select(RuntimeDispatcher([self.small], Model()))

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeDispatcher([self.small, self.small])


class WorkloadTests(unittest.TestCase):
    def test_default_workloads_have_disjoint_splits(self):
        train = load_workloads("configs/workloads.json", split="train")
        unseen = load_workloads("configs/workloads.json", split="unseen")
        self.assertTrue(train and unseen)
        self.assertTrue({w.problem for w in train}.isdisjoint(w.problem for w in unseen))
        self.assertEqual({w.problem.op for w in train}, {"gemm", "bmm"})

    def test_reject_duplicate_shape_between_splits(self):
        self.check_invalid([
            {"name": "one", "split": "train", "op": "gemm", "m": 8, "n": 8, "k": 8},
            {"name": "two", "split": "unseen", "op": "gemm", "m": 8, "n": 8, "k": 8},
        ])

    def test_reject_unknown_split(self):
        self.check_invalid([{"name": "one", "split": "test", "op": "gemm", "m": 8, "n": 8, "k": 8}])

    def check_invalid(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.json"
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                load_workloads(path)


if __name__ == "__main__":
    unittest.main()
