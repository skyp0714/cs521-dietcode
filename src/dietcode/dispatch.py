"""Python runtime dispatch scaffold; this is not an Inductor lowering."""

import math
from typing import Any, Iterable

from .contracts import KernelCandidate, Predictor, Problem


class RuntimeDispatcher:
    """Choose a supported candidate and cache by shape, dtype, device and strides.

    Without a predictor, use registration order. Without a supported candidate,
    call torch.matmul. Construct a new dispatcher after updating model weights
    or candidates, or explicitly call clear_cache().
    """

    def __init__(self, candidates: Iterable[KernelCandidate] = (), predictor: Predictor | None = None):
        self.candidates = tuple(candidates)
        self.predictor = predictor
        self._cache: dict[tuple, KernelCandidate | None] = {}
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")

    def clear_cache(self) -> None:
        self._cache.clear()

    def select(self, problem: Problem, device: str, a_strides: tuple[int, ...],
               b_strides: tuple[int, ...]) -> KernelCandidate | None:
        key = (problem, device, a_strides, b_strides)
        if key not in self._cache:
            supported = [c for c in self.candidates if c.supports(problem, device, a_strides, b_strides)]
            if self.predictor is not None and supported:
                scores = [float(self.predictor.predict_us(problem, c.config)) for c in supported]
                if any(not math.isfinite(s) or s < 0 for s in scores):
                    raise ValueError("predicted latencies must be finite and nonnegative")
                chosen = supported[min(range(len(scores)), key=scores.__getitem__)]
            else:
                chosen = supported[0] if supported else None
            self._cache[key] = chosen
        return self._cache[key]

    def __call__(self, a: Any, b: Any) -> Any:
        import torch

        if a.device != b.device or a.dtype != b.dtype:
            raise ValueError("inputs must have the same device and dtype")
        if a.ndim != b.ndim or a.ndim not in (2, 3):
            raise ValueError("expected two rank-2 GEMM or rank-3 BMM inputs")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("inner dimensions do not match")
        if a.ndim == 3 and a.shape[0] != b.shape[0]:
            raise ValueError("BMM requires matching batch sizes; broadcasting is unsupported")
        problem = Problem(
            op="gemm" if a.ndim == 2 else "bmm", m=a.shape[-2], n=b.shape[-1],
            k=a.shape[-1], batch=a.shape[0] if a.ndim == 3 else 1,
            dtype=str(a.dtype).removeprefix("torch."),
        )
        if a.requires_grad or b.requires_grad:
            raise ValueError("the dispatcher currently supports inference inputs only")
        candidate = self.select(problem, str(a.device), tuple(a.stride()), tuple(b.stride()))
        return candidate.launch(a, b) if candidate is not None else torch.matmul(a, b)
