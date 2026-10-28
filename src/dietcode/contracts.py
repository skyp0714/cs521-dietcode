"""Shared boundary between kernels (Person 1), modeling (2), and dispatch (3)."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Problem:
    op: str
    m: int
    n: int
    k: int
    batch: int = 1
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.op not in {"gemm", "bmm"}:
            raise ValueError("op must be gemm or bmm")
        if any(type(v) is not int or v <= 0 for v in (self.m, self.n, self.k, self.batch)):
            raise ValueError("dimensions must be positive integers")
        if self.op == "gemm" and self.batch != 1:
            raise ValueError("gemm requires batch=1")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported dtype")

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        a, b = (self.m, self.k), (self.k, self.n)
        if self.op == "bmm":
            return (self.batch, *a), (self.batch, *b)
        return a, b

    @property
    def flops(self) -> int:
        return 2 * self.batch * self.m * self.n * self.k


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = 32
    block_n: int = 32
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (
            self.block_m, self.block_n, self.block_k, self.num_warps, self.num_stages
        )):
            raise ValueError("kernel parameters must be positive integers")


@dataclass(frozen=True)
class KernelCandidate:
    name: str
    config: KernelConfig
    # Person 1 must check operation, dtype, device and layout restrictions here.
    supports: Callable[[Problem, str, tuple[int, ...], tuple[int, ...]], bool]
    # Launch must return A @ B, using masked loads/stores for irregular sizes.
    launch: Callable[[Any, Any], Any]


class Predictor(Protocol):
    """Person 2 supplies a model trained for one recorded target device."""

    def predict_us(self, problem: Problem, config: KernelConfig) -> float:
        """Return estimated execution time in microseconds (lower is better)."""
        ...
