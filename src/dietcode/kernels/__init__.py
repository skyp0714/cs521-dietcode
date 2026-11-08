"""Explicit, shared Triton candidates; importing this module needs no GPU packages."""

from functools import partial
from typing import Iterable

from ..contracts import KernelCandidate, KernelConfig, Problem


# Deliberately small initial search space, reused across all shapes and both ops.
# No per-shape autotuning occurs inside a candidate.
DEFAULT_CONFIGS = (
    KernelConfig(32, 32, 32, 4, 2),
    KernelConfig(32, 64, 32, 4, 2),
    KernelConfig(64, 32, 32, 4, 2),
    KernelConfig(64, 64, 32, 4, 3),
)


def _validate_config(config: KernelConfig) -> None:
    if any(v not in (16, 32, 64) for v in
           (config.block_m, config.block_n, config.block_k)):
        raise ValueError("initial Triton tiles must be 16, 32, or 64")
    if config.num_warps not in (4, 8) or config.num_stages not in (2, 3, 4):
        raise ValueError("initial Triton kernels require 4/8 warps and 2/3/4 stages")


def _row_major(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    if len(shape) != len(strides):
        return False
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if stride != expected:
            return False
        expected *= size
    return True


def _supports(problem: Problem, device: str, a_strides: tuple[int, ...],
              b_strides: tuple[int, ...]) -> bool:
    if problem.dtype != "float16" or not (device == "cuda" or device.startswith("cuda:")):
        return False
    if problem.batch > 65535:  # CUDA grid y limit.
        return False
    a_shape, b_shape = problem.input_shapes
    if not (_row_major(a_shape, a_strides) and _row_major(b_shape, b_strides)):
        return False
    try:
        import torch
        import triton  # noqa: F401 -- dependency must exist before advertising support
    except ImportError:
        return False
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(torch.device(device))[0] >= 8
    except (ValueError, RuntimeError, AssertionError):
        return False


def _launch(config: KernelConfig, a, b):
    # Validate direct calls too: Person 2 can time candidates without a dispatcher.
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("inputs must have matching device and dtype")
    if a.ndim != b.ndim or a.ndim not in (2, 3):
        raise ValueError("expected rank-2 GEMM or rank-3 BMM inputs")
    if a.shape[-1] != b.shape[-2] or (a.ndim == 3 and a.shape[0] != b.shape[0]):
        raise ValueError("incompatible matrix or batch dimensions; broadcasting is unsupported")
    if a.requires_grad or b.requires_grad:
        raise ValueError("Triton candidates support inference inputs only")
    problem = Problem("gemm" if a.ndim == 2 else "bmm", a.shape[-2], b.shape[-1],
                      a.shape[-1], a.shape[0] if a.ndim == 3 else 1,
                      str(a.dtype).removeprefix("torch."))
    if not _supports(problem, str(a.device), tuple(a.stride()), tuple(b.stride())):
        raise ValueError("candidate requires row-major FP16 CUDA inputs, Triton, and NVIDIA SM80+")
    from ._triton import launch
    return launch(config, problem, a, b)


def make_candidates(configs: Iterable[KernelConfig] = DEFAULT_CONFIGS) -> tuple[KernelCandidate, ...]:
    """Build candidates without importing torch/Triton or compiling kernels.

    Each candidate supports both GEMM and BMM. Unsupported inputs are rejected
    by direct launch, or handled by RuntimeDispatcher's existing fallback.
    """
    candidates = []
    names = set()
    for config in configs:
        _validate_config(config)
        name = (f"triton_m{config.block_m}_n{config.block_n}_k{config.block_k}"
                f"_w{config.num_warps}_s{config.num_stages}")
        if name in names:
            raise ValueError("duplicate kernel configuration")
        names.add(name)
        candidates.append(KernelCandidate(name, config, _supports, partial(_launch, config)))
    return tuple(candidates)
