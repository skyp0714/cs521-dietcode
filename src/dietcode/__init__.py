"""DietCode-inspired course project; GPU kernels and learned tuning are pending."""

from .contracts import KernelCandidate, KernelConfig, Predictor, Problem
from .dispatch import RuntimeDispatcher

__all__ = ["KernelCandidate", "KernelConfig", "Predictor", "Problem", "RuntimeDispatcher"]
