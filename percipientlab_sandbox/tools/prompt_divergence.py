"""Utilities for running prompt divergence tests across models."""
from __future__ import annotations

from typing import Iterable, Tuple


def divergence(prompt: str, outputs: Iterable[str]) -> Tuple[str, float]:
    """Return the most divergent output and a simple variance score."""
    if not outputs:
        return "", 0.0
    lengths = [len(o) for o in outputs]
    average = sum(lengths) / len(lengths)
    variance = sum((l - average) ** 2 for l in lengths) / len(lengths)
    most_divergent = max(outputs, key=lambda o: abs(len(o) - average))
    return most_divergent, variance
