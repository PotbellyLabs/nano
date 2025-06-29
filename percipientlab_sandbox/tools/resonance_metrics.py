"""Utility functions for token-level resonance metrics."""
from __future__ import annotations

from typing import List
import numpy as np


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ResonanceMetric:
    """Simple metric placeholder."""

    def score(self, tokens_a: List[str], tokens_b: List[str]) -> float:
        """Return a resonance score between two token lists."""
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(set(tokens_a) & set(tokens_b))
        return overlap / max(len(tokens_a), len(tokens_b))
