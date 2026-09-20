"""Bootstrap confidence intervals over documents.

Resampling happens at the **document** level, never the component level.
Components inside one essay are not independent observations -- essay length,
annotation density and difficulty all correlate within a document -- so
resampling components would understate variance and produce intervals narrow
enough to be misleading.

Two intervals are provided:

* :func:`bootstrap_f1` -- a marginal CI for a single system.
* :func:`paired_bootstrap_delta` -- a CI on the *difference* between two
  systems, computed over the same resampled document indices. This is the one
  that licenses a claim like "the cascade matches Claude-only F1 within CI".
  Checking whether two marginal CIs happen to overlap is a weaker and more
  conservative test, and it is the wrong tool for a paired comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from argmap.metrics.scores import Counts

DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 0
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE
    iterations: int = DEFAULT_ITERATIONS

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "iterations": self.iterations,
        }

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely above or entirely below zero."""
        return self.low > 0.0 or self.high < 0.0


def _as_matrix(per_doc: Sequence[Counts]) -> np.ndarray:
    """Stack per-document counts into an (n, 3) array of tp/fp/fn."""
    if not per_doc:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array([[c.tp, c.fp, c.fn] for c in per_doc], dtype=np.int64)


def _f1_from_totals(totals: np.ndarray) -> np.ndarray:
    """Micro-F1 from an array of summed `[tp, fp, fn]` rows."""
    tp = totals[..., 0].astype(np.float64)
    fp = totals[..., 1].astype(np.float64)
    fn = totals[..., 2].astype(np.float64)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def _resample_indices(n: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(iterations, n))


def bootstrap_f1(
    per_doc: Sequence[Counts],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Percentile bootstrap CI for micro-F1, resampling documents."""
    matrix = _as_matrix(per_doc)
    n = len(matrix)
    point = float(_f1_from_totals(matrix.sum(axis=0))) if n else 0.0
    if n < 2:
        return Interval(point, point, point, confidence, iterations)

    indices = _resample_indices(n, iterations, seed)
    totals = matrix[indices].sum(axis=1)
    samples = _f1_from_totals(totals)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return Interval(point, float(low), float(high), confidence, iterations)


def paired_bootstrap_delta(
    a_per_doc: Sequence[Counts],
    b_per_doc: Sequence[Counts],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """CI on `F1(a) - F1(b)`, resampling the same documents for both systems.

    Both sequences must be aligned: entry *i* of each must describe the same
    document, which is what makes the comparison paired.
    """
    if len(a_per_doc) != len(b_per_doc):
        msg = (
            "paired bootstrap needs aligned per-document counts: "
            f"got {len(a_per_doc)} and {len(b_per_doc)}"
        )
        raise ValueError(msg)

    a = _as_matrix(a_per_doc)
    b = _as_matrix(b_per_doc)
    n = len(a)
    point = float(_f1_from_totals(a.sum(axis=0)) - _f1_from_totals(b.sum(axis=0))) if n else 0.0
    if n < 2:
        return Interval(point, point, point, confidence, iterations)

    indices = _resample_indices(n, iterations, seed)
    deltas = _f1_from_totals(a[indices].sum(axis=1)) - _f1_from_totals(b[indices].sum(axis=1))

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    return Interval(point, float(low), float(high), confidence, iterations)
