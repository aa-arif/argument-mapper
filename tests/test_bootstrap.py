"""Bootstrap tests.

These check the properties that make the interval trustworthy rather than a
specific numeric output: determinism under a fixed seed, coverage of the point
estimate, correct handling of degenerate inputs, and -- the one that matters
most for this project's headline claim -- that a paired comparison of a system
against itself produces a delta of exactly zero with a zero-width interval.
"""

from __future__ import annotations

import pytest

from argmap.metrics.bootstrap import bootstrap_f1, paired_bootstrap_delta
from argmap.metrics.scores import Counts


def _perfect(n: int) -> list[Counts]:
    return [Counts(tp=5, fp=0, fn=0) for _ in range(n)]


def _mediocre(n: int) -> list[Counts]:
    return [Counts(tp=3, fp=2, fn=2) for _ in range(n)]


def test_point_estimate_is_micro_f1() -> None:
    """20 docs of tp=3, fp=2, fn=2 -> tp=60, fp=40, fn=40.

    P = 60/100 = 0.6, R = 60/100 = 0.6, F1 = 0.6.
    """
    interval = bootstrap_f1(_mediocre(20), iterations=200)
    assert interval.point == pytest.approx(0.6)


def test_interval_contains_the_point_estimate() -> None:
    interval = bootstrap_f1(_mediocre(30), iterations=500)
    assert interval.low <= interval.point <= interval.high


def test_is_deterministic_for_a_fixed_seed() -> None:
    a = bootstrap_f1(_mediocre(30), iterations=500, seed=7)
    b = bootstrap_f1(_mediocre(30), iterations=500, seed=7)
    assert (a.low, a.point, a.high) == (b.low, b.point, b.high)


def test_homogeneous_documents_give_a_degenerate_interval() -> None:
    """Every document identical means every resample is identical."""
    interval = bootstrap_f1(_perfect(25), iterations=300)
    assert interval.point == pytest.approx(1.0)
    assert interval.low == pytest.approx(1.0)
    assert interval.high == pytest.approx(1.0)


def test_heterogeneous_documents_give_a_non_degenerate_interval() -> None:
    counts = [Counts(tp=5, fp=0, fn=0)] * 10 + [Counts(tp=0, fp=5, fn=5)] * 10
    interval = bootstrap_f1(counts, iterations=2000, seed=1)
    assert interval.low < interval.point < interval.high


def test_more_documents_narrows_the_interval() -> None:
    pattern = [Counts(tp=5, fp=0, fn=0), Counts(tp=0, fp=5, fn=5)]
    narrow = bootstrap_f1(pattern * 100, iterations=2000, seed=3)
    wide = bootstrap_f1(pattern * 5, iterations=2000, seed=3)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_single_document_yields_a_degenerate_interval() -> None:
    interval = bootstrap_f1([Counts(tp=1, fp=1, fn=1)], iterations=100)
    assert interval.low == interval.point == interval.high


def test_no_documents_scores_zero() -> None:
    assert bootstrap_f1([], iterations=100).point == 0.0


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def test_paired_delta_against_itself_is_exactly_zero() -> None:
    """The property that makes the paired test the right tool.

    Comparing a system to itself must give a delta of 0 with no spread, even
    though the marginal interval for that system is wide.
    """
    counts = [Counts(tp=5, fp=0, fn=0)] * 10 + [Counts(tp=0, fp=5, fn=5)] * 10
    marginal = bootstrap_f1(counts, iterations=2000, seed=2)
    delta = paired_bootstrap_delta(counts, counts, iterations=2000, seed=2)

    assert marginal.high > marginal.low, "sanity: the marginal interval is wide"
    assert delta.point == pytest.approx(0.0)
    assert delta.low == pytest.approx(0.0)
    assert delta.high == pytest.approx(0.0)
    assert not delta.excludes_zero


def test_paired_delta_is_positive_when_the_first_system_is_better() -> None:
    delta = paired_bootstrap_delta(_perfect(30), _mediocre(30), iterations=1000, seed=4)
    assert delta.point == pytest.approx(1.0 - 0.6)
    assert delta.excludes_zero


def test_paired_delta_is_antisymmetric() -> None:
    forward = paired_bootstrap_delta(_perfect(20), _mediocre(20), iterations=500, seed=5)
    backward = paired_bootstrap_delta(_mediocre(20), _perfect(20), iterations=500, seed=5)
    assert forward.point == pytest.approx(-backward.point)


def test_paired_delta_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned per-document counts"):
        paired_bootstrap_delta(_perfect(5), _perfect(6))


def test_excludes_zero_is_false_for_a_straddling_interval() -> None:
    a = [Counts(tp=3, fp=2, fn=2)] * 10 + [Counts(tp=4, fp=1, fn=1)] * 10
    b = [Counts(tp=4, fp=1, fn=1)] * 10 + [Counts(tp=3, fp=2, fn=2)] * 10
    delta = paired_bootstrap_delta(a, b, iterations=2000, seed=6)
    assert not delta.excludes_zero
