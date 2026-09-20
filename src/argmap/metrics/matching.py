"""Matching predicted argument components against gold components.

Every metric in this project rests on one question: does this predicted span
count as the same component as that gold span? Four criteria are reported,
crossing two independent choices:

* **span**: `exact` requires identical character offsets; `overlap` requires
  the two spans to share at least `threshold` of their tokens.
* **typed**: whether the component type must also agree.

The overlap measure is **Jaccard over token indices**, not the
recall-oriented `|intersection| / |gold|` that some papers use. The asymmetric
form is trivially gamed: a system that predicts one component spanning the
whole document matches every gold component at 100%. Jaccard punishes that,
so it is the honest default here. `OverlapMode` keeps the alternative
available for comparison against published numbers that use it.

Matching is greedy, highest-overlap-first, and one-to-one: a gold component can
absorb at most one prediction and vice versa. Ties are broken by document order
so results are deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from argmap.schema import Component

SpanCriterion = Literal["exact", "overlap"]
OverlapMode = Literal["jaccard", "gold_recall"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class MatchCriterion:
    """How strict a match has to be."""

    span: SpanCriterion = "overlap"
    typed: bool = True
    threshold: float = 0.5
    overlap_mode: OverlapMode = "jaccard"

    @property
    def label(self) -> str:
        span = "exact" if self.span == "exact" else f"overlap>={self.threshold:g}"
        return f"{span}/{'typed' if self.typed else 'untyped'}"


#: The four criteria reported in every results table.
STANDARD_CRITERIA: tuple[MatchCriterion, ...] = (
    MatchCriterion(span="exact", typed=True),
    MatchCriterion(span="exact", typed=False),
    MatchCriterion(span="overlap", typed=True),
    MatchCriterion(span="overlap", typed=False),
)


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every word token in `text`, in order."""
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]


def tokens_in_span(spans: Sequence[tuple[int, int]], start: int, end: int) -> frozenset[int]:
    """Indices of tokens that overlap the character range `[start, end)`.

    A token counts as inside the span if it overlaps it at all, so a component
    boundary landing mid-word still claims that word rather than silently
    dropping it.
    """
    return frozenset(i for i, (s, e) in enumerate(spans) if s < end and e > start)


def overlap_ratio(
    gold_tokens: frozenset[int],
    pred_tokens: frozenset[int],
    mode: OverlapMode = "jaccard",
) -> float:
    """Token overlap between a gold and a predicted span."""
    if not gold_tokens and not pred_tokens:
        return 1.0
    intersection = len(gold_tokens & pred_tokens)
    if intersection == 0:
        return 0.0
    if mode == "jaccard":
        return intersection / len(gold_tokens | pred_tokens)
    return intersection / len(gold_tokens)


def _types_agree(gold: Component, pred: Component, criterion: MatchCriterion) -> bool:
    if not criterion.typed:
        return True
    return gold.type == pred.type


def match_components(
    gold: Sequence[Component],
    pred: Sequence[Component],
    doc_text: str,
    criterion: MatchCriterion,
) -> list[tuple[int, int]]:
    """Greedily pair gold and predicted components under `criterion`.

    Returns `(gold_index, pred_index)` pairs. Unpaired gold components are false
    negatives; unpaired predictions are false positives.
    """
    if not gold or not pred:
        return []

    if criterion.span == "exact":
        return _match_exact(gold, pred, criterion)
    return _match_overlap(gold, pred, doc_text, criterion)


def _match_exact(
    gold: Sequence[Component],
    pred: Sequence[Component],
    criterion: MatchCriterion,
) -> list[tuple[int, int]]:
    by_span: dict[tuple[int, int], list[int]] = {}
    for j, p in enumerate(pred):
        by_span.setdefault((p.start, p.end), []).append(j)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, g in enumerate(gold):
        for j in by_span.get((g.start, g.end), ()):
            if j in used or not _types_agree(g, pred[j], criterion):
                continue
            used.add(j)
            pairs.append((i, j))
            break
    return pairs


def _match_overlap(
    gold: Sequence[Component],
    pred: Sequence[Component],
    doc_text: str,
    criterion: MatchCriterion,
) -> list[tuple[int, int]]:
    spans = token_spans(doc_text)
    gold_tokens = [tokens_in_span(spans, g.start, g.end) for g in gold]
    pred_tokens = [tokens_in_span(spans, p.start, p.end) for p in pred]

    candidates: list[tuple[float, int, int]] = []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if not _types_agree(g, p, criterion):
                continue
            ratio = overlap_ratio(gold_tokens[i], pred_tokens[j], criterion.overlap_mode)
            if ratio >= criterion.threshold:
                candidates.append((ratio, i, j))

    # Highest overlap first; ties broken by document order for determinism.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_gold or j in used_pred:
            continue
        used_gold.add(i)
        used_pred.add(j)
        pairs.append((i, j))

    pairs.sort()
    return pairs
