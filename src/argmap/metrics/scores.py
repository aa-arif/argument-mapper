"""Precision, recall and F1 for components and relations.

Scores are computed and kept **per document**, then summed. Keeping the counts
disaggregated is not incidental: the bootstrap in `argmap.metrics.bootstrap`
resamples documents, which it can only do if the per-document contribution to
every count is still available.

Relation scoring is conditional on component matching. A predicted relation is
correct only when both of its endpoints matched gold components under the
active criterion *and* a gold relation connects those two gold components. A
system that invents a plausible edge between two components it hallucinated
gets no credit for it, which is the behaviour you want.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from argmap.metrics.matching import MatchCriterion, match_components
from argmap.schema import Component, Document, Prediction, Relation, RelationType


@dataclass(frozen=True)
class Counts:
    """True positives, false positives and false negatives for one comparison."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        """Number of gold items -- what a per-type row's CI width is driven by."""
        return self.tp + self.fn

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def sum_counts(counts: Iterable[Counts]) -> Counts:
    total = Counts()
    for c in counts:
        total = total + c
    return total


@dataclass(frozen=True)
class DocumentScore:
    """Everything measured for a single document under a single criterion."""

    doc_id: str
    components: Counts
    relations: Counts
    components_by_type: dict[str, Counts]
    relations_by_type: dict[str, Counts]


def score_components(
    gold: Sequence[Component],
    pred: Sequence[Component],
    doc_text: str,
    criterion: MatchCriterion,
) -> tuple[Counts, dict[str, Counts], list[tuple[int, int]]]:
    """Score components and return the pairing so relations can reuse it."""
    pairs = match_components(gold, pred, doc_text, criterion)
    matched_gold = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}

    overall = Counts(
        tp=len(pairs),
        fp=len(pred) - len(matched_pred),
        fn=len(gold) - len(matched_gold),
    )

    by_type: dict[str, Counts] = {}
    for i, g in enumerate(gold):
        key = g.type or "untyped"
        hit = i in matched_gold
        by_type[key] = by_type.get(key, Counts()) + Counts(tp=int(hit), fn=int(not hit))
    for j, p in enumerate(pred):
        if j in matched_pred:
            continue
        key = p.type or "untyped"
        by_type[key] = by_type.get(key, Counts()) + Counts(fp=1)

    return overall, by_type, pairs


def score_relations(
    gold_relations: Sequence[Relation],
    pred_relations: Sequence[Relation],
    gold: Sequence[Component],
    pred: Sequence[Component],
    pairs: Sequence[tuple[int, int]],
    *,
    typed: bool,
) -> tuple[Counts, dict[str, Counts]]:
    """Score relations, given an already-computed component pairing.

    `typed` here refers to the *relation* type (supports/attacks), which is
    independent of whether component types had to agree for the endpoints to
    match.
    """
    pred_to_gold: dict[str, str] = {pred[j].id: gold[i].id for i, j in pairs}

    def key(rel: Relation, src: str, tgt: str) -> tuple[str, str, str]:
        return (src, tgt, rel.type if typed else "*")

    gold_keys: dict[tuple[str, str, str], int] = {}
    for rel in gold_relations:
        k = key(rel, rel.src, rel.tgt)
        gold_keys[k] = gold_keys.get(k, 0) + 1

    remaining = dict(gold_keys)
    tp = 0
    fp = 0
    matched_gold_types: list[RelationType] = []

    for rel in pred_relations:
        src = pred_to_gold.get(rel.src)
        tgt = pred_to_gold.get(rel.tgt)
        if src is None or tgt is None:
            fp += 1
            continue
        k = key(rel, src, tgt)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1
            tp += 1
            matched_gold_types.append(rel.type)
        else:
            fp += 1

    overall = Counts(tp=tp, fp=fp, fn=len(gold_relations) - tp)

    if not typed:
        # Matching ignored relation type, so there is no per-type split to
        # report. Returning empty says that; returning zero-tp rows would read
        # as "the model got every attack wrong", which is a different claim.
        return overall, {}

    hits: dict[str, int] = {}
    for t in matched_gold_types:
        hits[t] = hits.get(t, 0) + 1

    gold_by_type: dict[str, int] = {}
    for rel in gold_relations:
        gold_by_type[rel.type] = gold_by_type.get(rel.type, 0) + 1
    pred_by_type: dict[str, int] = {}
    for rel in pred_relations:
        pred_by_type[rel.type] = pred_by_type.get(rel.type, 0) + 1

    by_type: dict[str, Counts] = {}
    for rel_type in set(gold_by_type) | set(pred_by_type):
        type_tp = hits.get(rel_type, 0)
        by_type[rel_type] = Counts(
            tp=type_tp,
            fp=max(0, pred_by_type.get(rel_type, 0) - type_tp),
            fn=max(0, gold_by_type.get(rel_type, 0) - type_tp),
        )

    return overall, by_type


def score_document(
    doc: Document,
    pred: Prediction,
    criterion: MatchCriterion,
) -> DocumentScore:
    """Score one document under one criterion."""
    comp_counts, comp_by_type, pairs = score_components(
        doc.components, pred.components, doc.text, criterion
    )
    rel_counts, rel_by_type = score_relations(
        doc.relations,
        pred.relations,
        doc.components,
        pred.components,
        pairs,
        typed=criterion.typed,
    )
    return DocumentScore(
        doc_id=doc.doc_id,
        components=comp_counts,
        relations=rel_counts,
        components_by_type=comp_by_type,
        relations_by_type=rel_by_type,
    )


def score_corpus(
    docs: Sequence[Document],
    predictions: dict[str, Prediction],
    criterion: MatchCriterion,
) -> list[DocumentScore]:
    """Score every document, using an empty prediction where one is missing.

    A missing prediction counts as all-false-negatives rather than being
    skipped -- dropping it would quietly inflate the score of a system that
    failed on hard documents.
    """
    return [
        score_document(doc, predictions.get(doc.doc_id, Prediction(doc_id=doc.doc_id)), criterion)
        for doc in docs
    ]
