"""Metric tests against cases computed by hand.

The fixture document is:

    "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
     ^0               ^18                 ^38

    tokens: 0=Alpha 1=beta 2=gamma 3=Delta 4=epsilon 5=zeta 6=Eta 7=theta 8=iota

    gold T1 Premise    [0, 17)  tokens {0,1,2}
    gold T2 Claim      [18, 37) tokens {3,4,5}
    gold T3 MajorClaim [38, 53) tokens {6,7,8}
    gold R1 T1 -supports-> T2

Each expectation below is derived in the test's own docstring, not read off a
previous run. That is the point of the file: if the matcher changes behaviour,
these numbers are an independent check rather than a regression snapshot.
"""

from __future__ import annotations

import pytest

from argmap.metrics.matching import (
    MatchCriterion,
    match_components,
    overlap_ratio,
    token_spans,
    tokens_in_span,
)
from argmap.metrics.scores import Counts, score_document, score_relations
from argmap.schema import Component, Document, Prediction, Relation

EXACT_TYPED = MatchCriterion(span="exact", typed=True)
EXACT_UNTYPED = MatchCriterion(span="exact", typed=False)
OVERLAP_TYPED = MatchCriterion(span="overlap", typed=True)
OVERLAP_UNTYPED = MatchCriterion(span="overlap", typed=False)


def _prediction() -> Prediction:
    """Three predictions chosen so each criterion gives a different answer.

    P1 [0,17)  Premise -- identical to T1
    P2 [18,31) Claim   -- "Delta epsilon", 2 of T2's 3 tokens
    P3 [38,53) Premise -- T3's exact span but the wrong type
    """
    return Prediction(
        doc_id="fixture001",
        components=[
            Component(id="P1", type="Premise", start=0, end=17, text="Alpha beta gamma."),
            Component(id="P2", type="Claim", start=18, end=31, text="Delta epsilon"),
            Component(id="P3", type="Premise", start=38, end=53, text="Eta theta iota."),
        ],
        relations=[Relation(id="Q1", src="P1", tgt="P2", type="supports")],
    )


# ---------------------------------------------------------------------------
# Tokenisation primitives
# ---------------------------------------------------------------------------


def test_token_spans_finds_nine_words(fixture_text: str) -> None:
    spans = token_spans(fixture_text)
    assert len(spans) == 9
    assert fixture_text[spans[0][0] : spans[0][1]] == "Alpha"
    assert fixture_text[spans[8][0] : spans[8][1]] == "iota"


def test_tokens_in_span_claims_partially_covered_words(fixture_text: str) -> None:
    """A span ending mid-word still claims that word.

    [0, 8) covers "Alpha be" -- all of token 0 and part of token 1.
    """
    spans = token_spans(fixture_text)
    assert tokens_in_span(spans, 0, 8) == frozenset({0, 1})


def test_overlap_ratio_jaccard_vs_gold_recall() -> None:
    """gold {0,1,2,3}, pred {0,1}: Jaccard is 2/4; gold-recall is also 2/4.

    gold {0,1,2,3,4}, pred {0,1}: Jaccard 2/5 = 0.4, gold-recall 2/5 = 0.4.
    The measures diverge when the prediction is *larger* than the gold span:
    gold {0,1}, pred {0,1,2,3} gives Jaccard 2/4 = 0.5 but gold-recall 2/2 = 1.0.
    """
    assert overlap_ratio(frozenset({0, 1, 2, 3}), frozenset({0, 1})) == pytest.approx(0.5)
    assert overlap_ratio(frozenset({0, 1, 2, 3, 4}), frozenset({0, 1})) == pytest.approx(0.4)

    gold, pred = frozenset({0, 1}), frozenset({0, 1, 2, 3})
    assert overlap_ratio(gold, pred, "jaccard") == pytest.approx(0.5)
    assert overlap_ratio(gold, pred, "gold_recall") == pytest.approx(1.0)


def test_overlap_at_exactly_the_threshold_matches(fixture_text: str) -> None:
    """Jaccard of exactly 0.5 must match, since the criterion is `>=`.

    gold [0,37) covers tokens {0..5}; pred [0,17) covers {0,1,2}.
    Jaccard = 3/6 = 0.5.
    """
    gold = [Component(id="G", type="Claim", start=0, end=37, text=fixture_text[0:37])]
    pred = [Component(id="P", type="Claim", start=0, end=17, text=fixture_text[0:17])]
    assert match_components(gold, pred, fixture_text, OVERLAP_TYPED) == [(0, 0)]

    # One token less in the prediction drops it to 2/6 = 0.33 and it must not match.
    pred_short = [Component(id="P", type="Claim", start=0, end=10, text=fixture_text[0:10])]
    assert match_components(gold, pred_short, fixture_text, OVERLAP_TYPED) == []


# ---------------------------------------------------------------------------
# Component scoring, all four criteria
# ---------------------------------------------------------------------------


def test_exact_typed(gold_doc: Document) -> None:
    """P1 matches T1. P2 has no exact span. P3 has T3's span but wrong type.

    tp=1, fp=2, fn=2 -> P = 1/3, R = 1/3, F1 = 1/3.
    """
    score = score_document(gold_doc, _prediction(), EXACT_TYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (1, 2, 2)
    assert score.components.precision == pytest.approx(1 / 3)
    assert score.components.recall == pytest.approx(1 / 3)
    assert score.components.f1 == pytest.approx(1 / 3)


def test_exact_untyped(gold_doc: Document) -> None:
    """Ignoring type, P3 now matches T3 on span. P2 still has no exact span.

    tp=2, fp=1, fn=1 -> P = R = F1 = 2/3.
    """
    score = score_document(gold_doc, _prediction(), EXACT_UNTYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (2, 1, 1)
    assert score.components.f1 == pytest.approx(2 / 3)


def test_overlap_typed(gold_doc: Document) -> None:
    """P2 now matches T2 (Jaccard 2/3), but P3 is still the wrong type.

    tp=2, fp=1, fn=1 -> F1 = 2/3.
    """
    score = score_document(gold_doc, _prediction(), OVERLAP_TYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (2, 1, 1)
    assert score.components.f1 == pytest.approx(2 / 3)


def test_overlap_untyped(gold_doc: Document) -> None:
    """Every prediction now finds its gold component: tp=3, fp=0, fn=0, F1=1."""
    score = score_document(gold_doc, _prediction(), OVERLAP_UNTYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (3, 0, 0)
    assert score.components.f1 == pytest.approx(1.0)


def test_per_type_breakdown(gold_doc: Document) -> None:
    """Under exact/typed only the Premise T1 is found.

    Premise: T1 matched (tp=1); P3 is an unmatched Premise prediction (fp=1).
    Claim:   T2 unmatched (fn=1); P2 unmatched prediction (fp=1).
    MajorClaim: T3 unmatched (fn=1).
    """
    score = score_document(gold_doc, _prediction(), EXACT_TYPED)
    by_type = score.components_by_type
    assert by_type["Premise"] == Counts(tp=1, fp=1, fn=0)
    assert by_type["Claim"] == Counts(tp=0, fp=1, fn=1)
    assert by_type["MajorClaim"] == Counts(tp=0, fp=0, fn=1)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def test_relation_correct_when_both_endpoints_match(gold_doc: Document) -> None:
    """Under overlap/typed, P1->T1 and P2->T2, and Q1's type matches R1's.

    tp=1, fp=0, fn=0 -> F1 = 1.
    """
    score = score_document(gold_doc, _prediction(), OVERLAP_TYPED)
    assert (score.relations.tp, score.relations.fp, score.relations.fn) == (1, 0, 0)
    assert score.relations.f1 == pytest.approx(1.0)


def test_relation_fails_when_an_endpoint_is_unmatched(gold_doc: Document) -> None:
    """Under exact/typed, P2 never matches, so Q1 has a dangling endpoint.

    tp=0, fp=1, fn=1 -> F1 = 0.
    """
    score = score_document(gold_doc, _prediction(), EXACT_TYPED)
    assert (score.relations.tp, score.relations.fp, score.relations.fn) == (0, 1, 1)
    assert score.relations.f1 == pytest.approx(0.0)


def test_relation_between_hallucinated_components_gets_no_credit(gold_doc: Document) -> None:
    """Components that match nothing cannot support a correct relation."""
    pred = Prediction(
        doc_id="fixture001",
        components=[
            Component(id="X1", type="Premise", start=0, end=5, text="Alpha"),
            Component(id="X2", type="Claim", start=48, end=52, text="iota"),
        ],
        relations=[Relation(id="Q9", src="X1", tgt="X2", type="supports")],
    )
    score = score_document(gold_doc, pred, OVERLAP_TYPED)
    assert score.relations.tp == 0
    assert score.relations.fp == 1
    assert score.relations.fn == 1


def test_relation_type_ignored_when_untyped(gold_doc: Document) -> None:
    """An `attacks` prediction where gold says `supports` is wrong when typed...

    ...and right when untyped, provided the endpoints match.
    """
    pred = _prediction().model_copy(
        update={"relations": [Relation(id="Q1", src="P1", tgt="P2", type="attacks")]}
    )
    typed = score_document(gold_doc, pred, OVERLAP_TYPED)
    untyped = score_document(gold_doc, pred, OVERLAP_UNTYPED)
    assert typed.relations.tp == 0
    assert untyped.relations.tp == 1
    assert untyped.relations_by_type == {}, "no per-type split is meaningful when untyped"


def test_duplicate_predicted_relation_counts_once(gold_doc: Document) -> None:
    """Two copies of the same correct edge: one true positive, one false positive."""
    pred = _prediction().model_copy(
        update={
            "relations": [
                Relation(id="Q1", src="P1", tgt="P2", type="supports"),
                Relation(id="Q2", src="P1", tgt="P2", type="supports"),
            ]
        }
    )
    score = score_document(gold_doc, pred, OVERLAP_TYPED)
    assert (score.relations.tp, score.relations.fp, score.relations.fn) == (1, 1, 0)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_prediction_is_all_false_negatives(gold_doc: Document) -> None:
    score = score_document(gold_doc, Prediction(doc_id=gold_doc.doc_id), OVERLAP_TYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (0, 0, 3)
    assert score.components.precision == 0.0
    assert score.components.f1 == 0.0


def test_empty_gold_makes_every_prediction_a_false_positive(fixture_text: str) -> None:
    empty = Document(doc_id="d", source="aae-v2", split="test", text=fixture_text)
    score = score_document(empty, _prediction(), OVERLAP_TYPED)
    assert (score.components.tp, score.components.fp, score.components.fn) == (0, 3, 0)
    assert score.components.recall == 0.0
    assert score.components.f1 == 0.0


def test_both_empty_scores_zero_not_nan(fixture_text: str) -> None:
    empty = Document(doc_id="d", source="aae-v2", split="test", text=fixture_text)
    score = score_document(empty, Prediction(doc_id="d"), OVERLAP_TYPED)
    assert score.components.f1 == 0.0
    assert score.components.precision == 0.0
    assert score.components.recall == 0.0


def test_duplicate_predictions_match_at_most_one_gold(gold_doc: Document) -> None:
    """Two identical predictions for T1: one matches, the other is a false positive."""
    pred = Prediction(
        doc_id="fixture001",
        components=[
            Component(id="A", type="Premise", start=0, end=17, text="Alpha beta gamma."),
            Component(id="B", type="Premise", start=0, end=17, text="Alpha beta gamma."),
        ],
    )
    for criterion in (EXACT_TYPED, OVERLAP_TYPED):
        score = score_document(gold_doc, pred, criterion)
        assert score.components.tp == 1, criterion.label
        assert score.components.fp == 1, criterion.label


def test_matching_is_one_to_one_and_deterministic(fixture_text: str) -> None:
    """One prediction spanning two gold components can only claim one of them."""
    gold = [
        Component(id="G1", type="Claim", start=0, end=17, text=fixture_text[0:17]),
        Component(id="G2", type="Claim", start=18, end=37, text=fixture_text[18:37]),
    ]
    pred = [Component(id="P1", type="Claim", start=0, end=37, text=fixture_text[0:37])]
    pairs = match_components(gold, pred, fixture_text, OVERLAP_TYPED)
    assert len(pairs) <= 1
    assert match_components(gold, pred, fixture_text, OVERLAP_TYPED) == pairs


def test_score_relations_handles_no_relations_either_side(gold_doc: Document) -> None:
    counts, by_type = score_relations([], [], gold_doc.components, [], [], typed=True)
    assert counts == Counts(tp=0, fp=0, fn=0)
    assert counts.f1 == 0.0
    assert by_type == {}
