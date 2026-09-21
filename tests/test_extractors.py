"""Tests for cost accounting, response caching, and prompt construction.

No network. Everything here runs on synthetic data so CI can execute it with
no API key and no money at stake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argmap.cost import (
    PRICING,
    LedgerEntry,
    SpendCapExceededError,
    SpendLedger,
    Usage,
    price,
)
from argmap.extractors.cache import ResponseCache, cache_key
from argmap.prompts import (
    document_to_graph,
    exemplar_turns,
    prompt_fingerprint,
    select_exemplars,
    write_exemplars,
)
from argmap.schema import Document

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_price_is_computed_from_the_published_rate() -> None:
    """Haiku 4.5 is $1/MTok in, $5/MTok out.

    1,000,000 in + 100,000 out = $1.00 + $0.50 = $1.50.
    """
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert price(usage, "claude-haiku-4-5") == pytest.approx(1.50)


def test_cache_reads_cost_a_tenth_of_input() -> None:
    """1M cached-read tokens on Haiku: $1.00 x 0.1 = $0.10."""
    usage = Usage(cache_read_tokens=1_000_000)
    assert price(usage, "claude-haiku-4-5") == pytest.approx(0.10)


def test_cache_writes_cost_a_25_percent_premium() -> None:
    usage = Usage(cache_write_tokens=1_000_000)
    assert price(usage, "claude-haiku-4-5") == pytest.approx(1.25)


def test_batch_halves_the_price() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    full = price(usage, "claude-sonnet-5")
    assert price(usage, "claude-sonnet-5", batch=True) == pytest.approx(full / 2)


def test_unknown_model_prices_at_zero_rather_than_guessing() -> None:
    assert price(Usage(input_tokens=1_000_000), "some-future-model") == 0.0


def test_every_model_used_in_this_project_has_a_price() -> None:
    assert {"claude-sonnet-5", "claude-haiku-4-5"} <= set(PRICING)


# ---------------------------------------------------------------------------
# Spend ledger
# ---------------------------------------------------------------------------


def _entry(dollars: float, doc_id: str = "d1") -> LedgerEntry:
    return LedgerEntry(
        timestamp="",
        model="claude-haiku-4-5",
        doc_id=doc_id,
        usage=Usage(input_tokens=10, output_tokens=10),
        dollars=dollars,
        latency_s=0.1,
        batch=False,
        cached=False,
    )


def test_ledger_accumulates_spend(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=10.0)
    ledger.record(_entry(1.25))
    ledger.record(_entry(2.75))
    assert ledger.spent_usd == pytest.approx(4.0)
    assert ledger.remaining_usd == pytest.approx(6.0)


def test_ledger_survives_a_restart(tmp_path: Path) -> None:
    """The cap must not silently reset when a script is re-run.

    A budget that resets per process is a budget that does not exist, since
    re-running a sweep is exactly when overspend happens.
    """
    path = tmp_path / "ledger.jsonl"
    first = SpendLedger(path, cap_usd=10.0)
    first.record(_entry(6.0))

    second = SpendLedger(path, cap_usd=10.0)
    assert second.spent_usd == pytest.approx(6.0)
    with pytest.raises(SpendCapExceededError):
        second.reserve(5.0)


def test_reserve_raises_before_the_cap_is_crossed(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=1.0)
    ledger.record(_entry(0.9))
    ledger.reserve(0.05)  # fits
    with pytest.raises(SpendCapExceededError, match="spend cap would be exceeded"):
        ledger.reserve(0.2)


def test_ledger_tolerates_a_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"dollars": 1.0}\nnot json\n{"dollars": 2.0}\n', encoding="utf-8")
    assert SpendLedger(path, cap_usd=10.0).spent_usd == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def test_cache_round_trips(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key(model="m", prompt_version="v1", document_text="hello")
    assert cache.get(key) is None
    cache.put(key, {"graph": {"components": []}})
    assert cache.get(key) == {"graph": {"components": []}}
    assert key in cache


def test_cache_key_changes_with_every_input_that_changes_the_answer() -> None:
    base = {"model": "m", "prompt_version": "v1", "document_text": "hello"}
    key = cache_key(**base)  # type: ignore[arg-type]
    assert cache_key(**{**base, "model": "other"}) != key  # type: ignore[arg-type]
    assert cache_key(**{**base, "prompt_version": "v2"}) != key  # type: ignore[arg-type]
    assert cache_key(**{**base, "document_text": "goodbye"}) != key  # type: ignore[arg-type]
    assert cache_key(**base, params={"max_tokens": 10}) != key  # type: ignore[arg-type]


def test_cache_key_is_order_independent_for_params() -> None:
    a = cache_key(model="m", prompt_version="v1", document_text="x", params={"a": 1, "b": 2})
    b = cache_key(model="m", prompt_version="v1", document_text="x", params={"b": 2, "a": 1})
    assert a == b


def test_corrupt_cache_blob_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key(model="m", prompt_version="v1", document_text="hello")
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"graph": {"compo', encoding="utf-8")  # interrupted write
    assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_exemplars_renumber_component_ids(gold_doc: Document) -> None:
    """Worked examples must not teach the model to emit gold ids.

    Predicted ids would then collide with gold ids meaning different
    components, making every error analysis ambiguous.
    """
    graph = document_to_graph(gold_doc)
    assert [c.id for c in graph.components] == ["c1", "c2", "c3"]
    assert [(r.src, r.tgt) for r in graph.relations] == [("c1", "c2")]


def test_fingerprint_tracks_exemplar_content_not_just_ids(
    gold_doc: Document, tmp_path: Path
) -> None:
    """Regression: the cache key must miss when rendering changes.

    Keying on document ids alone let a prompt change (renumbering component
    ids) reuse a response generated by the previous prompt.
    """
    path = tmp_path / "exemplars.json"
    write_exemplars([gold_doc], path)
    exemplars = json.loads(path.read_text(encoding="utf-8"))

    before = prompt_fingerprint(exemplars)

    mutated = json.loads(json.dumps(exemplars))
    mutated[0]["graph"]["components"][0]["text"] = "something else entirely"
    assert prompt_fingerprint(mutated) != before

    same_docs_different_render = json.loads(json.dumps(exemplars))
    same_docs_different_render[0]["graph"]["components"][0]["id"] = "zzz"
    assert prompt_fingerprint(same_docs_different_render) != before


def test_exemplar_turns_alternate_user_and_assistant(gold_doc: Document, tmp_path: Path) -> None:
    path = tmp_path / "exemplars.json"
    write_exemplars([gold_doc], path)
    turns = exemplar_turns(json.loads(path.read_text(encoding="utf-8")))
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert json.loads(str(turns[1]["content"]))["components"][0]["id"] == "c1"


def test_exemplars_come_only_from_train(gold_doc: Document) -> None:
    """Drawing an exemplar from val or test would invalidate every later number."""
    val = gold_doc.model_copy(update={"doc_id": "v1", "split": "val"})
    test = gold_doc.model_copy(update={"doc_id": "t1", "split": "test"})
    chosen = select_exemplars([gold_doc, val, test], k=3)
    assert {d.doc_id for d in chosen} == {gold_doc.doc_id}


def test_exemplar_selection_is_deterministic(gold_doc: Document) -> None:
    docs = [gold_doc.model_copy(update={"doc_id": f"essay{i:03d}"}) for i in range(20)]
    assert [d.doc_id for d in select_exemplars(docs, k=2)] == [
        d.doc_id for d in select_exemplars(docs, k=2)
    ]


# ---------------------------------------------------------------------------
# Constrained-decoding schema
# ---------------------------------------------------------------------------


def test_bounded_schema_adds_array_limits() -> None:
    """Constrained decoding guarantees syntax, not termination.

    A model emitting non-unique component ids can generate relations between
    them forever and stay schema-valid, so generation only stops at the token
    cap -- producing truncated JSON that looks like a constraint failure.
    `maxItems` makes termination something the grammar enforces.
    """
    from argmap.prompts import MAX_COMPONENTS, MAX_RELATIONS, bounded_graph_schema

    schema = bounded_graph_schema()
    assert schema["properties"]["components"]["maxItems"] == MAX_COMPONENTS
    assert schema["properties"]["relations"]["maxItems"] == MAX_RELATIONS


def test_bounded_schema_leaves_the_model_schema_untouched() -> None:
    """The Claude route's cache key covers its schema.

    Mutating `RawGraph` in place would invalidate every cached API response and
    re-spend the budget for a change only the local route needs.
    """
    import json as _json

    from argmap.prompts import RawGraph, bounded_graph_schema

    bounded_graph_schema()
    assert "maxItems" not in _json.dumps(RawGraph.model_json_schema())


def test_bounded_schema_limits_exceed_the_corpus() -> None:
    """Bounds must constrain runaway output only, never a real document.

    The densest AAE essay has 28 components and 20 relations.
    """
    from argmap.prompts import MAX_COMPONENTS, MAX_RELATIONS

    assert MAX_COMPONENTS > 28
    assert MAX_RELATIONS > 20


#: Rough token cost of one rendered entry, measured from real generations.
_TOKENS_PER_COMPONENT = 50
_TOKENS_PER_RELATION = 25

#: The generation cap in scripts/gpu/generate.py.
_MAX_OUTPUT_TOKENS = 3072


def test_bounded_schema_fits_inside_the_token_budget() -> None:
    """A bound the model cannot reach before the token cap is no bound at all.

    An earlier attempt used 60/60, reasoning that more headroom is safer. It is
    not: 60 relations alone is ~1,500 tokens on top of the components, so
    generation hit the cap and truncated before the grammar ever required a
    closing bracket -- indistinguishable from having no bound. The limits must
    be reachable *within* the budget for termination to be enforceable.
    """
    from argmap.prompts import MAX_COMPONENTS, MAX_RELATIONS

    worst_case = MAX_COMPONENTS * _TOKENS_PER_COMPONENT + MAX_RELATIONS * _TOKENS_PER_RELATION
    assert worst_case < _MAX_OUTPUT_TOKENS, (
        f"a maximal answer needs ~{worst_case} tokens but the cap is "
        f"{_MAX_OUTPUT_TOKENS}; the grammar could never close the object"
    )
