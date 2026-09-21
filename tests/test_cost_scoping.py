"""A cost for one corpus must never include calls from another.

The spend ledger is one flat append-only file covering every run the project
ever made, and for a while the cascade priced Claude by averaging all of a
model's rows. AAE essays average 1,974 characters and arg-microtexts 423, so
that mean blended two prices that differ by roughly 2x and reported a figure
that was correct for neither corpus — and that moved with how many of each had
happened to be run.

These tests pin the scoping rule rather than the numbers: a cost computed for
one set of documents must be unchanged by the existence of rows for any other
set. They use synthetic ledgers, so they need no corpus and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argmap.cli.cascade import claude_usd_per_doc
from argmap.cli.cost_report import split_ids, usd_per_doc

#: Two corpora with prices far enough apart that a blend is unmistakable.
_CHEAP: list[dict[str, object]] = [{"doc_id": f"micro_{i}", "dollars": 0.001} for i in range(10)]
_DEAR: list[dict[str, object]] = [{"doc_id": f"essay{i}", "dollars": 0.020} for i in range(10)]


def _write_ledger(
    root: Path, rows: list[dict[str, object]], model: str = "claude-sonnet-5"
) -> None:
    path = root / "results" / "cost"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "ledger.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({"model": model, "cached": False, **row}))
            fh.write("\n")


def test_cost_for_one_corpus_ignores_the_other(tmp_path: Path) -> None:
    """The headline regression: adding another corpus must not move a price."""
    dear_ids = [str(row["doc_id"]) for row in _DEAR]

    _write_ledger(tmp_path, list(_DEAR))
    alone = claude_usd_per_doc(tmp_path, "claude-sonnet-5", dear_ids)

    _write_ledger(tmp_path, [*_DEAR, *_CHEAP])
    together = claude_usd_per_doc(tmp_path, "claude-sonnet-5", dear_ids)

    assert alone == together == pytest.approx(0.020), (
        "a cost scoped to one corpus changed when rows for another corpus were "
        f"added to the ledger: {alone} then {together}"
    )

    # And the blend that used to be reported is visibly neither price.
    every_id = dear_ids + [str(row["doc_id"]) for row in _CHEAP]
    blended = claude_usd_per_doc(tmp_path, "claude-sonnet-5", every_id)
    assert blended == pytest.approx(0.0105)
    assert blended != alone


def test_cost_ignores_other_models_and_cache_hits(tmp_path: Path) -> None:
    ids = [str(row["doc_id"]) for row in _DEAR]
    rows: list[dict[str, object]] = [
        *_DEAR,
        # A cache hit costs nothing and must not drag the mean down; the
        # ledger records it so the hit rate is visible, not so it is averaged.
        {"doc_id": "essay0", "dollars": 0.0, "cached": True},
    ]
    _write_ledger(tmp_path, rows)
    assert claude_usd_per_doc(tmp_path, "claude-sonnet-5", ids) == pytest.approx(0.020)
    # A different model's rows are not this model's cost.
    assert claude_usd_per_doc(tmp_path, "claude-haiku-4-5", ids) == 0.0


def test_unscoped_rows_are_excluded(tmp_path: Path) -> None:
    """Rows the serving layer wrote belong to no corpus and no split."""
    ids = [str(row["doc_id"]) for row in _DEAR]
    _write_ledger(tmp_path, [*_DEAR, {"doc_id": "request", "dollars": 5.0}])
    assert claude_usd_per_doc(tmp_path, "claude-sonnet-5", ids) == pytest.approx(0.020)


def test_cost_report_scopes_match_the_split_files(tmp_path: Path) -> None:
    """`usd_per_doc` and the cascade's pricing must agree on the same ids."""
    ids = {str(row["doc_id"]) for row in _DEAR}
    _write_ledger(tmp_path, [*_DEAR, *_CHEAP])
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "cost" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    report = usd_per_doc(rows, "claude-sonnet-5", ids)
    assert report["calls"] == len(_DEAR)
    assert report["usd_per_doc"] == claude_usd_per_doc(tmp_path, "claude-sonnet-5", sorted(ids))


def test_real_split_files_do_not_share_document_ids() -> None:
    """The scoping is only sound if corpora cannot collide on an id.

    Everything above filters by document id, which silently breaks if two
    corpora ever use the same one. They do not — `essay001` against
    `micro_b001` — and this fails the moment that stops being true.
    """
    root = Path(__file__).resolve().parent.parent
    ids = split_ids(root)
    if not ids:
        return  # splits not built; the path guard covers this case

    by_corpus: dict[str, set[str]] = {}
    for (corpus, _split), id_set in ids.items():
        by_corpus.setdefault(corpus, set()).update(id_set)

    corpora = sorted(by_corpus)
    for i, first in enumerate(corpora):
        for second in corpora[i + 1 :]:
            shared = by_corpus[first] & by_corpus[second]
            assert not shared, (
                f"{first} and {second} share document ids {sorted(shared)[:5]}, so a "
                "cost filtered by document id would silently blend the two corpora"
            )
