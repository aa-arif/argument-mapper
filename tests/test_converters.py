"""Converter tests against synthetic stand-ins for both corpora.

The real corpora are licensed and gitignored, so these use miniature fixtures
shaped exactly like the real distributions -- nested zip and all. Tests that
need the genuine article are marked `corpus` and live in
`test_corpus_integrity.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argmap.data.aae import AAEFormatError, load_aae
from argmap.data.microtexts import MicrotextFormatError, parse_arggraph
from argmap.data.splits import apply_validation_split, carve_validation, write_split_ids
from argmap.schema import Document, read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Argument Annotated Essays
# ---------------------------------------------------------------------------


def test_aae_reads_both_essays_and_the_official_split(aae_zip: Path) -> None:
    docs = load_aae(aae_zip)
    assert [d.doc_id for d in docs] == ["essay001", "essay002"]
    assert {d.doc_id: d.split for d in docs} == {"essay001": "train", "essay002": "test"}


def test_aae_component_spans_match_the_source_text(aae_zip: Path) -> None:
    """The invariant that protects every downstream number."""
    for doc in load_aae(aae_zip):
        for component in doc.components:
            assert doc.text[component.start : component.end] == component.text


def test_aae_maps_types_relations_and_stance(aae_zip: Path) -> None:
    doc = load_aae(aae_zip)[0]
    assert [c.type for c in doc.components] == ["MajorClaim", "Claim", "Premise"]
    assert [(r.src, r.type, r.tgt) for r in doc.relations] == [("T3", "supports", "T2")]
    assert {c.id: c.stance for c in doc.components if c.stance} == {"T2": "For"}


def test_aae_skips_macos_resource_forks(aae_zip: Path) -> None:
    """The real distribution ships __MACOSX entries that must not become essays."""
    assert len(load_aae(aae_zip)) == 2


def test_aae_rejects_a_non_aae_zip(tmp_path: Path) -> None:
    import zipfile

    path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "not a corpus")
    with pytest.raises(AAEFormatError, match="not an Argument Annotated Essays"):
        load_aae(path)


def test_aae_detects_offset_drift(tmp_path: Path) -> None:
    """A shifted offset must raise, not silently produce a wrong span.

    This is what would happen if the `.txt` were read with newline translation.
    """
    import io
    import zipfile

    text = "Title here\n\nAlpha beta gamma.\n"
    ann = "T1\tClaim 13 29\tAlpha beta gamma.\n"  # one character too far right

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("brat-project-final/essay001.txt", text)
        zf.writestr("brat-project-final/essay001.ann", ann)

    path = tmp_path / "drift.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ArgumentAnnotatedEssays-2.0/train-test-split.csv", '"ID";"SET"\n')
        zf.writestr("ArgumentAnnotatedEssays-2.0/brat-project-final.zip", inner.getvalue())

    with pytest.raises(AAEFormatError, match="newline translation"):
        load_aae(path)


def test_aae_tolerates_a_trailing_tab_in_the_annotation(tmp_path: Path) -> None:
    """Real `.ann` files carry stray trailing tabs; offsets remain authoritative."""
    import io
    import zipfile

    text = "Alpha beta gamma."
    ann = "T1\tClaim 0 17\tAlpha beta gamma.\t\n"

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("brat-project-final/essay001.txt", text)
        zf.writestr("brat-project-final/essay001.ann", ann)

    path = tmp_path / "tab.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ArgumentAnnotatedEssays-2.0/train-test-split.csv", '"ID";"SET"\n')
        zf.writestr("ArgumentAnnotatedEssays-2.0/brat-project-final.zip", inner.getvalue())

    doc = load_aae(path)[0]
    assert doc.components[0].text == "Alpha beta gamma."
    assert doc.meta["whitespace_fixups"] == ["T1"]


# ---------------------------------------------------------------------------
# arg-microtexts
# ---------------------------------------------------------------------------


def test_microtext_text_is_edus_joined_by_a_single_space(microtext_xml: str) -> None:
    doc = parse_arggraph(microtext_xml, "micro_t001")
    assert doc.text == "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."


def test_microtext_spans_index_the_reconstructed_text(microtext_xml: str) -> None:
    doc = parse_arggraph(microtext_xml, "micro_t001")
    for component in doc.components:
        assert doc.text[component.start : component.end] == component.text
    assert [(c.start, c.end) for c in doc.components] == [(0, 17), (18, 37), (38, 53)]


def test_microtext_components_are_deliberately_untyped(microtext_xml: str) -> None:
    """pro/opp is a dialectical role, not MajorClaim/Claim/Premise."""
    doc = parse_arggraph(microtext_xml, "micro_t001")
    assert all(c.type is None for c in doc.components)
    assert doc.typed is False
    assert doc.meta["adu_roles"] == {"a1": "opp", "a2": "pro", "a3": "pro"}


def test_microtext_undercut_is_flattened_onto_the_edges_source(microtext_xml: str) -> None:
    """c2 undercuts edge c1 (a1 -reb-> a2), so it becomes a3 attacks a1."""
    doc = parse_arggraph(microtext_xml, "micro_t001")
    edges = {(r.src, r.type, r.tgt) for r in doc.relations}
    assert ("a1", "attacks", "a2") in edges
    assert ("a3", "attacks", "a1") in edges
    assert doc.meta["undercuts_flattened"] == 1
    assert doc.meta["unresolvable_edges"] == []


def test_microtext_rejects_an_unknown_edge_type() -> None:
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<arggraph id="x" topic_id="t" stance="pro">
  <edu id="e1"><![CDATA[Alpha.]]></edu>
  <edu id="e2"><![CDATA[Beta.]]></edu>
  <adu id="a1" type="pro"/>
  <adu id="a2" type="pro"/>
  <edge id="s1" src="e1" trg="a1" type="seg"/>
  <edge id="s2" src="e2" trg="a2" type="seg"/>
  <edge id="c1" src="a1" trg="a2" type="bogus"/>
</arggraph>
"""
    with pytest.raises(MicrotextFormatError, match="unknown edge type"):
        parse_arggraph(xml, "x")


def test_microtext_is_entirely_held_out(microtext_xml: str) -> None:
    assert parse_arggraph(microtext_xml, "micro_t001").split == "test"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _synthetic_train(n: int) -> list[Document]:
    from argmap.schema import Component

    docs: list[Document] = []
    for i in range(n):
        text = "Alpha beta gamma. " * (1 + i % 5)
        n_components = 1 + i % 5
        components = [
            Component(
                id=f"T{k}",
                type="Premise",
                start=18 * k,
                end=18 * k + 17,
                text=text[18 * k : 18 * k + 17],
            )
            for k in range(n_components)
        ]
        docs.append(
            Document(
                doc_id=f"essay{i:03d}",
                source="aae-v2",
                split="train" if i % 5 else "test",
                text=text,
                components=components,
            )
        )
    return docs


def test_validation_carve_is_reproducible_and_proportional() -> None:
    docs = _synthetic_train(100)
    first = carve_validation(docs)
    second = carve_validation(docs)
    assert first == second

    n_train = sum(1 for d in docs if d.split == "train")
    assert 0.15 * n_train <= len(first) <= 0.25 * n_train


def test_validation_carve_never_touches_test() -> None:
    docs = _synthetic_train(100)
    val_ids = carve_validation(docs)
    test_ids = {d.doc_id for d in docs if d.split == "test"}
    assert not (val_ids & test_ids)

    updated = apply_validation_split(docs, val_ids)
    assert {d.doc_id for d in updated if d.split == "test"} == test_ids
    assert {d.doc_id for d in updated if d.split == "val"} == val_ids


def test_split_ids_round_trip_without_corpus_text(tmp_path: Path) -> None:
    docs = apply_validation_split(_synthetic_train(40), carve_validation(_synthetic_train(40)))
    path = tmp_path / "splits.json"
    write_split_ids(docs, path)

    raw = path.read_text(encoding="utf-8")
    assert "Alpha beta gamma" not in raw, "split files must never carry corpus text"
    assert set(json.loads(raw)) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(gold_doc: Document, tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    write_jsonl([gold_doc], path)
    assert read_jsonl(path) == [gold_doc]
