"""Argument Annotated Essays v2 (brat standoff) -> unified schema.

The corpus ships as a zip containing a nested `brat-project-final.zip` with one
`.txt` and one `.ann` per essay, plus `train-test-split.csv` carrying the
official 322/80 split.

Offsets are the delicate part. brat offsets index into the raw `.txt` bytes as
decoded UTF-8, and Python's text-mode `open()` performs universal-newline
translation that silently collapses CRLF to LF and shifts every subsequent
offset by one per preceding line. Everything here reads bytes and decodes
explicitly, and every component is checked against the source text before a
`Document` is constructed.

License: the corpus may not be redistributed. Nothing in this module writes
corpus text anywhere except under `data/`, which is gitignored.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from argmap.schema import Component, ComponentType, Document, Relation, RelationType, Split, Stance

#: Path of the nested brat archive inside the distributed zip.
_INNER_ZIP = "ArgumentAnnotatedEssays-2.0/brat-project-final.zip"
_SPLIT_CSV = "ArgumentAnnotatedEssays-2.0/train-test-split.csv"

#: brat component labels, which happen to match our schema one-to-one.
_COMPONENT_LABELS: frozenset[str] = frozenset({"MajorClaim", "Claim", "Premise"})
_RELATION_LABELS: frozenset[str] = frozenset({"supports", "attacks"})

_T_LINE = re.compile(r"^(T\d+)\t([A-Za-z_]+) ([\d ;]+)\t(.*)$", re.DOTALL)
_R_LINE = re.compile(r"^(R\d+)\t([A-Za-z_]+) Arg1:(T\d+) Arg2:(T\d+)\s*$")
_A_LINE = re.compile(r"^(A\d+)\t([A-Za-z_]+) (T\d+) ([A-Za-z_]+)\s*$")


class AAEFormatError(RuntimeError):
    """Raised when the corpus does not match the documented brat format."""


@dataclass(frozen=True)
class _Span:
    """A `T` line: offsets plus the text brat recorded alongside them."""

    id: str
    type: ComponentType
    start: int
    end: int
    declared_text: str


@dataclass(frozen=True)
class _Annotation:
    spans: list[_Span]
    relations: list[Relation]
    stances: dict[str, Stance]
    #: Components whose span was discontinuous and had to be dropped.
    discontinuous: list[str]


def _parse_ann(ann: str, doc_id: str) -> _Annotation:
    """Parse one `.ann` file into components and relations.

    Stance attributes (`A` lines) are folded onto their component rather than
    becoming relations -- see DECISIONS.md D4.
    """
    spans: dict[str, tuple[ComponentType, int, int, str]] = {}
    stances: dict[str, Stance] = {}
    raw_relations: list[tuple[str, RelationType, str, str]] = []
    discontinuous: list[str] = []

    for raw_line in ann.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        if line.startswith("T"):
            m = _T_LINE.match(line)
            if m is None:
                msg = f"{doc_id}: unparseable T line: {line!r}"
                raise AAEFormatError(msg)
            tid, label, offsets, text = m.groups()
            if label not in _COMPONENT_LABELS:
                msg = f"{doc_id}/{tid}: unexpected component label {label!r}"
                raise AAEFormatError(msg)
            if ";" in offsets:
                # brat permits discontinuous spans. AAE v2 does not use them,
                # but drop rather than guess if that ever changes.
                discontinuous.append(tid)
                continue
            start_s, end_s = offsets.split()
            spans[tid] = (cast(ComponentType, label), int(start_s), int(end_s), text)

        elif line.startswith("R"):
            m = _R_LINE.match(line)
            if m is None:
                msg = f"{doc_id}: unparseable R line: {line!r}"
                raise AAEFormatError(msg)
            rid, label, arg1, arg2 = m.groups()
            if label not in _RELATION_LABELS:
                msg = f"{doc_id}/{rid}: unexpected relation label {label!r}"
                raise AAEFormatError(msg)
            raw_relations.append((rid, cast(RelationType, label), arg1, arg2))

        elif line.startswith("A"):
            m = _A_LINE.match(line)
            if m is None:
                msg = f"{doc_id}: unparseable A line: {line!r}"
                raise AAEFormatError(msg)
            _, attr, target, value = m.groups()
            if attr != "Stance":
                msg = f"{doc_id}: unexpected attribute {attr!r}"
                raise AAEFormatError(msg)
            if value not in ("For", "Against"):
                msg = f"{doc_id}: unexpected stance value {value!r}"
                raise AAEFormatError(msg)
            stances[target] = value

        else:
            msg = f"{doc_id}: unrecognised annotation line: {line!r}"
            raise AAEFormatError(msg)

    ordered = [
        _Span(id=tid, type=label, start=start, end=end, declared_text=text)
        for tid, (label, start, end, text) in sorted(spans.items(), key=lambda kv: kv[1][1])
    ]

    dropped = set(discontinuous)
    relations = [
        Relation(id=rid, src=src, tgt=tgt, type=label)
        for rid, label, src, tgt in raw_relations
        if src not in dropped and tgt not in dropped
    ]

    return _Annotation(
        spans=ordered,
        relations=relations,
        stances=stances,
        discontinuous=discontinuous,
    )


def _read_split_csv(data: bytes) -> dict[str, Split]:
    """Parse `train-test-split.csv` (semicolon-delimited, quoted fields)."""
    text = data.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
    rows = [row for row in reader if row]
    header = [cell.strip().strip('"').upper() for cell in rows[0]]
    if header[:2] != ["ID", "SET"]:
        msg = f"unexpected split header: {rows[0]!r}"
        raise AAEFormatError(msg)

    mapping: dict[str, Split] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        essay_id = row[0].strip().strip('"')
        set_name = row[1].strip().strip('"').upper()
        if set_name == "TRAIN":
            mapping[essay_id] = "train"
        elif set_name == "TEST":
            mapping[essay_id] = "test"
        else:
            msg = f"unexpected split value {set_name!r} for {essay_id}"
            raise AAEFormatError(msg)
    return mapping


def load_aae(zip_path: Path) -> list[Document]:
    """Convert the distributed AAE v2 zip into unified-schema documents.

    Splits are the official `train`/`test`; carving `val` out of `train` is a
    separate, deliberate step (see `argmap.data.splits`).
    """
    with zipfile.ZipFile(zip_path) as outer:
        try:
            split_map = _read_split_csv(outer.read(_SPLIT_CSV))
            inner_bytes = outer.read(_INNER_ZIP)
        except KeyError as exc:
            msg = f"{zip_path} is not an Argument Annotated Essays v2 distribution: {exc}"
            raise AAEFormatError(msg) from exc

        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            essays = _collect_essay_names(inner)
            documents: list[Document] = []
            for essay_id, txt_name, ann_name in essays:
                # Decode explicitly: text-mode reads would translate CRLF and
                # invalidate every offset in the file.
                text = inner.read(txt_name).decode("utf-8")
                ann = inner.read(ann_name).decode("utf-8")
                parsed = _parse_ann(ann, essay_id)
                components, whitespace_fixups = _build_components(parsed, text, essay_id)
                documents.append(
                    Document(
                        doc_id=essay_id,
                        source="aae-v2",
                        split=split_map.get(essay_id, "train"),
                        text=text,
                        components=components,
                        relations=parsed.relations,
                        meta={
                            "discontinuous_dropped": parsed.discontinuous,
                            "whitespace_fixups": whitespace_fixups,
                        },
                    )
                )

    documents.sort(key=lambda d: d.doc_id)
    return documents


def _build_components(
    parsed: _Annotation,
    text: str,
    doc_id: str,
) -> tuple[list[Component], list[str]]:
    """Build components, treating brat offsets -- not the `.ann` text -- as truth.

    Some `.ann` text fields carry a trailing tab that the annotation tool left
    behind, so the recorded text is not byte-identical to the span it describes.
    The offsets are authoritative; the recorded text is kept as a cross-check.

    That check is the reason this is not circular: if the `.txt` were read with
    newline translation, or with a stray BOM, the derived text would diverge
    from the recorded text by far more than whitespace and this raises.
    """
    components: list[Component] = []
    fixups: list[str] = []

    for span in parsed.spans:
        derived = text[span.start : span.end]
        if derived != span.declared_text:
            if derived.strip() != span.declared_text.strip():
                msg = (
                    f"{doc_id}/{span.id}: offsets [{span.start}, {span.end}) yield "
                    f"{derived!r} but the annotation records {span.declared_text!r}. "
                    "This usually means the .txt was read with newline translation."
                )
                raise AAEFormatError(msg)
            fixups.append(span.id)

        components.append(
            Component(
                id=span.id,
                type=span.type,
                start=span.start,
                end=span.end,
                text=derived,
                stance=parsed.stances.get(span.id),
            )
        )

    return components, fixups


def _collect_essay_names(inner: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """Pair up `.txt`/`.ann` members, skipping macOS resource forks."""
    txts: dict[str, str] = {}
    anns: dict[str, str] = {}
    for name in inner.namelist():
        if name.startswith("__MACOSX") or "/._" in name or name.endswith("/"):
            continue
        stem = Path(name).stem
        if name.endswith(".txt"):
            txts[stem] = name
        elif name.endswith(".ann"):
            anns[stem] = name

    missing = set(txts) ^ set(anns)
    if missing:
        msg = f"unpaired essay files: {sorted(missing)[:5]}"
        raise AAEFormatError(msg)

    return sorted((stem, txts[stem], anns[stem]) for stem in txts)
