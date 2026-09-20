"""arg-microtexts (arggraph XML) -> unified schema. Out-of-domain test set.

Two properties of this corpus force decisions that are recorded in DECISIONS.md
D3 and re-stated here because they change what the metrics mean:

1. **There are no character offsets.** Text is reconstructed by joining the
   `<edu>` contents in document order with a single space, and ADU spans are
   derived from that reconstruction. The join rule is part of the data
   contract -- change it and every span moves.

2. **There are no component type labels.** ADUs carry `pro`/`opp`, which is a
   dialectical role (proponent/opponent), not the MajorClaim/Claim/Premise
   typology. Components are therefore emitted with `type=None`, which makes
   typed component F1 *not computable* on this corpus. Reporting it as a number
   would invent labels the annotators never assigned.

A third wrinkle: `und` (undercut) and `add` (linked premise) edges target
*other edges*, not nodes. The unified schema has no edge-to-edge relation, so
they are flattened -- lossily, and counted -- as described in `_resolve_edge`.

License: CC BY-NC-SA 4.0 (Peldszus & Stede 2015). Not redistributed here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from argmap.schema import Component, Document, Relation, RelationType

#: Separator used to join EDUs into the reconstructed document text.
EDU_JOINER = " "

#: arggraph edge types that map directly onto a relation between two ADUs.
_DIRECT_EDGE_TYPES: dict[str, RelationType] = {
    "sup": "supports",
    "exa": "supports",  # 'example' is a support subtype
    "reb": "attacks",  # rebuttal: attacks the node itself
}

#: Edge types whose target is another edge rather than an ADU.
_EDGE_TARGETED_TYPES = frozenset({"und", "add"})


class MicrotextFormatError(RuntimeError):
    """Raised when an arggraph file does not match the documented format."""


@dataclass
class _FlattenStats:
    """How much was lost turning an edge-targeting graph into a node graph."""

    undercuts_flattened: int = 0
    linked_premises_flattened: int = 0
    unresolvable: list[str] = field(default_factory=list[str])

    def as_meta(self) -> dict[str, object]:
        return {
            "undercuts_flattened": self.undercuts_flattened,
            "linked_premises_flattened": self.linked_premises_flattened,
            "unresolvable_edges": self.unresolvable,
        }


@dataclass(frozen=True)
class _Edge:
    id: str
    src: str
    trg: str
    type: str


def _parse_graph(xml_text: str, doc_id: str) -> tuple[ET.Element, list[_Edge]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        msg = f"{doc_id}: malformed arggraph XML: {exc}"
        raise MicrotextFormatError(msg) from exc

    edges = [
        _Edge(
            id=el.attrib["id"],
            src=el.attrib["src"],
            trg=el.attrib["trg"],
            type=el.attrib["type"],
        )
        for el in root.findall("edge")
    ]
    return root, edges


def _build_text_and_spans(root: ET.Element, doc_id: str) -> tuple[str, dict[str, tuple[int, int]]]:
    """Reconstruct document text from EDUs and record each EDU's span."""
    text_parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0

    for el in root.findall("edu"):
        edu_id = el.attrib["id"]
        content = (el.text or "").strip()
        if not content:
            msg = f"{doc_id}/{edu_id}: empty EDU"
            raise MicrotextFormatError(msg)
        if text_parts:
            cursor += len(EDU_JOINER)
        spans[edu_id] = (cursor, cursor + len(content))
        cursor += len(content)
        text_parts.append(content)

    return EDU_JOINER.join(text_parts), spans


def _adu_spans(
    edges: list[_Edge],
    edu_spans: dict[str, tuple[int, int]],
    doc_id: str,
) -> dict[str, tuple[int, int]]:
    """Map each ADU to the union of the EDU spans segmented into it."""
    members: dict[str, list[tuple[int, int]]] = {}
    for edge in edges:
        if edge.type != "seg":
            continue
        span = edu_spans.get(edge.src)
        if span is None:
            msg = f"{doc_id}/{edge.id}: seg edge from unknown EDU {edge.src}"
            raise MicrotextFormatError(msg)
        members.setdefault(edge.trg, []).append(span)

    # EDUs are ordered, so the union of an ADU's EDUs is contiguous.
    return {
        adu_id: (min(s for s, _ in spans), max(e for _, e in spans))
        for adu_id, spans in members.items()
    }


def _resolve_edge(
    edge: _Edge,
    by_id: dict[str, _Edge],
    adu_ids: frozenset[str],
    stats: _FlattenStats,
    doc_id: str,
    _seen: frozenset[str] = frozenset(),
) -> tuple[RelationType, str] | None:
    """Reduce an arggraph edge to a `(type, target_adu)` pair.

    * `sup`/`exa` -> supports, `reb` -> attacks, targeting the ADU directly.
    * `und(a -> e)` attacks the *inference* drawn by edge `e`. The schema has no
      way to attack an edge, so it is flattened to an attack on `e`'s source --
      the conventional reduction. The undercut's distinctness is lost.
    * `add(a -> e)` marks `a` as a premise linked with `e`'s source, jointly
      making the same point. It inherits `e`'s resolved type and target, so a
      linked pair becomes two independent edges and their jointness is lost.
    """
    if edge.id in _seen:
        stats.unresolvable.append(edge.id)
        return None

    direct = _DIRECT_EDGE_TYPES.get(edge.type)
    if direct is not None:
        if edge.trg in adu_ids:
            return direct, edge.trg
        # A support or rebuttal pointing at an edge is not documented; treat it
        # like an undercut rather than guessing.
        referenced = by_id.get(edge.trg)
        if referenced is None:
            stats.unresolvable.append(edge.id)
            return None
        return direct, referenced.src

    if edge.type not in _EDGE_TARGETED_TYPES:
        msg = f"{doc_id}/{edge.id}: unknown edge type {edge.type!r}"
        raise MicrotextFormatError(msg)

    referenced = by_id.get(edge.trg)
    if referenced is None:
        stats.unresolvable.append(edge.id)
        return None

    if edge.type == "und":
        return "attacks", referenced.src

    # 'add': inherit whatever the edge it attaches to resolves to.
    inherited = _resolve_edge(referenced, by_id, adu_ids, stats, doc_id, _seen | {edge.id})
    if inherited is None:
        stats.unresolvable.append(edge.id)
        return None
    return inherited


def parse_arggraph(xml_text: str, doc_id: str) -> Document:
    """Convert one arggraph XML document into the unified schema."""
    root, edges = _parse_graph(xml_text, doc_id)
    text, edu_spans = _build_text_and_spans(root, doc_id)
    spans = _adu_spans(edges, edu_spans, doc_id)

    roles = {el.attrib["id"]: el.attrib.get("type", "") for el in root.findall("adu")}
    missing = set(roles) - set(spans)
    if missing:
        msg = f"{doc_id}: ADUs with no segmented EDU: {sorted(missing)}"
        raise MicrotextFormatError(msg)

    components = [
        Component(
            id=adu_id,
            type=None,  # deliberately unlabelled; see module docstring
            start=spans[adu_id][0],
            end=spans[adu_id][1],
            text=text[spans[adu_id][0] : spans[adu_id][1]],
        )
        for adu_id in sorted(spans, key=lambda a: spans[a][0])
    ]

    by_id = {e.id: e for e in edges}
    adu_ids = frozenset(roles)
    stats = _FlattenStats()

    relations: list[Relation] = []
    for edge in edges:
        if edge.type == "seg":
            continue
        resolved = _resolve_edge(edge, by_id, adu_ids, stats, doc_id)
        if resolved is None:
            continue
        # Counted here, at the top level, so an 'add' recursing through an 'und'
        # does not increment the undercut tally a second time.
        if edge.type == "und":
            stats.undercuts_flattened += 1
        elif edge.type == "add":
            stats.linked_premises_flattened += 1
        rel_type, target = resolved
        if target == edge.src:
            # Flattening can produce a self-loop when an undercut attacks an
            # inference its own source drew. Drop rather than emit an invalid edge.
            stats.unresolvable.append(edge.id)
            continue
        relations.append(Relation(id=edge.id, src=edge.src, tgt=target, type=rel_type))

    return Document(
        doc_id=doc_id,
        source="microtexts-en",
        split="test",  # the whole corpus is held out as out-of-domain evaluation
        text=text,
        components=components,
        relations=relations,
        meta={
            "topic_id": root.attrib.get("topic_id", ""),
            "stance": root.attrib.get("stance", ""),
            "adu_roles": roles,
            **stats.as_meta(),
        },
    )


def load_microtexts(source: Path) -> list[Document]:
    """Load the English arg-microtexts corpus from a repo zip or a directory."""
    if source.is_dir():
        files = sorted(source.glob("*.xml"))
        if not files:
            msg = f"no arggraph XML files under {source}"
            raise MicrotextFormatError(msg)
        docs = [parse_arggraph(p.read_text(encoding="utf-8"), p.stem) for p in files]
    else:
        with zipfile.ZipFile(source) as zf:
            names = sorted(
                n
                for n in zf.namelist()
                if "/corpus/en/" in n and n.endswith(".xml") and not n.startswith("__MACOSX")
            )
            if not names:
                msg = f"{source} contains no corpus/en/*.xml files"
                raise MicrotextFormatError(msg)
            docs = [parse_arggraph(zf.read(n).decode("utf-8"), Path(n).stem) for n in names]

    docs.sort(key=lambda d: d.doc_id)
    return docs
