"""Turn a model's raw graph into a `Prediction` with character spans.

Shared by every extraction route. That sharing is not a convenience: if the
Claude route and the local route assembled predictions differently -- a
different alignment threshold, a different rule for unlocatable text, a
different way of dropping dangling relations -- then any measured quality gap
between them would partly be a gap between two assembly functions, and the
comparison this repository exists to make would be invalid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from argmap.align import align
from argmap.schema import Component, ComponentType, Document, Prediction, Relation, RelationType

# Lookup tables rather than sets: a membership test does not narrow `str` to
# the Literal the schema requires, so a dict lookup keeps the types honest
# without a cast.
_COMPONENT_TYPES: dict[str, ComponentType] = {
    "MajorClaim": "MajorClaim",
    "Claim": "Claim",
    "Premise": "Premise",
}
_RELATION_TYPES: dict[str, RelationType] = {
    "supports": "supports",
    "attacks": "attacks",
}


@dataclass(frozen=True)
class AssemblyStats:
    """What had to be discarded turning raw output into a prediction."""

    #: Components whose text could not be located in the document at all.
    unalignable: int = 0
    #: Components with empty text, or a type outside the schema.
    malformed_components: int = 0
    #: Relations pointing at a component id that was never emitted or survived.
    dangling_relations: int = 0

    def as_meta(self) -> dict[str, object]:
        return {
            "unalignable_components": self.unalignable,
            "malformed_components": self.malformed_components,
            "dangling_relations": self.dangling_relations,
        }


def parse_graph_json(raw: str) -> dict[str, Any] | None:
    """Parse a model's raw text as a graph object, or return None.

    No repair is attempted. Under JSON-schema constrained decoding the output
    is valid by construction, and the *rate* at which unconstrained output
    fails to parse is itself a measurement (milestone 4) -- silently repairing
    it would erase the number being collected.
    """
    try:
        value: object = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _object_list(value: object) -> tuple[list[dict[str, object]], int]:
    """Coerce a JSON array of objects, returning it and how many were dropped.

    Model output arrives as `Any` from `json.loads`, and entries that are not
    objects have to be counted rather than silently skipped -- the count is
    part of the malformed-output rate reported in milestone 4.
    """
    if not isinstance(value, list):
        return [], 0
    kept: list[dict[str, object]] = []
    skipped = 0
    for item in cast("list[object]", value):
        if isinstance(item, dict):
            kept.append(cast("dict[str, object]", item))
        else:
            skipped += 1
    return kept, skipped


def graph_to_prediction(
    doc: Document,
    graph: dict[str, Any],
) -> tuple[Prediction, AssemblyStats]:
    """Align each component's text to a span and assemble the prediction."""
    components: list[Component] = []
    resolved: set[str] = set()
    unalignable = 0
    malformed = 0

    raw_components, skipped = _object_list(graph.get("components"))
    malformed += skipped
    for index, raw in enumerate(raw_components):
        text = str(raw.get("text", "")).strip()
        if not text:
            malformed += 1
            continue

        raw_type = raw.get("type")
        component_type = _COMPONENT_TYPES.get(raw_type) if isinstance(raw_type, str) else None
        if raw_type is not None and component_type is None:
            malformed += 1

        located = align(text, doc.text)
        if located is None:
            # The model produced text that is not in the document. Dropping it
            # costs recall, which it earned; inventing a span would manufacture
            # a false positive out of nothing.
            unalignable += 1
            continue

        component_id = str(raw.get("id") or f"c{index + 1}")
        if component_id in resolved:
            malformed += 1
            continue

        components.append(
            Component(
                id=component_id,
                type=component_type,
                start=located.start,
                end=located.end,
                text=doc.text[located.start : located.end],
            )
        )
        resolved.add(component_id)

    relations: list[Relation] = []
    dangling = 0
    raw_relations, skipped_relations = _object_list(graph.get("relations"))
    dangling += skipped_relations
    for index, raw in enumerate(raw_relations):
        src, tgt = str(raw.get("src", "")), str(raw.get("tgt", ""))
        raw_type = raw.get("type")
        if src not in resolved or tgt not in resolved or src == tgt:
            dangling += 1
            continue
        relation_type = (
            _RELATION_TYPES.get(raw_type, "supports") if isinstance(raw_type, str) else "supports"
        )
        relations.append(Relation(id=f"r{index + 1}", src=src, tgt=tgt, type=relation_type))

    stats = AssemblyStats(
        unalignable=unalignable,
        malformed_components=malformed,
        dangling_relations=dangling,
    )
    prediction = Prediction(
        doc_id=doc.doc_id,
        components=components,
        relations=relations,
        meta=dict(stats.as_meta()),
    )
    return prediction, stats
