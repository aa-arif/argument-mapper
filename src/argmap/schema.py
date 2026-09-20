"""The single argument-graph schema both corpora are converted into.

Everything downstream -- metrics, prompts, fine-tuning targets, the serving
layer -- speaks this schema and nothing else. The corpus-specific shapes (brat
standoff for Argument Annotated Essays, arggraph XML for arg-microtexts) exist
only inside `argmap.data`.

Design notes that are easy to get wrong:

* Character offsets are half-open, `text[start:end]`, and are indices into
  `Document.text` exactly as stored. They are *not* normalized in any way.
  Normalizing newlines shifts every offset in a document silently, so
  `Document` validates the invariant `text[start:end] == component.text`.
* `Component.type` is optional. arg-microtexts does not label components with
  the MajorClaim/Claim/Premise typology at all, so typed metrics are simply
  not computable there and must report N/A rather than a fabricated value.
* `Component.stance` is an attribute rather than a relation, matching how
  Argument Annotated Essays encodes a claim's orientation toward the major
  claim. See DECISIONS.md D4.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComponentType = Literal["MajorClaim", "Claim", "Premise"]
RelationType = Literal["supports", "attacks"]
Stance = Literal["For", "Against"]
Split = Literal["train", "val", "test"]
Source = Literal["aae-v2", "microtexts-en"]

#: Component types in the order they are reported in tables.
COMPONENT_TYPES: tuple[ComponentType, ...] = ("MajorClaim", "Claim", "Premise")

#: Relation types in the order they are reported in tables.
RELATION_TYPES: tuple[RelationType, ...] = ("supports", "attacks")


class Component(BaseModel):
    """An argument component: a span of the document with an optional type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: ComponentType | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    stance: Stance | None = None

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            msg = f"component {self.id}: empty or inverted span [{self.start}, {self.end})"
            raise ValueError(msg)
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class Relation(BaseModel):
    """A directed, typed edge between two components of the same document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    src: str
    tgt: str
    type: RelationType

    @model_validator(mode="after")
    def _check_endpoints(self) -> Self:
        if self.src == self.tgt:
            msg = f"relation {self.id}: self-loop on {self.src}"
            raise ValueError(msg)
        return self


class Document(BaseModel):
    """One annotated document, in the unified schema."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: Source
    split: Split
    text: str
    components: list[Component] = Field(default_factory=list[Component])
    relations: list[Relation] = Field(default_factory=list[Relation])
    meta: dict[str, object] = Field(default_factory=dict[str, object])

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        seen: set[str] = set()
        n = len(self.text)
        for c in self.components:
            if c.id in seen:
                msg = f"{self.doc_id}: duplicate component id {c.id}"
                raise ValueError(msg)
            seen.add(c.id)
            if c.end > n:
                msg = f"{self.doc_id}/{c.id}: span end {c.end} exceeds text length {n}"
                raise ValueError(msg)
            actual = self.text[c.start : c.end]
            if actual != c.text:
                msg = (
                    f"{self.doc_id}/{c.id}: span [{c.start}, {c.end}) yields {actual!r} "
                    f"but component text is {c.text!r}"
                )
                raise ValueError(msg)

        rel_ids: set[str] = set()
        for r in self.relations:
            if r.id in rel_ids:
                msg = f"{self.doc_id}: duplicate relation id {r.id}"
                raise ValueError(msg)
            rel_ids.add(r.id)
            for endpoint in (r.src, r.tgt):
                if endpoint not in seen:
                    msg = f"{self.doc_id}/{r.id}: endpoint {endpoint} is not a component"
                    raise ValueError(msg)
        return self

    def component_by_id(self, component_id: str) -> Component | None:
        return next((c for c in self.components if c.id == component_id), None)

    @property
    def typed(self) -> bool:
        """Whether this document carries component type labels at all.

        False for arg-microtexts, which makes typed metrics N/A rather than zero.
        """
        return any(c.type is not None for c in self.components)


class Prediction(BaseModel):
    """A model's output for one document, in the same shape as the gold graph.

    Kept separate from `Document` because a prediction carries no gold text and
    must never be validated against the span/text invariant -- a model's spans
    come from fuzzy alignment and are allowed to be wrong, which is the whole
    thing being measured.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    components: list[Component] = Field(default_factory=list[Component])
    relations: list[Relation] = Field(default_factory=list[Relation])
    meta: dict[str, object] = Field(default_factory=dict[str, object])


# ---------------------------------------------------------------------------
# JSONL corpus I/O
# ---------------------------------------------------------------------------


def write_jsonl(docs: list[Document], path: Path) -> None:
    """Write documents as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json(exclude_defaults=False))
            fh.write("\n")


def read_jsonl(path: Path) -> list[Document]:
    """Read documents written by :func:`write_jsonl`."""
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Document.model_validate(json.loads(line))
