"""Prompt construction and few-shot exemplar selection.

The model is asked for component **text**, not character offsets. Two reasons:

* Asking a language model for character indices is unreliable, and v1 did
  exactly that before overriding every answer with fuzzy matching anyway.
* The alignment study (`results/alignment/`) shows that locating component text
  costs ~1% under overlap matching, so the offsets are cheap to recover and
  expensive to ask for.

Exemplars are drawn from the **training** split only and are written to a
gitignored file rather than hardcoded here: committed source on a public
repository is publication, and the Argument Annotated Essays license forbids
redistributing the data. See DECISIONS.md D2.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from argmap.schema import Document

#: Bump when the prompt changes. It is part of the response cache key, so a
#: stale cache can never silently serve an answer to a different prompt.
PROMPT_VERSION = "v2"

#: Three, not two, for a reason that is measurable rather than aesthetic.
#:
#: Haiku 4.5 will not cache a prefix shorter than 4,096 tokens, and it reports
#: no error when it declines -- `cache_creation_input_tokens` simply stays 0.
#: Two exemplars put the prefix at ~3,600 tokens, just under the line, so every
#: call paid full price for a prefix identical across all 402 documents. A
#: third exemplar clears the threshold. See DECISIONS.md D13.
DEFAULT_EXEMPLAR_COUNT = 3
DEFAULT_EXEMPLAR_SEED = 17

#: Minimum prefix Haiku 4.5 will cache. Opus 4.6/4.5 share it; Sonnet 5 needs
#: only 1,024 and Opus 5 only 512.
HAIKU_MIN_CACHEABLE_TOKENS = 4096


# ---------------------------------------------------------------------------
# What the model is asked to return
# ---------------------------------------------------------------------------


class RawComponent(BaseModel):
    """One argument component as the model states it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Short unique identifier, e.g. c1, c2, c3.")
    type: Literal["MajorClaim", "Claim", "Premise"] = Field(
        description=(
            "MajorClaim: the author's overall position, usually stated in the "
            "introduction or conclusion. Claim: a controversial statement that "
            "supports or attacks the major claim. Premise: a reason given for a claim."
        )
    )
    text: str = Field(
        description=(
            "The component quoted from the source text, as close to verbatim as "
            "possible. Do not paraphrase, summarise, or add words."
        )
    )


class RawRelation(BaseModel):
    """A directed relation between two components the model identified."""

    model_config = ConfigDict(extra="forbid")

    src: str = Field(description="id of the component doing the supporting or attacking")
    tgt: str = Field(description="id of the component being supported or attacked")
    type: Literal["supports", "attacks"]


class RawGraph(BaseModel):
    """The full argument graph for one document."""

    model_config = ConfigDict(extra="forbid")

    components: list[RawComponent]
    relations: list[RawRelation]


SYSTEM_PROMPT = """\
You are an expert annotator of argumentation structure, working to the \
Argument Annotated Essays guidelines.

Given an argumentative text, identify its argument components and the \
relations between them.

Components:
- MajorClaim: the author's overall standpoint on the topic. Usually stated in \
the introduction and restated in the conclusion. A text typically has one or two.
- Claim: a controversial statement that takes a position on the major claim. \
Needs support to be accepted.
- Premise: a reason, evidence or example given to support or attack a claim.

Relations:
- supports: the source gives a reason to accept the target.
- attacks: the source gives a reason to reject the target.

Telling a Claim from a Premise:
- Each body paragraph usually advances exactly one Claim, stated in its topic \
sentence. The rest of the paragraph is Premises supporting it.
- If a statement needs support to be believed, it is a Claim. If it is the \
support -- an example, statistic, consequence, or explanation -- it is a Premise.

Rules:
- Quote component text **verbatim from the source**. Do not paraphrase, \
shorten, or add connecting words. Exclude leading discourse connectives such \
as "However", "Therefore", "First of all" and the comma that follows them.
- A component is one clause or sentence. Do not split a single argumentative \
statement into several components, and do not merge a whole paragraph into one.
- Attach each Premise to the **Claim of its own paragraph**, not to another \
Premise. Use a Premise-to-Premise relation only when one premise exists purely \
to justify another, which is uncommon.
- Use `attacks` when a component gives a reason to *reject* its target, such \
as a concession or counter-argument the author then rebuts. Most relations are \
`supports`; do not force attacks that are not there.
- In this annotation scheme Claims are **not** linked to MajorClaims by a \
relation, so do not emit those edges.
- Do not include non-argumentative material: topic restatements, rhetorical \
questions, or transitional sentences that assert nothing. Precision matters as \
much as coverage -- a component you are unsure about is better left out.
- Every relation's src and tgt must be ids you listed in components."""


def build_user_prompt(text: str) -> str:
    return (
        "Identify the argument components and relations in the following text.\n\n"
        "<text>\n"
        f"{text}\n"
        "</text>"
    )


# ---------------------------------------------------------------------------
# Few-shot exemplars
# ---------------------------------------------------------------------------


def document_to_graph(doc: Document) -> RawGraph:
    """Render a gold document as the model is expected to answer.

    Components are renumbered `c1..cN` in document order rather than reusing
    the corpus's own `T` identifiers. Showing gold ids in a worked example
    teaches the model to emit them, and predicted ids then collide with gold
    ids that mean entirely different components -- which makes every debugging
    session and every error analysis ambiguous for no benefit.
    """
    renumbered = {c.id: f"c{i + 1}" for i, c in enumerate(doc.components)}
    return RawGraph(
        components=[
            RawComponent(id=renumbered[c.id], type=c.type or "Premise", text=c.text)
            for c in doc.components
        ],
        relations=[
            RawRelation(src=renumbered[r.src], tgt=renumbered[r.tgt], type=r.type)
            for r in doc.relations
            if r.src in renumbered and r.tgt in renumbered
        ],
    )


def select_exemplars(
    docs: Sequence[Document],
    *,
    k: int = DEFAULT_EXEMPLAR_COUNT,
    seed: int = DEFAULT_EXEMPLAR_SEED,
) -> list[Document]:
    """Pick `k` training documents to show as worked examples.

    Training split only -- drawing an exemplar from validation would leak the
    set every tuning decision is made against, and from test would be
    straightforwardly invalid.

    Chosen near the median component count so the examples are representative
    rather than unusually dense or sparse.
    """
    train = [d for d in docs if d.split == "train" and d.components]
    if not train:
        return []

    ordered = sorted(train, key=lambda d: (len(d.components), d.doc_id))
    middle = ordered[len(ordered) // 4 : 3 * len(ordered) // 4] or ordered
    rng = random.Random(seed)
    picked = rng.sample(middle, min(k, len(middle)))
    return sorted(picked, key=lambda d: d.doc_id)


def write_exemplars(docs: Sequence[Document], path: Path) -> list[str]:
    """Persist exemplars to a gitignored file. Returns the document ids used."""
    payload = [
        {
            "doc_id": d.doc_id,
            "source": d.source,
            "text": d.text,
            "graph": document_to_graph(d).model_dump(),
        }
        for d in docs
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return [d.doc_id for d in docs]


def load_exemplars(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        data: list[dict[str, object]] = json.load(fh)
    return data


def exemplar_turns(exemplars: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Render exemplars as alternating user/assistant turns.

    Worked examples as real conversation turns rather than as a blob inside the
    system prompt: it matches the shape of the actual request, and it keeps the
    cacheable prefix contiguous.
    """
    turns: list[dict[str, object]] = []
    for item in exemplars:
        text = str(item["text"])
        graph = item["graph"]
        turns.append({"role": "user", "content": build_user_prompt(text)})
        turns.append(
            {
                "role": "assistant",
                "content": json.dumps(graph, ensure_ascii=False),
            }
        )
    return turns


def prompt_fingerprint(exemplars: Sequence[dict[str, object]]) -> str:
    """Identify the whole prompt configuration for cache keying.

    Hashes the **rendered** exemplar turns, not merely which documents were
    chosen. Two prompt configurations can use the same exemplar documents and
    still render them differently -- renumbering component ids, for instance --
    and keying on document ids alone would serve a cached answer produced by a
    different prompt. That is the exact failure the cache key exists to
    prevent, so the rendered bytes are what gets hashed.
    """
    payload = json.dumps(
        {
            "version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "schema": RawGraph.model_json_schema(),
            "exemplar_turns": exemplar_turns(exemplars),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{PROMPT_VERSION}-{digest}"
