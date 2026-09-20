"""Carving a validation split out of the official training set.

The Argument Annotated Essays distribution ships a 322/80 train/test split and
nothing else. Every tuning decision in this project -- prompt wording, few-shot
choice, LoRA checkpoint selection, the cascade threshold -- has to be made
against held-out data that is *not* the test set, so a validation split is
carved from the 322 training essays.

Stratified by component count rather than sampled uniformly: essay length and
annotation density vary enough that a uniform 20% draw can land a validation
set noticeably easier or harder than the training remainder, which would bias
every checkpoint-selection decision made against it.

The resulting essay IDs are written to `data/splits/` and committed. The IDs
are not corpus content, so committing them redistributes nothing while keeping
the split exactly reproducible.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from argmap.schema import Document

#: Fixed so the split is identical on every machine and every re-run.
DEFAULT_SEED = 13

#: Fraction of the official training set held out for validation.
DEFAULT_VAL_FRACTION = 0.2

#: Number of strata used when balancing the draw by component count.
_N_STRATA = 4


def carve_validation(
    docs: Sequence[Document],
    *,
    fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    n_strata: int = _N_STRATA,
) -> set[str]:
    """Choose validation document IDs from the documents currently split `train`.

    Returns the chosen IDs; the caller applies them. Draws proportionally from
    each component-count stratum so the validation set matches the training
    remainder in annotation density.
    """
    train = sorted((d for d in docs if d.split == "train"), key=lambda d: d.doc_id)
    if not train:
        return set()

    ordered = sorted(train, key=lambda d: (len(d.components), d.doc_id))
    rng = random.Random(seed)
    chosen: set[str] = set()

    stratum_size = max(1, len(ordered) // n_strata)
    for index in range(0, len(ordered), stratum_size):
        stratum = ordered[index : index + stratum_size]
        take = round(len(stratum) * fraction)
        if take:
            chosen.update(d.doc_id for d in rng.sample(stratum, take))

    return chosen


def apply_validation_split(docs: Sequence[Document], val_ids: set[str]) -> list[Document]:
    """Return copies of `docs` with the chosen training documents marked `val`."""
    out: list[Document] = []
    for doc in docs:
        if doc.doc_id in val_ids and doc.split == "train":
            out.append(doc.model_copy(update={"split": "val"}))
        else:
            out.append(doc)
    return out


def write_split_ids(docs: Sequence[Document], path: Path) -> None:
    """Persist the split as plain ID lists -- no corpus text, safe to commit."""
    payload: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for doc in sorted(docs, key=lambda d: d.doc_id):
        payload[doc.split].append(doc.doc_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def read_split_ids(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as fh:
        data: dict[str, list[str]] = json.load(fh)
    return data
