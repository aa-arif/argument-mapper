"""Build both corpora into the unified schema.

    uv run argmap-build-datasets

Reads from `data/raw/`, writes unified JSONL to `data/processed/` (gitignored),
split ID lists to `splits/` and a statistics summary to `results/datasets/`.

The split files and the summary are committed. Neither contains corpus text --
only essay identifiers and counts -- so committing them keeps the pipeline
reproducible without redistributing anything the licenses forbid.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from argmap.data.aae import load_aae
from argmap.data.download import ensure_aae, ensure_microtexts
from argmap.data.microtexts import load_microtexts
from argmap.data.splits import (
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    apply_validation_split,
    carve_validation,
    write_split_ids,
)
from argmap.schema import Document, write_jsonl


def _meta_int(docs: list[Document], key: str) -> int:
    """Sum an integer counter stored in `Document.meta`, which is loosely typed."""
    total = 0
    for doc in docs:
        value = doc.meta.get(key, 0)
        if isinstance(value, int):
            total += value
    return total


def _summarise(docs: list[Document]) -> dict[str, object]:
    """Counts only -- never text. Safe to commit."""
    by_split = Counter(d.split for d in docs)
    return {
        "documents": len(docs),
        "by_split": dict(sorted(by_split.items())),
        "components": len([c for d in docs for c in d.components]),
        "components_by_type": dict(
            sorted(Counter(c.type or "untyped" for d in docs for c in d.components).items())
        ),
        "relations": len([r for d in docs for r in d.relations]),
        "relations_by_type": dict(
            sorted(Counter(r.type for d in docs for r in d.relations).items())
        ),
        "stance": dict(
            sorted(Counter(c.stance for d in docs for c in d.components if c.stance).items())
        ),
        "typed_components": any(d.typed for d in docs),
        "mean_chars": round(sum(len(d.text) for d in docs) / len(docs), 1) if docs else 0.0,
        "max_chars": max((len(d.text) for d in docs), default=0),
    }


def build(
    root: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    splits_dir = root / "splits"
    results_dir = root / "results" / "datasets"

    aae_file = ensure_aae(raw)
    microtexts_file = ensure_microtexts(raw)

    aae = load_aae(aae_file.path)
    val_ids = carve_validation(aae, fraction=val_fraction, seed=seed)
    aae = apply_validation_split(aae, val_ids)

    microtexts = load_microtexts(microtexts_file.path)

    write_jsonl(aae, processed / "aae.jsonl")
    write_jsonl(microtexts, processed / "microtexts.jsonl")
    write_split_ids(aae, splits_dir / "aae_splits.json")
    write_split_ids(microtexts, splits_dir / "microtexts_splits.json")

    summary: dict[str, object] = {
        "aae-v2": {
            **_summarise(aae),
            "sha256": aae_file.sha256,
            "val_fraction": val_fraction,
            "val_seed": seed,
        },
        "microtexts-en": {
            **_summarise(microtexts),
            "sha256": microtexts_file.sha256,
            "undercuts_flattened": _meta_int(microtexts, "undercuts_flattened"),
            "linked_premises_flattened": _meta_int(microtexts, "linked_premises_flattened"),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    summary = build(args.root, val_fraction=args.val_fraction, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
