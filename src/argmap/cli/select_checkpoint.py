"""Score every checkpoint on validation and pick one.

    uv run python -m argmap.cli.select_checkpoint

Reads the raw generations written by `scripts/gpu/gen_val.py`, parses them with
the *same* assembler the Claude route uses, scores them with the project's
metrics, and writes a ranked report to `results/training/`.

Selection is on **validation only**. Nothing here reads the test split, and
the selected checkpoint's identity is fixed before test is ever scored.

Ranking is by the mean of component and relation F1 under overlap matching
with types required. One scalar has to decide, and that one weighs the two
halves of the task equally; components alone would let a checkpoint that
ignores relations win. Every other metric is written out too, so a different
ranking can be argued from the same file without regenerating anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

from argmap.cli.run_baseline import load_split
from argmap.extractors.assemble import graph_to_prediction, parse_graph_json
from argmap.metrics.bootstrap import bootstrap_f1
from argmap.metrics.matching import STANDARD_CRITERIA, MatchCriterion
from argmap.metrics.scores import score_corpus, sum_counts
from argmap.schema import Document, Prediction

#: The criterion the ranking scalar is computed under.
SELECTION_CRITERION = MatchCriterion(span="overlap", typed=True)


def _predictions_for_run(
    docs: list[Document],
    results: list[dict[str, Any]],
) -> tuple[dict[str, Prediction], dict[str, int]]:
    """Parse one checkpoint's raw generations into predictions."""
    by_id = {d.doc_id: d for d in docs}
    predictions: dict[str, Prediction] = {}
    invalid_json = 0
    unalignable = 0
    truncated = 0

    for entry in results:
        doc_id = str(entry.get("doc_id", ""))
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        if entry.get("finish_reason") == "length":
            truncated += 1

        graph = parse_graph_json(str(entry.get("text", "")))
        if graph is None:
            invalid_json += 1
            predictions[doc_id] = Prediction(doc_id=doc_id)
            continue

        prediction, stats = graph_to_prediction(doc, graph)
        unalignable += stats.unalignable
        predictions[doc_id] = prediction

    return predictions, {
        "invalid_json": invalid_json,
        "unalignable_components": unalignable,
        "truncated": truncated,
    }


def score_run(docs: list[Document], predictions: dict[str, Prediction]) -> dict[str, object]:
    scores: dict[str, object] = {}
    for criterion in STANDARD_CRITERIA:
        per_doc = score_corpus(docs, predictions, criterion)
        comp = [s.components for s in per_doc]
        rel = [s.relations for s in per_doc]
        scores[criterion.label] = {
            "components": {**sum_counts(comp).as_dict(), "f1_ci": bootstrap_f1(comp).as_dict()},
            "relations": {**sum_counts(rel).as_dict(), "f1_ci": bootstrap_f1(rel).as_dict()},
        }
    return scores


def _selection_score(scores: dict[str, object]) -> float:
    entry = cast("dict[str, Any]", scores[SELECTION_CRITERION.label])
    return (float(entry["components"]["f1"]) + float(entry["relations"]["f1"])) / 2


def run(root: Path, generations: Path, split: str) -> dict[str, object]:
    payload = cast("dict[str, Any]", json.loads(generations.read_text(encoding="utf-8")))
    docs = load_split(root, "aae-v2", split)

    ranked: list[dict[str, object]] = []
    for entry in payload["runs"]:
        adapter = entry.get("adapter") or "base"
        predictions, health = _predictions_for_run(docs, entry["results"])
        scores = score_run(docs, predictions)
        ranked.append(
            {
                "adapter": adapter,
                "seed": _field(adapter, "seed"),
                "epoch": _field(adapter, "epoch"),
                "selection_score": round(_selection_score(scores), 6),
                "health": health,
                "generate_seconds": entry.get("generate_seconds"),
                "throughput_docs_per_second": entry.get("throughput_docs_per_second"),
                "scores": scores,
            }
        )

    ranked.sort(key=lambda r: float(cast("float", r["selection_score"])), reverse=True)
    best = ranked[0] if ranked else None

    # Mean and standard deviation across seeds at the selected epoch: with 258
    # training documents, how much of a gap between checkpoints is real matters
    # more than which one happened to win.
    by_epoch: dict[object, list[float]] = {}
    for entry in ranked:
        by_epoch.setdefault(entry["epoch"], []).append(
            float(cast("float", entry["selection_score"]))
        )
    epoch_stats = {
        str(epoch): {
            "seeds": len(values),
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        }
        for epoch, values in sorted(by_epoch.items(), key=lambda kv: str(kv[0]))
    }

    report: dict[str, object] = {
        "base_model": payload.get("model_path"),
        "split": split,
        "documents": len(docs),
        "selection_criterion": SELECTION_CRITERION.label,
        "selection_scalar": "mean of component and relation F1",
        "selected": best["adapter"] if best else None,
        "across_seeds_by_epoch": epoch_stats,
        "checkpoints": ranked,
    }

    out_dir = root / "results" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def _field(adapter: str, key: str) -> str | None:
    for part in adapter.split("/"):
        if part.startswith(key):
            return part[len(key) :]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--generations",
        type=Path,
        default=Path("data/generations/val_checkpoints.json"),
    )
    args = parser.parse_args()

    report = run(args.root, args.root / args.generations, args.split)

    print(f"{report['documents']} {args.split} documents, ranked by {report['selection_scalar']}")
    print(f"  under {report['selection_criterion']}\n")
    header = (
        f"  {'checkpoint':<22}{'score':>8}{'comp F1':>10}"
        f"{'rel F1':>9}{'bad json':>10}{'unalign':>9}"
    )
    print(header)
    for entry in cast("list[dict[str, Any]]", report["checkpoints"]):
        s = entry["scores"][SELECTION_CRITERION.label]
        print(
            f"  seed{entry['seed']}/epoch{entry['epoch']:<12}"
            f"{entry['selection_score']:>8.3f}"
            f"{s['components']['f1']:>10.3f}{s['relations']['f1']:>9.3f}"
            f"{entry['health']['invalid_json']:>10}{entry['health']['unalignable_components']:>9}"
        )

    print("\nacross seeds:")
    for epoch, stats in cast("dict[str, Any]", report["across_seeds_by_epoch"]).items():
        print(
            f"  epoch {epoch}: {stats['mean']:.3f} +/- {stats['std']:.3f} ({stats['seeds']} seeds)"
        )
    print(f"\nselected: {report['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
