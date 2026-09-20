"""Compare two systems with a paired bootstrap.

    uv run python -m argmap.cli.compare --a claude-sonnet-5 --b claude-haiku-4-5 \
        --corpus aae-v2 --split test

Answers "is A actually better than B, or is that within noise" by resampling
the *same* documents for both systems and reporting a confidence interval on
the difference.

Checking whether two marginal intervals happen to overlap is the common
shortcut and it is the wrong test: it is strictly more conservative, and it
throws away the pairing that makes the comparison powerful. Two systems can
have heavily overlapping marginal intervals while one beats the other on
almost every single document.

This is also the test milestone 5 needs, where the claim is that a cascade
matches Claude-only F1 within its interval -- a statement about a difference,
not about two separate numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from argmap.cli.run_baseline import load_split
from argmap.metrics.bootstrap import bootstrap_f1, paired_bootstrap_delta
from argmap.metrics.matching import STANDARD_CRITERIA
from argmap.metrics.scores import score_corpus
from argmap.schema import Document, Prediction


def load_predictions(root: Path, stem: str) -> dict[str, Prediction]:
    """Load the newest prediction file matching a run stem."""
    pred_dir = root / "data" / "predictions"
    matches = sorted(pred_dir.glob(f"{stem}*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        msg = f"no predictions found for {stem} in {pred_dir}"
        raise FileNotFoundError(msg)

    predictions: dict[str, Prediction] = {}
    for line in matches[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            pred = Prediction.model_validate_json(line)
            predictions[pred.doc_id] = pred
    return predictions


def compare(
    docs: Sequence[Document],
    a_preds: dict[str, Prediction],
    b_preds: dict[str, Prediction],
    *,
    typed_available: bool,
) -> dict[str, object]:
    report: dict[str, object] = {}

    for criterion in STANDARD_CRITERIA:
        if criterion.typed and not typed_available:
            report[criterion.label] = {"status": "not_applicable"}
            continue

        a_scores = score_corpus(docs, a_preds, criterion)
        b_scores = score_corpus(docs, b_preds, criterion)

        entry: dict[str, object] = {}
        for kind in ("components", "relations"):
            a_counts = [getattr(s, kind) for s in a_scores]
            b_counts = [getattr(s, kind) for s in b_scores]
            delta = paired_bootstrap_delta(a_counts, b_counts)
            entry[kind] = {
                "a": bootstrap_f1(a_counts).as_dict(),
                "b": bootstrap_f1(b_counts).as_dict(),
                "delta": delta.as_dict(),
                "significant": delta.excludes_zero,
            }
        report[criterion.label] = entry

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--a", required=True, help="model name of the first system")
    parser.add_argument("--b", required=True, help="model name of the second system")
    parser.add_argument("--corpus", default="aae-v2")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    docs = load_split(args.root, args.corpus, args.split)
    stem_a = f"{args.a}_{args.corpus}_{args.split}"
    stem_b = f"{args.b}_{args.corpus}_{args.split}"
    a_preds = load_predictions(args.root, stem_a)
    b_preds = load_predictions(args.root, stem_b)

    report = compare(docs, a_preds, b_preds, typed_available=any(d.typed for d in docs))
    payload: dict[str, object] = {
        "a": args.a,
        "b": args.b,
        "corpus": args.corpus,
        "split": args.split,
        "documents": len(docs),
        "comparisons": report,
    }

    out_dir = args.root / "results" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.a}_vs_{args.b}_{args.corpus}_{args.split}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"{args.a} vs {args.b} on {args.corpus}/{args.split} ({len(docs)} docs)\n")
    for criterion in STANDARD_CRITERIA:
        entry = report.get(criterion.label)
        if not isinstance(entry, dict) or entry.get("status") == "not_applicable":
            continue
        for kind in ("components", "relations"):
            block = cast("dict[str, Any]", entry[kind])
            a_f1 = float(block["a"]["point"])
            b_f1 = float(block["b"]["point"])
            d = block["delta"]
            verdict = "significant" if block["significant"] else "within noise"
            print(
                f"  {criterion.label:<22} {kind:<11} "
                f"{a_f1:.3f} vs {b_f1:.3f}  "
                f"delta {float(d['point']):+.3f} "
                f"[{float(d['ci_low']):+.3f},{float(d['ci_high']):+.3f}]  {verdict}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
