"""Measure how much error the fuzzy matcher introduces.

    uv run python -m argmap.cli.eval_alignment

A model that emits component *text* has not told us where that text sits in the
document, so the text has to be aligned back to a span before any span-based
metric can be computed. Whatever error that alignment introduces is a ceiling
on the F1 such a system can reach -- it is charged to the extractor even though
the extractor may have been right.

This measures the ceiling directly by aligning **gold** component text against
**gold** documents, where the correct answer is known exactly. Conditions model
what a generative model actually emits: rarely a verbatim copy, usually a
lightly cleaned or trimmed version.

Results are written to `results/alignment/alignment_error.json`. Counts and
error statistics only -- no corpus text.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from argmap.align import align
from argmap.metrics.matching import overlap_ratio, token_spans, tokens_in_span
from argmap.schema import Document, iter_jsonl

Perturbation = Callable[[str], str]

_WS = re.compile(r"\s+")


def _verbatim(text: str) -> str:
    return text


def _whitespace_collapsed(text: str) -> str:
    """Models normalise internal whitespace and newlines almost universally."""
    return _WS.sub(" ", text).strip()


def _lowercased(text: str) -> str:
    return text.lower()


def _trimmed(text: str) -> str:
    """Drop the leading and trailing token.

    Models routinely shed a leading discourse connective ("However,", "And")
    or a trailing clause boundary when restating a component.
    """
    words = text.split()
    return " ".join(words[1:-1]) if len(words) > 2 else text


def _truncated(text: str) -> str:
    """Keep the first 80% of tokens -- a proxy for a paraphrase that stops early."""
    words = text.split()
    keep = max(1, int(len(words) * 0.8))
    return " ".join(words[:keep])


CONDITIONS: dict[str, Perturbation] = {
    "verbatim": _verbatim,
    "whitespace_collapsed": _whitespace_collapsed,
    "lowercased": _lowercased,
    "trimmed": _trimmed,
    "truncated_80pct": _truncated,
}


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return float(ordered[index])


def measure(
    docs: Sequence[Document],
    condition: str,
    *,
    refine: bool,
) -> dict[str, object]:
    """Align every gold component under one condition and summarise the error."""
    perturb = CONDITIONS[condition]

    start_errors: list[float] = []
    end_errors: list[float] = []
    jaccards: list[float] = []
    exact_hits = 0
    failures = 0
    total = 0
    started = time.perf_counter()

    for doc in docs:
        spans = token_spans(doc.text)
        for component in doc.components:
            total += 1
            needle = perturb(component.text)
            result = align(needle, doc.text, refine=refine)
            if result is None:
                failures += 1
                jaccards.append(0.0)
                continue

            if result.start == component.start and result.end == component.end:
                exact_hits += 1
            start_errors.append(abs(result.start - component.start))
            end_errors.append(abs(result.end - component.end))
            jaccards.append(
                overlap_ratio(
                    tokens_in_span(spans, component.start, component.end),
                    tokens_in_span(spans, result.start, result.end),
                )
            )

    elapsed = time.perf_counter() - started
    located = total - failures

    return {
        "condition": condition,
        "refined": refine,
        "components": total,
        "exact_span_rate": exact_hits / total if total else 0.0,
        "located_rate": located / total if total else 0.0,
        "failure_rate": failures / total if total else 0.0,
        "overlap_ge_50_rate": sum(1 for j in jaccards if j >= 0.5) / total if total else 0.0,
        "mean_jaccard": statistics.fmean(jaccards) if jaccards else 0.0,
        "start_error_median": statistics.median(start_errors) if start_errors else 0.0,
        "start_error_p90": _percentile(start_errors, 0.90),
        "end_error_median": statistics.median(end_errors) if end_errors else 0.0,
        "end_error_p90": _percentile(end_errors, 0.90),
        "seconds": round(elapsed, 2),
    }


def run(
    root: Path,
    *,
    splits: Sequence[str],
    limit: int | None,
    refine_modes: Sequence[bool],
) -> dict[str, object]:
    corpora = {
        "aae-v2": root / "data" / "processed" / "aae.jsonl",
        "microtexts-en": root / "data" / "processed" / "microtexts.jsonl",
    }

    report: dict[str, object] = {
        "splits": list(splits),
        "conditions": list(CONDITIONS),
        "corpora": {},
    }

    for name, path in corpora.items():
        if not path.exists():
            continue
        docs = [d for d in iter_jsonl(path) if d.split in splits]
        if limit is not None:
            docs = docs[:limit]
        if not docs:
            continue

        rows = [
            measure(docs, condition, refine=refine)
            for refine in refine_modes
            for condition in CONDITIONS
        ]
        corpora_report: dict[str, object] = report["corpora"]  # type: ignore[assignment]
        corpora_report[name] = {
            "documents": len(docs),
            "components": sum(len(d.components) for d in docs),
            "measurements": rows,
        }

    out_dir = root / "results" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "alignment_error.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="measure only the v1 coarse matcher, skipping the refinement pass",
    )
    args = parser.parse_args()

    refine_modes = [False] if args.no_refine else [False, True]
    report = run(args.root, splits=args.splits, limit=args.limit, refine_modes=refine_modes)

    corpora: dict[str, dict[str, object]] = report["corpora"]  # pyright: ignore[reportAssignmentType]
    for corpus, payload in corpora.items():
        print(f"\n{corpus}: {payload['documents']} docs, {payload['components']} components")
        print(
            f"  {'condition':<22} {'refined':>7} {'exact':>7} "
            f"{'>=50%':>7} {'med|ds|':>8} {'p90|ds|':>8}"
        )
        rows: list[dict[str, object]] = payload["measurements"]  # pyright: ignore[reportAssignmentType]
        for row in rows:
            print(
                f"  {row['condition']:<22} {row['refined']!s:>7} "
                f"{row['exact_span_rate']:>7.3f} {row['overlap_ge_50_rate']:>7.3f} "
                f"{row['start_error_median']:>8.0f} {row['start_error_p90']:>8.0f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
