"""Run an extractor over a corpus split and score it.

    uv run python -m argmap.cli.run_baseline --model claude-haiku-4-5 --corpus aae-v2 --split test

Writes per-document predictions and a scored report to `results/baselines/`.
Responses are cached on disk, so re-running is free and idempotent; the cost
ledger records only calls that actually went to the API.

Tuning happens on `val`. `test` is for reporting a final number once.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from argmap.cost import SpendLedger
from argmap.extractors.base import ExtractionResult
from argmap.extractors.cache import ResponseCache
from argmap.extractors.claude import ClaudeExtractor
from argmap.metrics.bootstrap import bootstrap_f1
from argmap.metrics.matching import STANDARD_CRITERIA
from argmap.metrics.scores import score_corpus, sum_counts
from argmap.prompts import load_exemplars, select_exemplars, write_exemplars
from argmap.schema import Document, Prediction, iter_jsonl

CORPUS_FILES = {
    "aae-v2": "aae.jsonl",
    "microtexts-en": "microtexts.jsonl",
}


def load_split(root: Path, corpus: str, split: str) -> list[Document]:
    path = root / "data" / "processed" / CORPUS_FILES[corpus]
    if not path.exists():
        msg = f"{path} not found - run `uv run argmap-build-datasets` first"
        raise FileNotFoundError(msg)
    return [d for d in iter_jsonl(path) if d.split == split]


def ensure_exemplars(root: Path, k: int, seed: int) -> list[dict[str, object]]:
    """Load the exemplar file, generating it from the training split if absent.

    The file is gitignored: it contains corpus text, and committed source on a
    public repository is publication (DECISIONS.md D2).
    """
    path = root / "prompts" / "exemplars.json"
    if path.exists():
        return load_exemplars(path)

    train = load_split(root, "aae-v2", "train")
    chosen = select_exemplars(train, k=k, seed=seed)
    write_exemplars(chosen, path)
    return load_exemplars(path)


def score_and_summarise(
    docs: Sequence[Document],
    predictions: dict[str, Prediction],
    *,
    typed_available: bool,
) -> dict[str, object]:
    """Score under every criterion, with document-level bootstrap intervals."""
    report: dict[str, object] = {}

    for criterion in STANDARD_CRITERIA:
        if criterion.typed and not typed_available:
            # arg-microtexts carries no component types, so typed metrics are
            # not computable there. N/A is the honest answer; a number would
            # imply labels the annotators never assigned.
            report[criterion.label] = {"status": "not_applicable"}
            continue

        per_doc = score_corpus(docs, predictions, criterion)
        comp = [s.components for s in per_doc]
        rel = [s.relations for s in per_doc]

        comp_total = sum_counts(comp)
        rel_total = sum_counts(rel)

        entry: dict[str, object] = {
            "components": {
                **comp_total.as_dict(),
                "f1_ci": bootstrap_f1(comp).as_dict(),
            },
            "relations": {
                **rel_total.as_dict(),
                "f1_ci": bootstrap_f1(rel).as_dict(),
            },
        }

        by_type: dict[str, dict[str, object]] = {}
        for label in sorted({k for s in per_doc for k in s.components_by_type}):
            counts = [s.components_by_type.get(label) for s in per_doc]
            present = [c for c in counts if c is not None]
            by_type[label] = {
                **sum_counts(present).as_dict(),
                "f1_ci": bootstrap_f1(present).as_dict(),
            }
        entry["components_by_type"] = by_type

        rel_by_type: dict[str, dict[str, object]] = {}
        for label in sorted({k for s in per_doc for k in s.relations_by_type}):
            counts = [s.relations_by_type.get(label) for s in per_doc]
            present = [c for c in counts if c is not None]
            rel_by_type[label] = {
                **sum_counts(present).as_dict(),
                "f1_ci": bootstrap_f1(present).as_dict(),
            }
        entry["relations_by_type"] = rel_by_type

        report[criterion.label] = entry

    return report


def summarise_run(results: dict[str, ExtractionResult]) -> dict[str, object]:
    """Cost, latency and failure statistics for the run itself."""
    live = [r for r in results.values() if not r.cached]
    latencies = sorted(r.latency_s for r in live)

    def pct(q: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(len(latencies) - 1, int(q * (len(latencies) - 1)))]

    return {
        "documents": len(results),
        "api_calls": len(live),
        "cache_hits": len(results) - len(live),
        "errors": sum(1 for r in results.values() if r.error),
        "invalid_outputs": sum(1 for r in results.values() if r.invalid_output),
        "unalignable_components": sum(
            v if isinstance(v := r.prediction.meta.get("unalignable_components"), int) else 0
            for r in results.values()
        ),
        "dollars_total": round(sum(r.dollars for r in results.values()), 6),
        "dollars_per_doc": (
            round(sum(r.dollars for r in results.values()) / len(results), 6) if results else 0.0
        ),
        "input_tokens": sum(r.usage.input_tokens for r in results.values()),
        "output_tokens": sum(r.usage.output_tokens for r in results.values()),
        "cache_read_tokens": sum(r.usage.cache_read_tokens for r in results.values()),
        "cache_write_tokens": sum(r.usage.cache_write_tokens for r in results.values()),
        "latency_s": {
            "n": len(latencies),
            "p50": round(pct(0.50), 3),
            "p95": round(pct(0.95), 3),
            "p99": round(pct(0.99), 3),
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        },
    }


def run(
    root: Path,
    *,
    model: str,
    corpus: str,
    split: str,
    limit: int | None,
    exemplar_count: int,
    exemplar_seed: int,
) -> dict[str, object]:
    docs = load_split(root, corpus, split)
    if limit is not None:
        docs = docs[:limit]
    if not docs:
        msg = f"no documents for {corpus}/{split}"
        raise SystemExit(msg)

    exemplars = ensure_exemplars(root, exemplar_count, exemplar_seed)
    ledger = SpendLedger(root / "results" / "cost" / "ledger.jsonl")
    cache = ResponseCache(root / ".cache" / "api")
    extractor = ClaudeExtractor(model=model, ledger=ledger, cache=cache, exemplars=exemplars)

    started = time.perf_counter()
    results: dict[str, ExtractionResult] = {}
    for index, doc in enumerate(docs, start=1):
        result = extractor.extract(doc)
        results[doc.doc_id] = result
        marker = "cache" if result.cached else f"${result.dollars:.4f}"
        status = "ERR " if result.error else "    "
        print(
            f"  [{index:>3}/{len(docs)}] {doc.doc_id:<14} {status}"
            f"{len(result.prediction.components):>3}c "
            f"{len(result.prediction.relations):>3}r  {marker}",
            flush=True,
        )
    wall = time.perf_counter() - started

    predictions = {k: v.prediction for k, v in results.items()}
    typed_available = any(d.typed for d in docs)

    report: dict[str, object] = {
        "model": model,
        "corpus": corpus,
        "split": split,
        "prompt_version": extractor.prompt_version,
        "exemplar_ids": [str(e.get("doc_id", "")) for e in exemplars],
        "wall_seconds": round(wall, 2),
        "run": summarise_run(results),
        "ledger": ledger.summary(),
        "scores": score_and_summarise(docs, predictions, typed_available=typed_available),
        "errors": {k: v.error for k, v in results.items() if v.error},
    }

    out_dir = root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{corpus}_{split}"
    payload = json.dumps(report, indent=2) + "\n"

    # Written twice on purpose. The stable name is what tables and the README
    # point at; the fingerprinted copy under history/ is immutable, so revising
    # the prompt cannot quietly erase the run that produced a number someone
    # has already written down.
    (out_dir / f"{stem}.json").write_text(payload, encoding="utf-8", newline="\n")
    history = out_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{stem}_{extractor.prompt_version}.json").write_text(
        payload, encoding="utf-8", newline="\n"
    )

    # Predictions carry corpus text, so they live under data/, which is
    # gitignored. Keyed by prompt version too, so an A/B comparison still has
    # both sides to run a paired bootstrap over.
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{stem}_{extractor.prompt_version}.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc_id in sorted(predictions):
            fh.write(predictions[doc_id].model_dump_json())
            fh.write("\n")

    return report


def _dig(value: object, *keys: str) -> object:
    """Walk a nested report dict without asserting its shape up front."""
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = cast("dict[str, object]", value).get(key)
    return value


def _format_ci(value: object) -> str:
    if not isinstance(value, dict):
        return "n/a"
    ci = cast("dict[str, object]", value)
    point, low, high = ci.get("point"), ci.get("ci_low"), ci.get("ci_high")
    if not all(isinstance(v, int | float) for v in (point, low, high)):
        return "n/a"
    return f"{float(point):.3f} [{float(low):.3f},{float(high):.3f}]"  # type: ignore[arg-type]


def _print_summary(report: dict[str, object]) -> None:
    print(f"\n{report['model']} on {report['corpus']}/{report['split']}")
    print(
        f"  {_dig(report, 'run', 'api_calls')} calls, "
        f"{_dig(report, 'run', 'cache_hits')} cached, "
        f"{_dig(report, 'run', 'errors')} errors, "
        f"${_dig(report, 'run', 'dollars_total')} total"
    )
    print(f"  {'criterion':<24}{'comp F1':>24}{'rel F1':>24}")
    for criterion in STANDARD_CRITERIA:
        label = criterion.label
        if _dig(report, "scores", label, "status") == "not_applicable":
            print(f"  {label:<24}{'n/a':>24}{'n/a':>24}")
            continue
        comp = _format_ci(_dig(report, "scores", label, "components", "f1_ci"))
        rel = _format_ci(_dig(report, "scores", label, "relations", "f1_ci"))
        print(f"  {label:<24}{comp:>24}{rel:>24}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--corpus", default="aae-v2", choices=sorted(CORPUS_FILES))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exemplars", type=int, default=2)
    parser.add_argument("--exemplar-seed", type=int, default=17)
    args = parser.parse_args()

    report = run(
        args.root,
        model=args.model,
        corpus=args.corpus,
        split=args.split,
        limit=args.limit,
        exemplar_count=args.exemplars,
        exemplar_seed=args.exemplar_seed,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
