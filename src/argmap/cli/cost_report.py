"""What a document costs, per model and per corpus, from the spend ledger.

    uv run python -m argmap.cli.cost_report

Writes `results/cost/summary.json`, which is where the README's cost table
comes from. It exists because the obvious way to compute a cost per document —
average every ledger row for a model — silently blends corpora.

AAE essays average 1,974 characters and arg-microtexts 423, so the same model
costs roughly twice as much per AAE document. A blended mean is a price for
neither corpus, and it moves with how many of each happen to have been run,
which is a property of the project's history rather than of anything being
measured. Every figure here is scoped to one model on one split of one corpus.

The local price is the A10G hourly rate divided by measured serving
throughput, so the ratios compare a metered API call against a metered GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

#: A10G on Modal, the GPU every local measurement in this project ran on.
GPU_USD_PER_HOUR = 1.10

#: Corpus name -> the splits file that lists its document ids.
SPLIT_FILES = {
    "aae-v2": "aae_splits.json",
    "microtexts-en": "microtexts_splits.json",
}


def load_ledger(root: Path) -> list[dict[str, Any]]:
    path = root / "results" / "cost" / "ledger.jsonl"
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_ids(root: Path) -> dict[tuple[str, str], set[str]]:
    """(corpus, split) -> document ids, from the committed split files."""
    out: dict[tuple[str, str], set[str]] = {}
    for corpus, filename in SPLIT_FILES.items():
        path = root / "splits" / filename
        if not path.exists():
            continue
        payload = cast("dict[str, list[str]]", json.loads(path.read_text(encoding="utf-8")))
        for split, ids in payload.items():
            if ids:
                out[(corpus, split)] = set(ids)
    return out


def usd_per_doc(rows: Iterable[dict[str, Any]], model: str, ids: set[str]) -> dict[str, Any]:
    """Mean dollars per document, restricted to one model and one id set."""
    costs = [
        float(row.get("dollars", 0.0))
        for row in rows
        if row.get("model") == model and not row.get("cached") and str(row.get("doc_id", "")) in ids
    ]
    if not costs:
        return {"calls": 0, "usd_per_doc": None, "usd_per_1k": None}
    mean = statistics.fmean(costs)
    return {
        "calls": len(costs),
        "usd_total": round(sum(costs), 6),
        "usd_per_doc": round(mean, 8),
        "usd_per_1k": round(mean * 1000, 4),
    }


def local_usd_per_doc(docs_per_second: float) -> float:
    if docs_per_second <= 0:
        return 0.0
    return GPU_USD_PER_HOUR / (docs_per_second * 3600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--local-req-per-second",
        type=float,
        default=None,
        help="served throughput; defaults to the peak in results/serving/loadtest.json",
    )
    args = parser.parse_args()

    rows = load_ledger(args.root)
    ids = split_ids(args.root)
    models = sorted({str(r["model"]) for r in rows if r.get("model")})

    # Serving throughput, measured, not assumed.
    throughput = args.local_req_per_second
    loadtest = args.root / "results" / "serving" / "loadtest.json"
    source = "--local-req-per-second"
    if throughput is None and loadtest.exists():
        payload = cast("dict[str, Any]", json.loads(loadtest.read_text(encoding="utf-8")))
        levels = cast("list[dict[str, Any]]", payload.get("levels", []))
        if levels:
            throughput = max(float(level["throughput_rps"]) for level in levels)
            source = "results/serving/loadtest.json (highest concurrency tested)"
    if throughput is None:
        msg = "no throughput available; pass --local-req-per-second"
        raise SystemExit(msg)

    local = local_usd_per_doc(throughput)

    by_scope: dict[str, Any] = {}
    for (corpus, split), id_set in sorted(ids.items()):
        scope: dict[str, Any] = {"documents": len(id_set)}
        for model in models:
            entry = usd_per_doc(rows, model, id_set)
            if entry["calls"]:
                per_doc = float(entry["usd_per_doc"])
                entry["vs_local"] = round(per_doc / local, 1) if local else None
            scope[model] = entry
        by_scope[f"{corpus}/{split}"] = scope

    # Rows belonging to no split at all: serving smoke tests, ad-hoc requests.
    every_id: set[str] = {doc_id for group in ids.values() for doc_id in group}
    unscoped = [r for r in rows if str(r.get("doc_id", "")) not in every_id]

    payload = {
        "note": (
            "Costs are scoped to one model on one split of one corpus. "
            "Averaging a model's whole ledger blends corpora of different "
            "lengths and prices neither."
        ),
        "local": {
            "gpu": "A10G",
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "requests_per_second": round(throughput, 4),
            "documents_per_hour": round(throughput * 3600, 1),
            "usd_per_doc": round(local, 8),
            "usd_per_1k": round(local * 1000, 4),
            "throughput_source": source,
        },
        "by_scope": by_scope,
        "unscoped_rows": {
            "calls": len(unscoped),
            "usd_total": round(sum(float(r.get("dollars", 0.0)) for r in unscoped), 6),
            "doc_ids": sorted({str(r.get("doc_id", "")) for r in unscoped}),
        },
        "ledger_totals": {
            model: {
                "calls": sum(1 for r in rows if r.get("model") == model),
                "usd_total": round(
                    sum(float(r.get("dollars", 0.0)) for r in rows if r.get("model") == model), 6
                ),
            }
            for model in models
        },
    }

    out = args.root / "results" / "cost" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(
        f"local: ${local:.8f}/doc  (${local * 1000:.4f}/1k) at {throughput:.3f} req/s on an A10G\n"
    )
    for label, entry in by_scope.items():
        print(f"{label} ({entry['documents']} docs)")
        for model in models:
            row = entry[model]
            if not row["calls"]:
                continue
            print(
                f"  {model:<18} ${row['usd_per_doc']:.6f}/doc  "
                f"${row['usd_per_1k']:>7.2f}/1k  {row['vs_local']:>6.1f}x local  "
                f"({row['calls']} calls)"
            )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
