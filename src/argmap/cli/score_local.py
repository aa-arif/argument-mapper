"""Score a local-model generation file into the baseline report shape.

    uv run python -m argmap.cli.score_local \
        --generations data/generations/aae-v2_test_constrained.json \
        --name qwen3.5-2b-lora

Produces `results/baselines/{name}_{corpus}_{split}.json` with the same
structure `run_baseline` writes for the Claude routes, so the two are directly
comparable and the README table has one shape throughout.

Cost is reported as GPU seconds rather than dollars. Converting to dollars per
document needs a throughput figure measured under load, which is milestone 6's
job; putting a guess here would be a number nobody could trace to a script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from argmap.cli.run_baseline import load_split, score_and_summarise
from argmap.cli.select_checkpoint import predictions_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--name", default="qwen3.5-2b-lora")
    parser.add_argument("--corpus", default="aae-v2")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    payload = cast(
        "dict[str, Any]",
        json.loads((args.root / args.generations).read_text(encoding="utf-8")),
    )
    run = payload["runs"][0]
    docs = load_split(args.root, args.corpus, args.split)
    predictions, health = predictions_for_run(docs, run["results"])

    tokens = [int(r.get("output_tokens", 0)) for r in run["results"]]
    seconds = float(run.get("generate_seconds", 0.0))
    n = len(run["results"])

    report: dict[str, object] = {
        "model": args.name,
        "corpus": args.corpus,
        "split": args.split,
        "constrained": payload.get("constrained"),
        "adapter": run.get("adapter"),
        "gpu": payload.get("gpu"),
        "run": {
            "documents": n,
            "invalid_outputs": health["invalid_json"],
            "unalignable_components": health["unalignable_components"],
            "truncated": health["truncated"],
            "output_tokens": sum(tokens),
            "mean_output_tokens": round(sum(tokens) / n, 1) if n else 0.0,
            # Batched throughput on one GPU, not per-request latency. The
            # serving numbers in milestone 6 are what a user would experience.
            "generate_seconds": round(seconds, 2),
            "batched_docs_per_second": round(n / seconds, 3) if seconds else 0.0,
            "engine_load_seconds": payload.get("load_seconds"),
        },
        "scores": score_and_summarise(
            docs,
            predictions,
            typed_available=any(d.typed for d in docs),
        ),
    }

    out_dir = args.root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.corpus}_{args.split}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    pred_dir = args.root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with (pred_dir / f"{stem}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for doc_id in sorted(predictions):
            fh.write(predictions[doc_id].model_dump_json())
            fh.write("\n")

    scores = cast("dict[str, Any]", report["scores"])
    run_info = cast("dict[str, Any]", report["run"])
    print(f"{args.name} on {args.corpus}/{args.split} ({n} docs)")
    print(
        f"  invalid {run_info['invalid_outputs']}  unalignable "
        f"{run_info['unalignable_components']}  "
        f"{run_info['batched_docs_per_second']} docs/s batched\n"
    )
    for label, entry in scores.items():
        if entry.get("status") == "not_applicable":
            continue
        c = entry["components"]["f1_ci"]
        r = entry["relations"]["f1_ci"]
        print(
            f"  {label:<22} comp {c['point']:.3f} [{c['ci_low']:.3f},{c['ci_high']:.3f}]"
            f"   rel {r['point']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
