"""How much of the headline result is the recipe, and how much is the seed?

    uv run python -m argmap.cli.seed_variance --split val
    uv run python -m argmap.cli.seed_variance --split test

Three seeds were trained, but after the serving fix (DECISIONS D27) only seed 0
had been merged and evaluated, so every number in the README rested on one
random draw. This reads the per-seed baseline reports and reports mean and
sample standard deviation for each criterion.

**Seed 0 remains the selected and reported model.** It was chosen on
validation before the other seeds were scored, and nothing here changes that:
this is a variance measurement, not a selection. Running it on `test` says how
stable the reported number is, not which seed to report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

#: Seed 0's report is written under the plain name; the others carry a suffix.
DEFAULT_MODELS = ("qwen3.5-2b-merged", "qwen3.5-2b-merged-seed1", "qwen3.5-2b-merged-seed2")


def _report(root: Path, model: str, corpus: str, split: str) -> dict[str, Any] | None:
    path = root / "results" / "baselines" / f"{model}_{corpus}_{split}.json"
    if not path.exists():
        return None
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus", default="aae-v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--selected", default=None, help="which model is the reported one")
    args = parser.parse_args()

    selected = args.selected or args.models[0]
    reports: dict[str, dict[str, Any]] = {}
    for model in args.models:
        report = _report(args.root, model, args.corpus, args.split)
        if report is None:
            print(f"missing: {model}_{args.corpus}_{args.split}.json -- skipped")
            continue
        reports[model] = report

    if len(reports) < 2:
        msg = "need at least two seeds to report variance"
        raise SystemExit(msg)

    criteria = sorted(
        {
            label
            for report in reports.values()
            for label, entry in report["scores"].items()
            if entry.get("status") != "not_applicable"
        }
    )

    summary: dict[str, Any] = {}
    for label in criteria:
        entry: dict[str, Any] = {}
        for kind in ("components", "relations"):
            values = [
                float(report["scores"][label][kind]["f1_ci"]["point"])
                for report in reports.values()
            ]
            entry[kind] = {
                "n_seeds": len(values),
                "mean": round(statistics.fmean(values), 6),
                # Sample standard deviation: three seeds are a sample of the
                # recipe's behaviour, not the whole population of seeds.
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "selected": round(
                    float(reports[selected]["scores"][label][kind]["f1_ci"]["point"]), 6
                )
                if selected in reports
                else None,
                "by_model": {m: round(v, 6) for m, v in zip(reports, values, strict=True)},
            }
        summary[label] = entry

    payload: dict[str, object] = {
        "corpus": args.corpus,
        "split": args.split,
        "models": list(reports),
        "selected_model": selected,
        "note": (
            "Seed 0 was selected on validation before seeds 1 and 2 were scored. "
            "This is a variance measurement, not a selection."
        ),
        "scores": summary,
    }

    out_dir = args.root / "results" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"seed_variance_{args.corpus}_{args.split}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{len(reports)} seeds on {args.corpus}/{args.split}: {', '.join(reports)}\n")
    for label in criteria:
        comp = summary[label]["components"]
        rel = summary[label]["relations"]
        print(
            f"  {label:<22} comp {comp['mean']:.3f} +/- {comp['std']:.3f}"
            f"   rel {rel['mean']:.3f} +/- {rel['std']:.3f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
