"""Route low-confidence documents from the local model to Claude.

    uv run python -m argmap.cli.cascade --sweep-split val --report-split test

The idea is cheap quality: run the fine-tuned model on everything, and pay for
the API only where the local model reports it was unsure. Confidence is the
mean token logprob of the generated graph, which vLLM returns for free.

**The threshold is swept on validation and reported once on test.** Sweeping on
test and then quoting the best point would be selecting the operating point
using the data it is evaluated on, which is the most common way a cascade
result turns out to be fiction.

Cost per 1,000 documents combines two measured numbers:

* the API price per document, from the spend ledger
* the local price per document, from GPU hourly rate divided by measured
  batched throughput

Neither is an estimate; both trace to a run recorded under `results/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from argmap.cli.run_baseline import load_split
from argmap.cli.select_checkpoint import predictions_for_run
from argmap.metrics.bootstrap import bootstrap_f1, paired_bootstrap_delta
from argmap.metrics.matching import MatchCriterion
from argmap.metrics.scores import Counts, score_corpus, sum_counts
from argmap.schema import Document, Prediction

#: The criterion the operating point is chosen under. Overlap matching with
#: types, the same scalar checkpoint selection used.
CRITERION = MatchCriterion(span="overlap", typed=True)

#: A10G on Modal, the GPU every local measurement in this project ran on.
GPU_USD_PER_HOUR = 1.10


@dataclass(frozen=True)
class Operating:
    """One threshold and what it costs."""

    threshold: float
    escalated: int
    documents: int
    component_f1: float
    relation_f1: float
    usd_per_1k: float

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.documents if self.documents else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": round(self.threshold, 6),
            "escalated": self.escalated,
            "documents": self.documents,
            "escalation_rate": round(self.escalation_rate, 4),
            "component_f1": round(self.component_f1, 6),
            "relation_f1": round(self.relation_f1, 6),
            "usd_per_1k_documents": round(self.usd_per_1k, 4),
        }


def load_local(path: Path, docs: list[Document]) -> tuple[dict[str, Prediction], dict[str, float]]:
    """Local predictions plus each document's confidence."""
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    run = payload["runs"][0]
    predictions, _ = predictions_for_run(docs, run["results"])
    confidence = {
        str(r["doc_id"]): float(r["mean_logprob"])
        for r in run["results"]
        if r.get("mean_logprob") is not None
    }
    return predictions, confidence


def load_claude(root: Path, stem: str) -> dict[str, Prediction]:
    pred_dir = root / "data" / "predictions"
    matches = sorted(pred_dir.glob(f"{stem}*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        msg = f"no cached predictions for {stem}; run the baseline first"
        raise SystemExit(msg)
    out: dict[str, Prediction] = {}
    for line in matches[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            pred = Prediction.model_validate_json(line)
            out[pred.doc_id] = pred
    return out


def claude_usd_per_doc(root: Path, model: str) -> float:
    """Mean dollars per document for a model, from the ledger."""
    ledger = root / "results" / "cost" / "ledger.jsonl"
    if not ledger.exists():
        return 0.0
    costs = [
        float(row.get("dollars", 0.0))
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("model") == model and not row.get("cached")
    ]
    return statistics.fmean(costs) if costs else 0.0


def local_usd_per_doc(docs_per_second: float) -> float:
    """GPU rental divided by measured throughput."""
    if docs_per_second <= 0:
        return 0.0
    return GPU_USD_PER_HOUR / (docs_per_second * 3600)


def evaluate(
    docs: list[Document],
    local: dict[str, Prediction],
    claude: dict[str, Prediction],
    confidence: dict[str, float],
    threshold: float,
    *,
    local_cost: float,
    claude_cost: float,
) -> tuple[Operating, list[Counts], list[Counts]]:
    """Score the cascade at one threshold.

    A document with no confidence score escalates: an answer the model could
    not even report a logprob for is not one to trust silently.
    """
    blended: dict[str, Prediction] = {}
    escalated = 0
    for doc in docs:
        score = confidence.get(doc.doc_id)
        if score is None or score < threshold:
            escalated += 1
            blended[doc.doc_id] = claude.get(doc.doc_id, Prediction(doc_id=doc.doc_id))
        else:
            blended[doc.doc_id] = local.get(doc.doc_id, Prediction(doc_id=doc.doc_id))

    per_doc = score_corpus(docs, blended, CRITERION)
    comp = [s.components for s in per_doc]
    rel = [s.relations for s in per_doc]

    n = len(docs)
    # Every document pays the local price; escalated ones also pay the API.
    total = n * local_cost + escalated * claude_cost
    usd_per_1k = (total / n) * 1000 if n else 0.0

    return (
        Operating(
            threshold=threshold,
            escalated=escalated,
            documents=n,
            component_f1=sum_counts(comp).f1,
            relation_f1=sum_counts(rel).f1,
            usd_per_1k=usd_per_1k,
        ),
        comp,
        rel,
    )


def thresholds_from(confidence: dict[str, float], steps: int = 21) -> list[float]:
    """Quantiles of the observed confidences, plus the two extremes.

    Sweeping quantiles rather than a fixed grid keeps every step meaningful:
    an evenly spaced grid over an unknown range wastes most of its points
    where no document sits.
    """
    values = sorted(confidence.values())
    if not values:
        return [float("inf")]
    points = [values[0] - 1e-9]
    for i in range(1, steps):
        index = min(len(values) - 1, int(i / steps * len(values)))
        points.append(values[index])
    points.append(values[-1] + 1e-9)
    return sorted(set(points))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus", default="aae-v2")
    parser.add_argument("--sweep-split", default="val")
    parser.add_argument("--report-split", default="test")
    parser.add_argument("--claude-model", default="claude-haiku-4-5")
    parser.add_argument(
        "--local-val",
        type=Path,
        default=Path("data/generations/val_e10_idpattern.json"),
    )
    parser.add_argument(
        "--local-test",
        type=Path,
        default=Path("data/generations/aae-v2_test_constrained.json"),
    )
    parser.add_argument("--local-docs-per-second", type=float, default=1.487)
    args = parser.parse_args()

    claude_cost = claude_usd_per_doc(args.root, args.claude_model)
    local_cost = local_usd_per_doc(args.local_docs_per_second)

    # ---- sweep on validation -------------------------------------------------
    val_docs = load_split(args.root, args.corpus, args.sweep_split)
    val_local, val_conf = load_local(args.root / args.local_val, val_docs)
    val_claude = load_claude(args.root, f"{args.claude_model}_{args.corpus}_{args.sweep_split}")

    sweep: list[Operating] = []
    for threshold in thresholds_from(val_conf):
        point, _, _ = evaluate(
            val_docs,
            val_local,
            val_claude,
            val_conf,
            threshold,
            local_cost=local_cost,
            claude_cost=claude_cost,
        )
        sweep.append(point)

    # Claude-only is the ceiling the cascade is trying to match: every document
    # escalated.
    ceiling, ceiling_comp, _ = evaluate(
        val_docs,
        val_local,
        val_claude,
        val_conf,
        float("inf"),
        local_cost=local_cost,
        claude_cost=claude_cost,
    )

    # ---- choose an operating point on validation ----------------------------
    chosen = None
    for point in sorted(sweep, key=lambda p: p.usd_per_1k):
        _, comp, _ = evaluate(
            val_docs,
            val_local,
            val_claude,
            val_conf,
            point.threshold,
            local_cost=local_cost,
            claude_cost=claude_cost,
        )
        delta = paired_bootstrap_delta(comp, ceiling_comp)
        # "Matches Claude within its interval" means the paired difference does
        # not exclude zero -- not that two marginal intervals happen to overlap.
        if not delta.excludes_zero:
            chosen = point
            break

    # ---- report once on test -------------------------------------------------
    test_docs = load_split(args.root, args.corpus, args.report_split)
    test_local, test_conf = load_local(args.root / args.local_test, test_docs)
    test_claude = load_claude(args.root, f"{args.claude_model}_{args.corpus}_{args.report_split}")

    threshold = chosen.threshold if chosen else float("inf")
    test_point, test_comp, test_rel = evaluate(
        test_docs,
        test_local,
        test_claude,
        test_conf,
        threshold,
        local_cost=local_cost,
        claude_cost=claude_cost,
    )
    _, test_ceiling_comp, test_ceiling_rel = evaluate(
        test_docs,
        test_local,
        test_claude,
        test_conf,
        float("inf"),
        local_cost=local_cost,
        claude_cost=claude_cost,
    )

    report: dict[str, object] = {
        "corpus": args.corpus,
        "sweep_split": args.sweep_split,
        "report_split": args.report_split,
        "criterion": CRITERION.label,
        "claude_model": args.claude_model,
        "costs": {
            "claude_usd_per_doc": round(claude_cost, 6),
            "local_usd_per_doc": round(local_cost, 8),
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "local_docs_per_second": args.local_docs_per_second,
        },
        "validation_sweep": [p.as_dict() for p in sweep],
        "validation_ceiling": ceiling.as_dict(),
        "selected_threshold": None if chosen is None else round(chosen.threshold, 6),
        "selected_on": args.sweep_split,
        "test": {
            **test_point.as_dict(),
            "component_f1_ci": bootstrap_f1(test_comp).as_dict(),
            "relation_f1_ci": bootstrap_f1(test_rel).as_dict(),
            "vs_claude_only_components": paired_bootstrap_delta(
                test_comp, test_ceiling_comp
            ).as_dict(),
            "vs_claude_only_relations": paired_bootstrap_delta(
                test_rel, test_ceiling_rel
            ).as_dict(),
        },
    }

    out_dir = args.root / "results" / "cascade"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cascade_{args.corpus}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"cost per document: local ${local_cost:.6f}  {args.claude_model} ${claude_cost:.4f}\n")
    print(f"validation sweep ({len(val_docs)} docs)")
    print(f"  {'threshold':>11}{'escalated':>11}{'comp F1':>10}{'rel F1':>9}{'$/1k':>10}")
    for point in sweep:
        print(
            f"  {point.threshold:>11.4f}{point.escalation_rate:>10.0%}"
            f"{point.component_f1:>10.3f}{point.relation_f1:>9.3f}{point.usd_per_1k:>10.2f}"
        )
    print(
        f"  {'claude-only':>11}{'100%':>10}{ceiling.component_f1:>10.3f}"
        f"{ceiling.relation_f1:>9.3f}{ceiling.usd_per_1k:>10.2f}"
    )

    print(f"\nselected threshold: {report['selected_threshold']} (on {args.sweep_split})")
    print(f"\ntest ({len(test_docs)} docs)")
    print(
        f"  escalated {test_point.escalation_rate:.0%}  "
        f"comp F1 {test_point.component_f1:.3f}  rel F1 {test_point.relation_f1:.3f}  "
        f"${test_point.usd_per_1k:.2f}/1k"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
