"""Measure what JSON-schema constrained decoding actually buys.

    uv run python -m argmap.cli.constrained_ablation \
        --constrained data/generations/aae-v2_test_constrained.json \
        --unconstrained data/generations/aae-v2_test_unconstrained.json

Milestone 4. The two inputs come from the same model, the same prompts and the
same greedy decoding, differing only in whether vLLM was given the schema, so
any difference between them is attributable to the constraint.

Reports three things:

* **Invalid-output rate** -- how often the raw text fails to parse as JSON at
  all. This is the number that decides whether the JSON-repair path inherited
  from v1 is needed on the local route.
* **Truncation rate** -- how often generation hit the token cap. Constrained
  decoding guarantees syntax but not termination, so a bounded grammar and an
  unbounded one fail very differently here (DECISIONS D23).
* **F1 with a paired bootstrap** -- whether the constraint changes extraction
  quality, or only its parseability. Reported twice: over every document, and
  over only those where *both* runs produced parseable JSON.

The second scoring is not a courtesy. An unconstrained run that fails to parse
contributes no predictions at all on that document, which costs it recall and
costs it nothing in precision -- so comparing over every document credits the
unconstrained run for the documents it gave up on. Restricting to the documents
both runs answered is the like-for-like comparison, and the difference between
the two numbers is the size of that effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from argmap.cli.run_baseline import load_split
from argmap.cli.select_checkpoint import predictions_for_run
from argmap.metrics.bootstrap import bootstrap_f1, paired_bootstrap_delta
from argmap.metrics.matching import STANDARD_CRITERIA
from argmap.metrics.scores import score_corpus
from argmap.schema import Document, Prediction


def _load_run(path: Path) -> tuple[list[dict[str, Any]], bool]:
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    runs = payload["runs"]
    if len(runs) != 1:
        msg = f"{path} holds {len(runs)} runs; the ablation compares one against one"
        raise SystemExit(msg)
    return runs[0]["results"], bool(payload.get("constrained"))


def _health(results: list[dict[str, Any]]) -> dict[str, object]:
    total = len(results)
    invalid = 0
    truncated = 0
    for entry in results:
        if entry.get("finish_reason") == "length":
            truncated += 1
        try:
            parsed = json.loads(str(entry.get("text", "")))
        except (json.JSONDecodeError, TypeError):
            invalid += 1
            continue
        if not isinstance(parsed, dict):
            invalid += 1

    return {
        "documents": total,
        "invalid_json": invalid,
        "invalid_rate": round(invalid / total, 4) if total else 0.0,
        "truncated": truncated,
        "truncation_rate": round(truncated / total, 4) if total else 0.0,
        "mean_output_tokens": (
            round(sum(int(r.get("output_tokens", 0)) for r in results) / total, 1) if total else 0.0
        ),
    }


def unparseable(results: list[dict[str, Any]]) -> set[str]:
    """Document ids whose generated text is not a JSON object."""
    out: set[str] = set()
    for entry in results:
        try:
            parsed = json.loads(str(entry.get("text", "")))
        except (json.JSONDecodeError, TypeError):
            out.add(str(entry.get("doc_id", "")))
            continue
        if not isinstance(parsed, dict):
            out.add(str(entry.get("doc_id", "")))
    return out


def compare(
    docs: list[Document],
    constrained: dict[str, Prediction],
    unconstrained: dict[str, Prediction],
) -> dict[str, object]:
    report: dict[str, object] = {}
    typed_available = any(d.typed for d in docs)

    for criterion in STANDARD_CRITERIA:
        if criterion.typed and not typed_available:
            report[criterion.label] = {"status": "not_applicable"}
            continue

        c_scores = score_corpus(docs, constrained, criterion)
        u_scores = score_corpus(docs, unconstrained, criterion)

        entry: dict[str, object] = {}
        for kind in ("components", "relations"):
            c = [getattr(s, kind) for s in c_scores]
            u = [getattr(s, kind) for s in u_scores]
            delta = paired_bootstrap_delta(c, u)
            entry[kind] = {
                "constrained": bootstrap_f1(c).as_dict(),
                "unconstrained": bootstrap_f1(u).as_dict(),
                "delta": delta.as_dict(),
                "significant": delta.excludes_zero,
            }
        report[criterion.label] = entry

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--unconstrained", type=Path, required=True)
    parser.add_argument("--corpus", default="aae-v2")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    docs = load_split(args.root, args.corpus, args.split)

    c_results, c_flag = _load_run(args.root / args.constrained)
    u_results, u_flag = _load_run(args.root / args.unconstrained)
    if not c_flag or u_flag:
        print(
            "warning: the files' own `constrained` flags do not match the "
            f"arguments (constrained={c_flag}, unconstrained={u_flag})"
        )

    c_preds, _ = predictions_for_run(docs, c_results)
    u_preds, _ = predictions_for_run(docs, u_results)

    payload: dict[str, object] = {
        "corpus": args.corpus,
        "split": args.split,
        "documents": len(docs),
        "health": {
            "constrained": _health(c_results),
            "unconstrained": _health(u_results),
        },
        "scores": compare(docs, c_preds, u_preds),
    }

    # Like-for-like: only the documents both runs actually answered. A run that
    # emits nothing is scored as predicting nothing, which is a free pass on
    # precision, so the all-documents comparison flatters whichever run failed
    # more often.
    skipped = unparseable(c_results) | unparseable(u_results)
    both = [d for d in docs if d.doc_id not in skipped]
    if skipped:
        payload["both_parsed"] = {
            "documents": len(both),
            "excluded": sorted(skipped),
            "scores": compare(both, c_preds, u_preds),
        }

    out_dir = args.root / "results" / "constrained"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ablation_{args.corpus}_{args.split}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    health = cast("dict[str, Any]", payload["health"])
    print(f"{len(docs)} {args.corpus}/{args.split} documents\n")
    print(f"  {'':<16}{'invalid':>10}{'truncated':>12}{'mean tokens':>14}")
    for name in ("constrained", "unconstrained"):
        h = health[name]
        print(
            f"  {name:<16}{h['invalid_rate']:>9.1%}{h['truncation_rate']:>12.1%}"
            f"{h['mean_output_tokens']:>14.0f}"
        )

    both_parsed = cast("dict[str, Any] | None", payload.get("both_parsed"))
    if both_parsed:
        print(
            f"\n  {len(both_parsed['excluded'])} document(s) excluded below: "
            "one run produced no parseable JSON, so scoring them compares an "
            "answer against an abstention."
        )

    print()
    scores = cast("dict[str, Any]", payload["scores"])
    for criterion in STANDARD_CRITERIA:
        entry = scores.get(criterion.label)
        if not isinstance(entry, dict) or entry.get("status") == "not_applicable":
            continue
        for kind in ("components", "relations"):
            block = cast("dict[str, Any]", entry[kind])
            d = cast("dict[str, float]", block["delta"])
            verdict = "significant" if block["significant"] else "within noise"
            print(
                f"  {criterion.label:<22}{kind:<12}"
                f"{block['constrained']['point']:.3f} vs {block['unconstrained']['point']:.3f}  "
                f"delta {d['point']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  {verdict}"
            )

    if both_parsed:
        print(f"\n  like-for-like, {both_parsed['documents']} documents both runs answered:")
        subset = cast("dict[str, Any]", both_parsed["scores"])
        for criterion in STANDARD_CRITERIA:
            entry = subset.get(criterion.label)
            if not isinstance(entry, dict) or entry.get("status") == "not_applicable":
                continue
            for kind in ("components", "relations"):
                block = cast("dict[str, Any]", entry[kind])
                d = cast("dict[str, float]", block["delta"])
                verdict = "significant" if block["significant"] else "within noise"
                print(
                    f"  {criterion.label:<22}{kind:<12}"
                    f"{block['constrained']['point']:.3f} vs "
                    f"{block['unconstrained']['point']:.3f}  "
                    f"delta {d['point']:+.3f} "
                    f"[{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  {verdict}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
