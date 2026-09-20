"""Export the training and validation splits as fine-tuning data.

    uv run python -m argmap.cli.export_sft

Writes `data/sft/train.jsonl` and `data/sft/val.jsonl`, which
`scripts/gpu/train_lora.py` reads and ships to Modal. Both live under `data/`
and are gitignored -- they embed corpus text verbatim.

The test split is deliberately not exported. Nothing that touches the GPU
should be able to read it by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from argmap.cli.run_baseline import load_split
from argmap.schema import Document
from argmap.train_data import answer_token_estimate, write_sft_jsonl


def _stats(docs: list[Document]) -> dict[str, object]:
    answers = [answer_token_estimate(d) for d in docs]
    prompts = [len(d.text) // 3 for d in docs]
    return {
        "documents": len(docs),
        "mean_answer_tokens": round(sum(answers) / len(answers), 1) if answers else 0.0,
        "max_answer_tokens": max(answers, default=0),
        "mean_prompt_tokens": round(sum(prompts) / len(prompts), 1) if prompts else 0.0,
        "max_prompt_tokens": max(prompts, default=0),
        "max_total_tokens": max((a + p for a, p in zip(answers, prompts, strict=True)), default=0),
    }


def export(root: Path) -> dict[str, object]:
    out_dir = root / "data" / "sft"
    summary: dict[str, object] = {}

    for split in ("train", "val"):
        docs = load_split(root, "aae-v2", split)
        write_sft_jsonl(docs, out_dir / f"{split}.jsonl")
        summary[split] = _stats(docs)

    report_dir = root / "results" / "training"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "sft_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    summary = export(args.root)
    print(json.dumps(summary, indent=2))

    longest = max(
        int(v["max_total_tokens"])  # type: ignore[index]
        for v in summary.values()
        if isinstance(v, dict)
    )
    print(f"\nlongest example ~{longest} tokens (max_seq_length must exceed this)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
