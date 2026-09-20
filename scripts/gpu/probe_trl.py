"""Print the accepted fields of TRL's SFTConfig.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_trl.py

TRL's training-argument surface moves between releases, and a rejected keyword
is only discovered after the container has pulled several gigabytes of weights
and started a GPU. Runs on CPU, costs almost nothing, and answers the question
exactly rather than from memory.
"""

from __future__ import annotations

import json

from modal_common import TRAIN_IMAGE, app


@app.function(image=TRAIN_IMAGE, timeout=60 * 10)
def fields() -> dict[str, object]:
    import dataclasses

    import trl
    from trl import SFTConfig

    names = sorted(f.name for f in dataclasses.fields(SFTConfig))
    interesting = [
        n
        for n in names
        if any(
            key in n
            for key in (
                "warmup",
                "lr",
                "learning",
                "scheduler",
                "epoch",
                "batch",
                "accum",
                "eval",
                "save",
                "length",
                "completion",
                "seed",
                "bf16",
                "logging",
                "report",
                "packing",
            )
        )
    ]
    return json.loads(
        json.dumps(
            {
                "trl_version": trl.__version__,
                "field_count": len(names),
                "relevant_fields": interesting,
                "all_fields": names,
            },
            default=str,
        )
    )


@app.local_entrypoint()
def main() -> None:
    report = fields.remote()
    print(f"trl {report['trl_version']}, {report['field_count']} config fields")
    print("\nrelevant:")
    for name in report["relevant_fields"]:  # type: ignore[union-attr]
        print(f"  {name}")
