"""Generate validation predictions for every saved checkpoint.

    PYTHONIOENCODING=utf-8 uv run --with modal modal run scripts/gpu/gen_val.py

Writes raw generations to `data/generations/` for
`argmap.cli.select_checkpoint` to parse and score. The split is deliberate:
this half needs a GPU and vLLM, the scoring half needs the project's metrics
and neither. Keeping them apart means checkpoint selection can be re-run,
re-scored, or re-argued without touching a GPU again.

Only the validation split is read. Nothing here can see test.
"""

from __future__ import annotations

import json
import pathlib

from generate import generate
from modal_common import MODELS_DIR, app

BASE_MODEL = "Qwen/Qwen3.5-2B"
SEEDS = (0, 1, 2)
EPOCHS = (1, 2, 3)


def _adapter_paths(base_model: str) -> list[str]:
    stem = f"{MODELS_DIR}/adapters/{base_model.replace('/', '__')}"
    return [f"{stem}/seed{seed}/epoch{epoch}" for seed in SEEDS for epoch in EPOCHS]


@app.local_entrypoint()
def main(
    base_model: str = BASE_MODEL,
    split: str = "val",
    out_name: str = "val_checkpoints",
) -> None:
    from argmap.cli.run_baseline import load_split
    from argmap.train_data import to_messages

    root = pathlib.Path(__file__).resolve().parents[2]
    docs = load_split(root, "aae-v2", split)
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    adapters = _adapter_paths(base_model)
    print(f"{len(prompts)} {split} documents x {len(adapters)} checkpoints")

    payload = generate.remote(
        model_path=base_model,
        prompts=prompts,
        adapters=adapters,
    )

    out = root / "data" / "generations" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nengine load {payload['load_seconds']}s")
    for run in payload["runs"]:
        name = (run["adapter"] or "base").split("/")[-3:]
        print(
            f"  {'/'.join(name):<28} {run['generate_seconds']:>7.1f}s  "
            f"{run['throughput_docs_per_second']:>6.2f} docs/s"
        )
    print(f"\nwrote {out}")
