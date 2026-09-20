"""Generate with the base model alone, for comparison against the adapters.

    PYTHONIOENCODING=utf-8 uv run --with modal modal run scripts/gpu/gen_base.py

Nine different adapters produced byte-identical output. The weights are real
and differ between checkpoints, and vLLM accepts the `lora_request` parameter,
so the question left is simply whether that output is the *base* model's --
which would mean the adapters are being loaded and then not applied.
"""

from __future__ import annotations

import json
import pathlib

from generate import generate
from modal_common import app


@app.local_entrypoint()
def main(base_model: str = "Qwen/Qwen3.5-2B", limit: int = 8) -> None:
    from argmap.cli.run_baseline import load_split
    from argmap.train_data import to_messages

    root = pathlib.Path(__file__).resolve().parents[2]
    docs = load_split(root, "aae-v2", "val")[:limit]
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    payload = generate.remote(model_path=base_model, prompts=prompts, adapters=None)

    out = root / "data" / "generations" / "val_base.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    adapter_file = root / "data" / "generations" / "val_checkpoints.json"
    if adapter_file.exists():
        adapter_runs = json.loads(adapter_file.read_text(encoding="utf-8"))["runs"]
        by_doc = {r["doc_id"]: r["text"] for r in adapter_runs[0]["results"]}
        same = 0
        for r in payload["runs"][0]["results"]:
            if by_doc.get(r["doc_id"]) == r["text"]:
                same += 1
        total = len(payload["runs"][0]["results"])
        print(f"\nbase vs adapter output: {same}/{total} documents identical")
        print(
            "  -> adapters are NOT being applied"
            if same == total
            else "  -> adapters ARE changing the output"
        )
