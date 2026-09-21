"""Generate test predictions for the selected checkpoint.

    PYTHONIOENCODING=utf-8 uv run --with modal modal run scripts/gpu/gen_test.py \
        --adapter /models/adapters/.../seed0/epoch6

Runs the same prompts twice, constrained and unconstrained, in one engine
session. That pairing is milestone 4's whole measurement: identical model,
identical prompts, identical decoding except for the JSON-schema grammar, so
the difference in invalid-output rate and F1 is attributable to the constraint
and nothing else.

This is the only step that touches the test split, and it runs once, after the
checkpoint has already been chosen on validation.
"""

from __future__ import annotations

import json
import pathlib

from generate import generate
from modal_common import MODELS_DIR, MODELS_VOLUME, app

BASE_MODEL = "Qwen/Qwen3.5-2B"


@app.local_entrypoint()
def main(
    adapter: str = "",  # volume-relative adapter, or use --merged
    base_model: str = BASE_MODEL,
    corpus: str = "aae-v2",
    split: str = "test",
    out_name: str = "",
    merged: str = "",
) -> None:
    from argmap.cli.run_baseline import load_split
    from argmap.prompts import bounded_graph_schema
    from argmap.train_data import to_messages

    # Git Bash rewrites a CLI argument beginning with "/" into a Windows
    # path ("/models/..." became "C:/Program Files/Git/models/..."), which
    # vLLM then tried to resolve as a Hugging Face repo id. Taking the
    # adapter path relative to the volume removes the leading slash and the
    # whole failure mode with it.
    # A merged checkpoint is served as a plain model. vLLM's dynamic LoRA
    # path silently produced base-model output for this architecture, so
    # merging is how the fine-tune actually reaches inference.
    if merged:
        model_path = merged if merged.startswith(MODELS_DIR) else f"{MODELS_DIR}/merged/{merged}"
        adapter_list = None
        label = model_path
    else:
        model_path = base_model
        adapter_list = [
            adapter if adapter.startswith(MODELS_DIR) else f"{MODELS_DIR}/adapters/{adapter}"
        ]
        label = adapter_list[0]

    root = pathlib.Path(__file__).resolve().parents[2]
    docs = load_split(root, corpus, split)
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]
    print(f"{len(prompts)} {corpus}/{split} documents, constrained and unconstrained")
    print(f"serving: {label}")

    out_dir = root / "data" / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_name or f"{corpus}_{split}"

    for constrained in (True, False):
        suffix = "constrained" if constrained else "unconstrained"
        # Written to the volume and streamed down: a full test split of
        # generated text is large enough to drop the gRPC stream after the GPU
        # work has already succeeded.
        remote_name = f"generations/{stem}_{suffix}.json"
        summary = generate.remote(
            model_path=model_path,
            prompts=prompts,
            adapters=adapter_list,
            json_schema=bounded_graph_schema() if constrained else None,
            out_path=f"{MODELS_DIR}/{remote_name}",
        )
        if "error" in summary:
            # Surface the remote failure instead of a confusing
            # FileNotFoundError for the output it never got to write.
            print(f"  {suffix}: FAILED -- {summary['error']}")
            print(summary.get("traceback", ""))
            raise SystemExit(1)

        path = out_dir / f"{stem}_{suffix}.json"
        with path.open("wb") as fh:
            for chunk in MODELS_VOLUME.read_file(remote_name):
                fh.write(chunk)

        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["runs"][0]
        truncated = sum(1 for r in run["results"] if r["finish_reason"] == "length")
        parsed = 0
        for r in run["results"]:
            try:
                json.loads(r["text"])
                parsed += 1
            except Exception:  # counting parse failures, not handling them
                pass
        total = len(run["results"])
        print(
            f"  {suffix:<14} {run['generate_seconds']:>7.1f}s  "
            f"{run['throughput_docs_per_second']:>5.2f} docs/s  "
            f"parsed {parsed}/{total}  truncated {truncated}"
        )
        print(f"    -> {path}")
