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

import pathlib

from generate import generate
from modal_common import MODELS_DIR, MODELS_VOLUME, app

BASE_MODEL = "Qwen/Qwen3.5-2B"
SEEDS = (0, 1, 2)
EPOCHS = (1, 2, 3)


def _adapter_paths(base_model: str, run_tag: str, epochs: tuple[int, ...]) -> list[str]:
    stem = f"{MODELS_DIR}/adapters/{base_model.replace('/', '__')}/{run_tag}"
    return [f"{stem}/seed{seed}/epoch{epoch}" for seed in SEEDS for epoch in epochs]


@app.local_entrypoint()
def main(
    base_model: str = BASE_MODEL,
    split: str = "val",
    out_name: str = "val_checkpoints",
    constrained: bool = True,
    run_tag: str = "e10",
    epochs: str = "4,6,8,10",
    merged: str = "",
) -> None:
    """Generate validation predictions for every checkpoint.

    `constrained` defaults to true because it matches how the local route
    actually runs in production, and because the Claude baseline it is compared
    against uses the API's structured outputs. Evaluating one side with format
    enforcement and the other without would measure the enforcement, not the
    models. The unconstrained run is milestone 4's ablation, not the baseline.
    """
    from argmap.cli.run_baseline import load_split
    from argmap.prompts import bounded_graph_schema
    from argmap.train_data import to_messages

    root = pathlib.Path(__file__).resolve().parents[2]
    docs = load_split(root, "aae-v2", split)
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    # A merged checkpoint is served as a plain model: vLLM's dynamic LoRA
    # path silently produced base-model output for this architecture, so
    # merging is how the adapter actually reaches inference.
    if merged:
        model_path = merged if merged.startswith(MODELS_DIR) else f"{MODELS_DIR}/merged/{merged}"
        adapters: list[str] = []
    else:
        model_path = base_model
        epoch_list = tuple(int(e) for e in epochs.split(",") if e.strip())
        adapters = _adapter_paths(base_model, run_tag, epoch_list)
    mode = "constrained" if constrained else "unconstrained"
    label = merged or f"{len(adapters)} checkpoints"
    print(f"{len(prompts)} {split} documents x {label} ({mode})")

    # Written to the volume rather than returned. A dozen checkpoints of
    # generated text is several megabytes, which is enough to drop the gRPC
    # stream -- "StreamTerminatedError: Connection lost" -- *after* the GPU
    # work has already succeeded, wasting the whole run on transport.
    remote_name = f"generations/{out_name}.json"
    summary = generate.remote(
        model_path=model_path,
        prompts=prompts,
        adapters=adapters or None,
        json_schema=bounded_graph_schema() if constrained else None,
        out_path=f"{MODELS_DIR}/{remote_name}",
    )

    out = root / "data" / "generations" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        for chunk in MODELS_VOLUME.read_file(remote_name):
            fh.write(chunk)
    payload = summary

    print(f"\nengine load {payload['load_seconds']}s  ({summary['bytes'] / 1e6:.1f} MB downloaded)")
    for run in payload["runs"]:
        name = (run["adapter"] or "base").split("/")[-3:]
        print(
            f"  {'/'.join(name):<28} {run['generate_seconds']:>7.1f}s  "
            f"{run['throughput_docs_per_second']:>6.2f} docs/s"
        )
    print(f"\nwrote {out}")
