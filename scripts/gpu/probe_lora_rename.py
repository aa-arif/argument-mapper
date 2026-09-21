"""Prove the root cause: rename the adapter's modules and watch vLLM apply it.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_lora_rename.py

`probe_lora_modules.py` established the mismatch on paper. The adapter carries
LoRA weights for modules named `model.layers.N...`, because PEFT saw the model
through `AutoModelForCausalLM`. vLLM registers Qwen3.5 as
`Qwen3_5ForConditionalGeneration`, a subclass of the Qwen3-VL multimodal class,
whose weight mapper puts the language model under `language_model.model.`. The
two naming schemes never meet, and a LoRA vLLM cannot find a module for is
skipped in silence.

That is a hypothesis until it makes a prediction that can fail. This is the
prediction: **rewrite the adapter's keys to the names vLLM expects and the same
vLLM, the same engine flags, the same prompts should start emitting the trained
format.** If it does not, the diagnosis is wrong and merging is a workaround
for something else.

Nothing here is a quality measurement -- it runs on training documents, where
a working adapter cannot fail -- and nothing generated is recorded, only
whether it parsed (DECISIONS D30).
"""

from __future__ import annotations

import json

from modal_common import GPU_SERVE, MODELS_DIR, MODELS_VOLUME, SERVE_IMAGE, app

MODEL = "Qwen/Qwen3.5-2B"

#: What PEFT wrote, and what vLLM's multimodal wrapper expects instead.
PEFT_PREFIX = "base_model.model.model."
VLLM_PREFIX = "base_model.model.language_model.model."


@app.function(
    gpu=GPU_SERVE,
    image=SERVE_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 40,
)
def probe(adapter: str, renamed: str, prompts: list[dict[str, object]]) -> dict[str, object]:
    import traceback

    try:
        return _probe(adapter, renamed, prompts)
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }


def _rewrite(adapter: str, renamed: str) -> dict[str, object]:
    """Copy the adapter with every module path moved under `language_model.`."""
    import pathlib
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    src, dst = pathlib.Path(adapter), pathlib.Path(renamed)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("adapter_config.json", "README.md"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    tensors: dict[str, torch.Tensor] = load_file(str(src / "adapter_model.safetensors"))
    moved = 0
    out: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key.startswith(PEFT_PREFIX):
            out[VLLM_PREFIX + key[len(PEFT_PREFIX) :]] = value
            moved += 1
        else:
            out[key] = value
    save_file(out, str(dst / "adapter_model.safetensors"))
    return {"tensors": len(tensors), "renamed": moved, "path": str(dst)}


def _probe(adapter: str, renamed: str, prompts: list[dict[str, object]]) -> dict[str, object]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    report: dict[str, object] = {"adapter": adapter, "renamed_adapter": renamed}
    report["rewrite"] = _rewrite(adapter, renamed)

    llm = LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=5120,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=900)
    messages = [p["messages"] for p in prompts]

    def run(label: str, lora: LoRARequest | None) -> dict[str, object]:
        outputs = llm.chat(messages, sampling, lora_request=lora)  # type: ignore[arg-type]
        parsed = 0
        for output in outputs:
            text = output.outputs[0].text.strip()
            try:
                json.loads(text)
            except Exception:  # counting, not handling
                continue
            parsed += 1
        return {"label": label, "documents": len(outputs), "valid_json": parsed}

    report["runs"] = [
        # The control: no adapter at all.
        run("base", None),
        # What the project actually ran for two days.
        run("original_names", LoRARequest("orig", 1, adapter)),
        # The same weights, under the names vLLM looks for.
        run("renamed", LoRARequest("renamed", 2, renamed)),
    ]
    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main(
    adapter: str = "Qwen__Qwen3.5-2B/e10/seed0/epoch10",
    limit: int = 6,
) -> None:
    import pathlib

    from argmap.cli.run_baseline import load_split
    from argmap.train_data import to_messages

    adapter_path = adapter if adapter.startswith(MODELS_DIR) else f"{MODELS_DIR}/adapters/{adapter}"
    renamed_path = f"{MODELS_DIR}/adapters-renamed/{adapter.replace('/', '__')}"

    root = pathlib.Path(__file__).resolve().parents[2]
    docs = load_split(root, "aae-v2", "train")[:limit]
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    report = probe.remote(adapter=adapter_path, renamed=renamed_path, prompts=prompts)
    if "error" in report:
        print(report["error"])
        print(report.get("traceback", ""))
        raise SystemExit(1)

    out = root / "results" / "diagnostics" / "lora_rename.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"rewrite: {report['rewrite']}\n")
    for run in report["runs"]:
        print(f"  {run['label']:<16} valid JSON {run['valid_json']}/{run['documents']}")
    print(f"\nwrote {out}")

    runs = {r["label"]: r for r in report["runs"]}
    if runs["original_names"]["valid_json"] != runs["base"]["valid_json"]:
        print(
            "\nINCONCLUSIVE: the adapter under its original names already differs "
            "from the base model, so the naming mismatch is not the whole story."
        )
    elif runs["renamed"]["valid_json"] > runs["base"]["valid_json"]:
        print(
            "\nCONFIRMED: identical weights, identical engine, identical prompts. "
            "Only the module names changed, and the adapter started working."
        )
    else:
        print("\nREFUTED: renaming changed nothing. The cause is elsewhere.")
