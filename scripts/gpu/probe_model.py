"""Check whether the candidate base models are actually usable before training.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_model.py

`Qwen/Qwen3.5-2B` is a *vision*-language checkpoint
(`Qwen3_5ForConditionalGeneration`), which is the single largest risk in
milestone 3: text-only supervised fine-tuning has to reach the language tower,
and vLLM has to know the `qwen3_5` architecture to serve the merged result.
The fallback is `Qwen/Qwen3-1.7B`, a plain `Qwen3ForCausalLM`.

This answers the question for about five cents instead of discovering it two
hours into a training run. It reports, per candidate: whether the config and
tokenizer load, whether the weights load on GPU, which modules a LoRA would
target, and whether a text-only generation round-trips.
"""

from __future__ import annotations

import json

from modal_common import GPU_TRAIN, MODELS_DIR, MODELS_VOLUME, TRAIN_IMAGE, app

CANDIDATES = [
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3-1.7B",
]

#: The projection names a LoRA normally targets on a Qwen-family decoder.
WANTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@app.function(
    gpu=GPU_TRAIN,
    image=TRAIN_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 45,
)
def probe(model_id: str) -> dict[str, object]:
    """Load one candidate and report what works, as plain JSON-safe values.

    Never raises. A probe whose job is to discover what breaks is useless if
    breaking takes the whole run down and buries the message in Modal's
    remote-exception wrapper.
    """
    import traceback

    try:
        report = _probe(model_id)
    except BaseException as exc:
        report = {
            "model_id": model_id,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2500:],
        }

    # Round-trip through JSON before returning. Values that look like plain
    # types often are not -- `torch.__version__` is a `str` *subclass* defined
    # by torch -- and unpickling one locally needs torch installed, which is
    # the whole reason this runs remotely. The failure surfaces as an opaque
    # DeserializationError long after the GPU work succeeded.
    return json.loads(json.dumps(report, default=str))


def _probe(model_id: str) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    report: dict[str, object] = {
        "model_id": model_id,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        report["architectures"] = list(getattr(config, "architectures", []) or [])
        report["model_type"] = getattr(config, "model_type", None)
        report["config_ok"] = True
    except Exception as exc:
        report["config_ok"] = False
        report["config_error"] = f"{type(exc).__name__}: {exc}"
        return report

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        report["tokenizer_ok"] = True
        report["has_chat_template"] = tokenizer.chat_template is not None
    except Exception as exc:
        report["tokenizer_ok"] = False
        report["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        return report

    # Try the plain causal-LM entry point first. If the checkpoint is a
    # multimodal wrapper this is exactly where it fails, which is the thing
    # worth knowing.
    model = None
    load_errors: dict[str, str] = {}
    for loader_name in ("AutoModelForCausalLM", "AutoModel"):
        try:
            loader = getattr(transformers, loader_name)
            model = loader.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="cuda",
            )
            report["loaded_with"] = loader_name
            break
        except Exception as exc:
            load_errors[loader_name] = f"{type(exc).__name__}: {exc}"[:600]
    report["load_errors"] = load_errors

    if model is None:
        report["weights_ok"] = False
        return report

    report["weights_ok"] = True
    report["parameters"] = sum(p.numel() for p in model.parameters())

    # Which of the usual LoRA targets exist, and where. On a VLM the vision
    # tower carries the same projection names, so record the full paths -- a
    # LoRA that accidentally adapts the vision encoder is wasted capacity.
    found: dict[str, int] = {}
    language_paths: list[str] = []
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in WANTED_TARGETS:
            found[leaf] = found.get(leaf, 0) + 1
            if "visual" not in name and "vision" not in name and len(language_paths) < 3:
                language_paths.append(name)
    report["lora_targets_found"] = found
    report["sample_language_module_paths"] = language_paths
    report["has_vision_tower"] = any(
        "visual" in n or "vision" in n for n, _ in model.named_modules()
    )

    # Text-only round trip. If a multimodal wrapper insists on pixel inputs,
    # this is where it says so.
    try:
        messages = [{"role": "user", "content": "Reply with the single word OK."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        report["text_generation_ok"] = True
        report["sample_output"] = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()[:80]
    except Exception as exc:
        report["text_generation_ok"] = False
        report["generation_error"] = f"{type(exc).__name__}: {exc}"

    return report


@app.local_entrypoint()
def main() -> None:
    results = list(probe.map(CANDIDATES))
    print(json.dumps(results, indent=2, default=str))

    print("\n--- verdict ---")
    for r in results:
        if r.get("fatal_error"):
            print(f"  {r['model_id']:<22} FATAL    {r['fatal_error']}")
            continue
        usable = r.get("weights_ok") and r.get("text_generation_ok")
        print(
            f"  {r['model_id']:<22} {'USABLE' if usable else 'BLOCKED':<8} "
            f"arch={r.get('architectures')} vision_tower={r.get('has_vision_tower')}"
        )
