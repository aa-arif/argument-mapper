"""Does the adapter actually change generation, and can it recall its training data?

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_adapter_effect.py

Prompt parity between training and inference has been ruled out, so the
question is narrower: is the LoRA having an effect at generation time at all?

Two discriminators, on **training** documents the adapter saw ten times:

1. **base vs adapter under transformers+PEFT.** If a ten-epoch fine-tune cannot
   reproduce the format of its own training data through the library that
   produced it, the problem is in training.
2. **transformers+PEFT vs vLLM, same adapter, same prompt.** If PEFT emits JSON
   and vLLM does not, the problem is in how vLLM loads or applies the adapter,
   not in the weights.

Using training documents is deliberate. This is not a quality measurement --
it is a test of whether the adapter fires at all, and memorised data is the
most favourable possible case. A model that fails here fails everywhere.
"""

from __future__ import annotations

import json

from modal_common import GPU_TRAIN, MODELS_DIR, MODELS_VOLUME, TRAIN_IMAGE, app

MODEL = "Qwen/Qwen3.5-2B"


@app.function(
    gpu=GPU_TRAIN,
    image=TRAIN_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 40,
)
def probe(adapter: str, prompts: list[dict[str, object]]) -> dict[str, object]:
    import traceback

    try:
        return _probe(adapter, prompts)
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
        }


def _probe(adapter: str, prompts: list[dict[str, object]]) -> dict[str, object]:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    )

    def generate(model: object, messages: list[dict[str, str]]) -> str:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=900,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def looks_like_json(text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith("{"):
            return False
        try:
            json.loads(stripped)
        except Exception:
            return False
        return True

    results: list[dict[str, object]] = []

    # Base model first, before the adapter is attached.
    for prompt in prompts:
        text = generate(base, prompt["messages"])  # type: ignore[arg-type]
        results.append(
            {
                "doc_id": prompt["doc_id"],
                "which": "base",
                "starts_with_brace": text.strip().startswith("{"),
                "parses": looks_like_json(text),
                "head": text[:180],
            }
        )

    # Same model object, adapter attached.
    adapted = PeftModel.from_pretrained(base, adapter)
    adapted.eval()

    for prompt in prompts:
        text = generate(adapted, prompt["messages"])  # type: ignore[arg-type]
        results.append(
            {
                "doc_id": prompt["doc_id"],
                "which": "peft",
                "starts_with_brace": text.strip().startswith("{"),
                "parses": looks_like_json(text),
                "head": text[:180],
            }
        )

    # Merged weights: if PEFT works and merged does not, merging is the fault.
    merged = adapted.merge_and_unload()
    for prompt in prompts[:2]:
        text = generate(merged, prompt["messages"])  # type: ignore[arg-type]
        results.append(
            {
                "doc_id": prompt["doc_id"],
                "which": "merged",
                "starts_with_brace": text.strip().startswith("{"),
                "parses": looks_like_json(text),
                "head": text[:180],
            }
        )

    summary: dict[str, dict[str, int]] = {}
    for row in results:
        which = str(row["which"])
        bucket = summary.setdefault(which, {"n": 0, "json": 0, "brace": 0})
        bucket["n"] += 1
        bucket["json"] += int(bool(row["parses"]))
        bucket["brace"] += int(bool(row["starts_with_brace"]))

    return json.loads(
        json.dumps(
            {"adapter": adapter, "summary": summary, "results": results},
            default=str,
        )
    )


@app.local_entrypoint()
def main(
    adapter: str = "Qwen__Qwen3.5-2B/e10/seed0/epoch10",
    limit: int = 6,
) -> None:
    import pathlib

    from argmap.cli.run_baseline import load_split
    from argmap.train_data import to_messages

    adapter_path = adapter if adapter.startswith(MODELS_DIR) else f"{MODELS_DIR}/adapters/{adapter}"

    root = pathlib.Path(__file__).resolve().parents[2]
    # TRAINING documents: the adapter saw these ten times. If it cannot
    # reproduce the format here, it cannot anywhere.
    docs = load_split(root, "aae-v2", "train")[:limit]
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    print(f"adapter: {adapter_path}")
    print(f"{len(prompts)} TRAINING documents (memorised; most favourable case)\n")

    report = probe.remote(adapter=adapter_path, prompts=prompts)
    if "error" in report:
        print(report["error"])
        print(report.get("traceback", ""))
        raise SystemExit(1)

    out = root / "results" / "diagnostics" / "adapter_effect.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for which, stats in report["summary"].items():
        print(
            f"  {which:<8} valid JSON {stats['json']}/{stats['n']}  "
            f"starts with brace {stats['brace']}/{stats['n']}"
        )

    print("\nsamples:")
    for row in report["results"]:
        print(f"  [{row['which']:<6}] {row['doc_id']}: {row['head'][:110]!r}")
    print(f"\nwrote {out}")
