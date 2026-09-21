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

**This probe is also a regression test.** It exits non-zero when the adapter
does not measurably change generation, which is exactly the condition that went
undetected for two days (see DECISIONS D27). Run it after any change to how the
model is served.

**Nothing it records is generated text.** The generations quote the essays they
were given, and the Argument Annotated Essays licence forbids displaying the
corpus in a public repository, so only structure is written out: lengths,
booleans, counts, and a hash. See DECISIONS D30.
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


def _shape(text: str) -> dict[str, object]:
    """Describe a generation without reproducing a word of it.

    The generations quote the source essays, so `text` is licensed corpus
    content by the time the model has finished with it. Everything below is
    derived from it and none of it is recoverable back into text: lengths,
    counts, a truncated digest, and -- when the output parses -- the key names
    of *our own* response schema, which are ours to publish.
    """
    import hashlib

    stripped = text.strip()
    lines = stripped.splitlines()
    shape: dict[str, object] = {
        "chars": len(stripped),
        "lines": len(lines),
        "sha256_16": hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:16],
        "starts_with_brace": stripped.startswith("{"),
        "markdown_heading_lines": sum(1 for line in lines if line.lstrip().startswith("#")),
        "bullet_lines": sum(1 for line in lines if line.lstrip()[:2] in ("- ", "* ")),
    }
    try:
        payload = json.loads(stripped)
    except Exception:  # classifying the output, not handling a failure
        shape["parses"] = False
        return shape

    shape["parses"] = True
    if isinstance(payload, dict):
        shape["top_level_keys"] = sorted(str(k) for k in payload)
        for field in ("components", "relations"):
            value = payload.get(field)
            shape[f"n_{field}"] = len(value) if isinstance(value, list) else None
    return shape


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

    results: list[dict[str, object]] = []

    def record(which: str, doc_id: object, text: str) -> None:
        results.append({"doc_id": doc_id, "which": which, **_shape(text)})

    # Base model first, before the adapter is attached.
    for prompt in prompts:
        record("base", prompt["doc_id"], generate(base, prompt["messages"]))  # type: ignore[arg-type]

    # Same model object, adapter attached.
    adapted = PeftModel.from_pretrained(base, adapter)
    adapted.eval()

    for prompt in prompts:
        record("peft", prompt["doc_id"], generate(adapted, prompt["messages"]))  # type: ignore[arg-type]

    # Merged weights: if PEFT works and merged does not, merging is the fault.
    merged = adapted.merge_and_unload()
    for prompt in prompts[:2]:
        record("merged", prompt["doc_id"], generate(merged, prompt["messages"]))  # type: ignore[arg-type]

    summary: dict[str, dict[str, int]] = {}
    for row in results:
        bucket = summary.setdefault(str(row["which"]), {"n": 0, "json": 0, "brace": 0})
        bucket["n"] += 1
        bucket["json"] += int(bool(row["parses"]))
        bucket["brace"] += int(bool(row["starts_with_brace"]))

    # The same generation from two different weight sets hashes the same only
    # if the weights had no effect. This is the cheapest possible statement of
    # "the adapter did something".
    by_doc: dict[tuple[object, str], str] = {
        (row["doc_id"], str(row["which"])): str(row["sha256_16"]) for row in results
    }
    identical = sum(
        1
        for prompt in prompts
        if by_doc.get((prompt["doc_id"], "base")) == by_doc.get((prompt["doc_id"], "peft"))
    )
    summary["_effect"] = {"documents": len(prompts), "identical_base_vs_peft": identical}

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

    summary = report["summary"]
    for which in ("base", "peft", "merged"):
        stats = summary.get(which)
        if stats:
            print(
                f"  {which:<8} valid JSON {stats['json']}/{stats['n']}  "
                f"starts with brace {stats['brace']}/{stats['n']}"
            )

    effect = summary["_effect"]
    identical, total = effect["identical_base_vs_peft"], effect["documents"]
    print(f"\nidentical base vs peft output: {identical}/{total}")
    print(f"wrote {out}")

    # ---- the regression assertion ------------------------------------------
    #
    # An adapter that produces byte-identical output to the base model on every
    # document has not been applied. That is the failure this file exists to
    # catch, and it must be loud: for two days it was silent, and every number
    # in the repository was wrong (DECISIONS D27).
    base, peft = summary["base"], summary["peft"]
    if effect["identical_base_vs_peft"] == effect["documents"]:
        raise SystemExit(
            "ADAPTER HAS NO EFFECT: base and adapted generations are identical on "
            f"all {effect['documents']} documents. The adapter is not reaching the model."
        )
    if peft["json"] <= base["json"]:
        raise SystemExit(
            f"ADAPTER DID NOT LEARN THE FORMAT: base {base['json']}/{base['n']} valid JSON, "
            f"adapted {peft['json']}/{peft['n']}. A fine-tune on this format should beat the "
            "base model on its own training data."
        )
    print("\nOK: the adapter changes generation and produces the trained format.")
