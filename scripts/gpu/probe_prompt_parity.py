"""Compare the prompt seen during training with the one seen at inference.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_prompt_parity.py

The fine-tuned model produces a *different ad-hoc Markdown format on every
document* when generating unconstrained, which is base-model behaviour rather
than the compact JSON it was trained on for ten epochs. Either the adapter is
barely being applied, or the model is being shown a different prompt at
inference than it saw during training -- and a mismatch would explain it
completely, because the LoRA only fires on the context it was fitted to.

This renders both sides and diffs them:

* training: what TRL feeds the model, the full conversation with the assistant
  turn, which is what the labels are computed against
* inference: what `vLLM.chat` builds, the same messages minus the answer, with
  `add_generation_prompt=True`

It also reports whether the end-of-turn token is present in the training target
(a model that never sees EOS in its labels never learns to stop) and whether
the template injects a thinking block.

Runs on CPU.
"""

from __future__ import annotations

import json

from modal_common import TRAIN_IMAGE, app

MODEL = "Qwen/Qwen3.5-2B"

SYSTEM = "You are an expert annotator of argumentation structure."
USER = "Identify the argument components in: Cars pollute. So we should cycle."
ANSWER = '{"components":[{"id":"c1","type":"Claim","text":"Cars pollute."}],"relations":[]}'


@app.function(image=TRAIN_IMAGE, timeout=60 * 20)
def probe() -> dict[str, object]:
    import traceback

    try:
        return _probe()
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2500:],
        }


def _probe() -> dict[str, object]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    full = [*messages, {"role": "assistant", "content": ANSWER}]

    report: dict[str, object] = {
        "model": MODEL,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "template_mentions_think": "think" in (tok.chat_template or "").lower(),
    }

    # What training sees: the whole conversation, rendered as one string.
    training_text = tok.apply_chat_template(full, tokenize=False)
    # What inference sees: prompt only, with the generation prompt appended.
    inference_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    report["training_text"] = training_text
    report["inference_text"] = inference_text

    # The training text should start with the inference text: inference stops
    # exactly where the model is expected to begin writing. If it does not,
    # the model is being cued differently than it was taught.
    report["inference_is_prefix_of_training"] = training_text.startswith(inference_text)
    if not report["inference_is_prefix_of_training"]:
        # Find where they diverge, which is the actionable part.
        limit = min(len(training_text), len(inference_text))
        index = next(
            (i for i in range(limit) if training_text[i] != inference_text[i]),
            limit,
        )
        report["diverges_at_char"] = index
        report["training_from_divergence"] = training_text[index : index + 220]
        report["inference_from_divergence"] = inference_text[index : index + 220]

    # What does the model have to generate, given the inference prompt?
    if report["inference_is_prefix_of_training"]:
        report["expected_completion"] = training_text[len(inference_text) :]
        report["completion_ends_with_eos"] = training_text.endswith(tok.eos_token or "")

    # Does the *tokenised* target include EOS? A completion-only loss that
    # masks EOS teaches the model never to stop.
    # transformers 5 returns an Encoding here, not a list of ids, so tokenise
    # the rendered string instead of unpacking whatever the template returns.
    full_ids = list(tok(training_text, add_special_tokens=False).input_ids)
    report["training_ids_end_with_eos"] = bool(full_ids) and full_ids[-1] == tok.eos_token_id
    report["last_training_ids"] = full_ids[-6:]
    report["last_training_tokens"] = tok.convert_ids_to_tokens(full_ids[-6:]) if full_ids else []

    # Some Qwen templates take an enable_thinking flag; if the default differs
    # between the two call sites the model is cued into a mode it never saw.
    for flag in (True, False):
        try:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=flag
            )
            report[f"generation_prompt_enable_thinking_{flag}"] = rendered[-240:]
        except Exception as exc:
            report[f"generation_prompt_enable_thinking_{flag}"] = f"unsupported: {exc}"

    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main() -> None:
    r = probe.remote()
    if "error" in r:
        print(r["error"])
        print(r.get("traceback", ""))
        raise SystemExit(1)

    print(f"model: {r['model']}  eos={r['eos_token']!r} (id {r['eos_token_id']})")
    print(f"template mentions 'think': {r['template_mentions_think']}")
    print(f"training ids end with EOS: {r['training_ids_end_with_eos']}")
    print(f"last training tokens: {r['last_training_tokens']}")
    print()
    print(f"inference prompt is a prefix of training text: {r['inference_is_prefix_of_training']}")
    if not r["inference_is_prefix_of_training"]:
        print(f"  DIVERGES at char {r['diverges_at_char']}")
        print(f"  training  : {r['training_from_divergence']!r}")
        print(f"  inference : {r['inference_from_divergence']!r}")
    else:
        print(f"  model must generate: {r['expected_completion']!r}")
    print()
    print("--- inference prompt (tail) ---")
    print(repr(str(r["inference_text"])[-400:]))
    print()
    print("--- enable_thinking=True (tail) ---")
    print(repr(str(r["generation_prompt_enable_thinking_True"])))
    print("--- enable_thinking=False (tail) ---")
    print(repr(str(r["generation_prompt_enable_thinking_False"])))
