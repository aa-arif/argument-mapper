"""Check that SFT loss is masked to the assistant turn.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_masking.py

The fine-tuned model still answers in Markdown instead of the JSON it was
trained on, while its content is right. Two explanations fit: too few
optimizer steps to shift the format, or a loss that was never concentrated on
the answer in the first place.

`completion_only_loss=True` is supposed to mask the prompt out of the labels.
If it silently did not, most of the gradient went into reproducing a system
prompt the model is never asked to generate, and no amount of extra epochs
fixes that. Runs on CPU.
"""

from __future__ import annotations

import json

from modal_common import TRAIN_IMAGE, app

SAMPLE = {
    "messages": [
        {"role": "system", "content": "You are an expert annotator. " * 40},
        {"role": "user", "content": "Identify components in: Cars pollute. So cycle."},
        {"role": "assistant", "content": '{"components":[{"id":"c1"}],"relations":[]}'},
    ]
}


@app.function(image=TRAIN_IMAGE, timeout=60 * 20)
def probe() -> dict[str, object]:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_list([SAMPLE] * 4)
    report: dict[str, object] = {}

    for completion_only in (True, False):
        config = SFTConfig(
            output_dir=f"/tmp/mask-{completion_only}",
            max_length=2048,
            completion_only_loss=completion_only,
            report_to=[],
            per_device_train_batch_size=1,
            # This probe only builds the dataset; it never steps the model,
            # so it runs on CPU where bf16 is unavailable.
            use_cpu=True,
            bf16=False,
        )
        # Build the dataset without a model: SFTTrainer prepares data in __init__.
        trainer = SFTTrainer(
            model=model_id,
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
        )
        example = trainer.train_dataset[0]
        labels = example.get("labels")
        input_ids = example.get("input_ids")
        if labels is None:
            report[f"completion_only={completion_only}"] = {
                "note": "no labels column; collator builds them",
                "columns": sorted(example.keys()),
                "input_length": len(input_ids) if input_ids else None,
                "completion_mask_sum": (
                    sum(example["completion_mask"]) if "completion_mask" in example else None
                ),
            }
            continue
        unmasked = sum(1 for x in labels if x != -100)
        report[f"completion_only={completion_only}"] = {
            "total_tokens": len(labels),
            "unmasked_tokens": unmasked,
            "unmasked_fraction": round(unmasked / len(labels), 4) if labels else None,
        }

    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), indent=2))
