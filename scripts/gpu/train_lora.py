"""LoRA fine-tuning for argument extraction, on Modal.

    PYTHONIOENCODING=utf-8 uv run python -m argmap.cli.train --seeds 0 1 2

Driven from `argmap.cli.train`, which builds the training data locally and
calls in here. Three things are deliberate:

* **Three seeds.** 258 training documents is a small corpus and a single run's
  score is largely luck. Reporting mean and standard deviation across seeds
  says how much of a difference is real.
* **Checkpoints are selected on validation, never test.** Training saves one
  adapter per epoch and scores none of them; selection happens afterwards from
  validation generations, so the test set is touched once, at the end.
* **Adapters stay on a Modal volume.** A model fine-tuned on the Argument
  Annotated Essays corpus is plausibly derivative of it, and that corpus may
  not be redistributed (DECISIONS.md D2).

LoRA target modules are discovered by walking the loaded model rather than
hardcoded. `Qwen3.5-2B` is a vision-language checkpoint whose vision tower
carries the same projection names as the language tower; adapting the vision
encoder would spend capacity on pixels this task never sees.
"""

from __future__ import annotations

import json

from modal_common import (
    DATA_DIR,
    DATA_VOLUME,
    GPU_TRAIN,
    MODELS_DIR,
    MODELS_VOLUME,
    TRAIN_IMAGE,
    app,
)

#: Projection names a LoRA targets on a Qwen-family decoder.
TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Substrings identifying a vision tower, which must not be adapted.
VISION_MARKERS = ("visual", "vision", "image", "patch_embed")

DEFAULT_HP = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 8,
    # Essay ~500 tokens + system ~450 + answer ~1500, with headroom.
    "max_seq_length": 3072,
}


def _language_target_modules(model: object) -> list[str]:
    """Every LoRA-target projection outside the vision tower."""
    names: list[str] = []
    for name, _ in model.named_modules():  # type: ignore[attr-defined]
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_LEAVES and not any(m in name.lower() for m in VISION_MARKERS):
            names.append(name)
    return names


@app.function(
    gpu=GPU_TRAIN,
    image=TRAIN_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME, DATA_DIR: DATA_VOLUME},
    timeout=60 * 90,
)
def train(
    train_jsonl: str,
    val_jsonl: str,
    base_model: str,
    seed: int,
    hp: dict[str, object] | None = None,
    run_tag: str = "e3",
) -> dict[str, object]:
    """Fine-tune one seed, saving one adapter per epoch."""
    import time
    from pathlib import Path

    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    settings = {**DEFAULT_HP, **(hp or {})}
    transformers.set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.use_cache = False

    targets = _language_target_modules(model)
    if not targets:
        msg = f"no LoRA target modules found on {base_model}"
        raise RuntimeError(msg)

    peft_config = LoraConfig(
        r=int(settings["lora_r"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def _load(blob: str) -> Dataset:
        rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
        return Dataset.from_list([{"messages": r["messages"]} for r in rows])

    train_ds, val_ds = _load(train_jsonl), _load(val_jsonl)

    # Tagged so a re-run with different hyperparameters cannot overwrite the
    # adapters an earlier run's numbers were measured from.
    out_root = f"{MODELS_DIR}/adapters/{base_model.replace('/', '__')}/{run_tag}/seed{seed}"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    per_epoch: list[dict[str, object]] = []

    class SaveEachEpoch(TrainerCallback):
        """Persist an adapter per epoch so selection can happen later.

        Selection is not done here on purpose: choosing a checkpoint needs
        generated output scored with the project's own metrics, and doing that
        inside the training loop would couple the two and tempt a peek at test.
        """

        def on_epoch_end(self, args, state, control, **kwargs) -> object:  # noqa: ANN001, ANN003
            epoch = round(state.epoch or 0)
            path = f"{out_root}/epoch{epoch}"
            kwargs["model"].save_pretrained(path)
            tokenizer.save_pretrained(path)
            # Only the path is recorded here. `on_epoch_end` fires *before* the
            # epoch's evaluation is written to log_history, so reading the last
            # eval_loss at this point yields the previous epoch's value -- or
            # None on the first epoch. Losses are attached after training from
            # the completed log instead.
            per_epoch.append({"epoch": epoch, "path": path})
            MODELS_VOLUME.commit()
            return control

    config = SFTConfig(
        output_dir=f"/tmp/sft-seed{seed}",
        num_train_epochs=int(settings["epochs"]),
        per_device_train_batch_size=int(settings["batch_size"]),
        gradient_accumulation_steps=int(settings["grad_accum"]),
        learning_rate=float(settings["learning_rate"]),
        lr_scheduler_type="cosine",
        # TRL's SFTConfig exposes warmup_steps, not warmup_ratio. 258 examples
        # at an effective batch of 8 is ~32 steps/epoch, ~97 total, so 5 steps
        # is the ~5% warmup a ratio would have given.
        warmup_steps=5,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",  # adapters are saved by the callback instead
        bf16=True,
        max_length=int(settings["max_seq_length"]),
        seed=seed,
        report_to=[],
        # Train on the answer only. Without this the model spends capacity
        # learning to reproduce the system prompt and the essay, neither of
        # which it is ever asked to generate.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[SaveEachEpoch()],
    )

    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started

    # Attach each epoch's metrics from the completed log, keyed by the epoch
    # the trainer itself recorded, so nothing is off by one.
    eval_by_epoch = {round(h["epoch"]): h for h in trainer.state.log_history if "eval_loss" in h}
    train_by_epoch = {round(h["epoch"]): h for h in trainer.state.log_history if "loss" in h}
    for entry in per_epoch:
        metrics = eval_by_epoch.get(entry["epoch"], {})
        entry["eval_loss"] = metrics.get("eval_loss")
        entry["eval_mean_token_accuracy"] = metrics.get("eval_mean_token_accuracy")
        entry["train_loss"] = train_by_epoch.get(entry["epoch"], {}).get("loss")

    return {
        "base_model": base_model,
        "run_tag": run_tag,
        "seed": seed,
        "hyperparameters": settings,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / total, 6) if total else 0.0,
        "lora_target_module_count": len(targets),
        "train_documents": len(train_ds),
        "val_documents": len(val_ds),
        "train_seconds": round(elapsed, 1),
        "gpu": GPU_TRAIN,
        "checkpoints": per_epoch,
    }


@app.function(
    gpu=GPU_TRAIN,
    image=TRAIN_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 60,
)
def merge_adapter(base_model: str, adapter_path: str, out_name: str) -> str:
    """Merge a LoRA adapter into the base weights for serving.

    vLLM can serve adapters directly, but merging removes a per-request
    indirection and makes the throughput numbers in milestone 6 describe a
    plain model rather than a model plus adapter machinery.
    """
    import torch
    import transformers
    from peft import PeftModel

    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    out_dir = f"{MODELS_DIR}/merged/{out_name}"
    merged.save_pretrained(out_dir, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    MODELS_VOLUME.commit()
    return out_dir


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-2B",
    seeds: str = "0,1,2",
    epochs: int = 10,
    learning_rate: float = 2e-4,
    run_tag: str = "e10",
) -> None:
    """Train one adapter per seed and record what each run produced.

    Runs locally, reads the exported SFT files, and ships their contents to the
    remote function. Seeds run concurrently -- they are independent, and three
    sequential runs would triple the wall clock for no benefit.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    train_blob = (root / "data" / "sft" / "train.jsonl").read_text(encoding="utf-8")
    val_blob = (root / "data" / "sft" / "val.jsonl").read_text(encoding="utf-8")

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    # 3 epochs over 258 examples is only ~99 optimizer steps, which was not
    # enough to move the model off the base checkpoint's prose-formatting
    # prior: it produced correct content in Markdown, and under constrained
    # decoding emitted generic component ids and looped. More steps and a
    # higher learning rate target exactly that. See DECISIONS D23.
    hp = {"epochs": epochs, "learning_rate": learning_rate}

    results = list(
        train.starmap([(train_blob, val_blob, base_model, seed, hp, run_tag) for seed in seed_list])
    )

    # Tagged: an untagged name let the 10-epoch run overwrite the 3-epoch
    # run's record, which is the same clobbering already fixed for adapters
    # and baselines.
    out = root / "results" / "training" / f"runs_{run_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2, default=str))
    print("\n--- summary ---")
    for r in results:
        losses = [c.get("eval_loss") for c in r.get("checkpoints", [])]
        print(
            f"  seed {r['seed']}: {r['train_seconds']}s, "
            f"{r['trainable_parameters'] / 1e6:.1f}M trainable "
            f"({r['trainable_fraction'] * 100:.2f}%), eval_loss per epoch = {losses}"
        )
