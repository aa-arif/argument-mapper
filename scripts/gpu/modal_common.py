"""Shared Modal app, images and volumes for every GPU step.

There is no local GPU on the development machine, so milestones 3 through 6 all
run on Modal. Everything GPU-shaped lives under `scripts/gpu/` so the split
between what runs locally and what needs hardware is visible from the tree.

Run these with `PYTHONIOENCODING=utf-8` on Windows -- the Modal CLI prints a
check mark that cp1252 cannot encode and dies before doing any work. See
DECISIONS.md D7.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_model.py
"""

from __future__ import annotations

import modal

APP_NAME = "argmap"

#: A10G: 24 GB, roughly $1.10/hour. Ample for a 2B LoRA and for serving the
#: merged model with vLLM, and the cheapest card that fits both comfortably.
GPU_TRAIN = "A10G"
GPU_SERVE = "A10G"

#: Weights and adapters persist between runs rather than being re-downloaded.
#: They also stay off the developer machine and out of git: a model fine-tuned
#: on the Argument Annotated Essays corpus is plausibly derivative of it, and
#: that corpus may not be redistributed (DECISIONS.md D2).
MODELS_VOLUME = modal.Volume.from_name("argmap-models", create_if_missing=True)
DATA_VOLUME = modal.Volume.from_name("argmap-data", create_if_missing=True)

MODELS_DIR = "/models"
DATA_DIR = "/data"

#: Hugging Face cache lives on the volume so a cold container does not re-pull
#: several gigabytes of weights on every run.
HF_CACHE = f"{MODELS_DIR}/hf-cache"

_COMMON_ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

#: Probing and training. Versions are pinned so a run six months from now
#: reproduces rather than silently picking up a breaking release. Every pin
#: below was resolved against PyPI rather than guessed -- an invented version
#: fails the image build several minutes in, with the error buried in pip's
#: full list of available releases.
TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "peft==0.21.0",
        "trl==1.13.0",
        "datasets==5.0.1",
        "bitsandbytes==0.50.2",
        "hf-transfer==0.1.9",
        "sentencepiece==0.2.2",
        "pydantic==2.13.5",
    )
    .env(_COMMON_ENV)
    # Modal uploads only the entrypoint file. Without this, every script
    # importing this module dies remotely with ModuleNotFoundError, which
    # surfaces locally as an opaque RemoteError.
    .add_local_python_source("modal_common")
)

#: Serving with JSON-schema constrained decoding (milestones 4 and 6).
SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.29.0",
        "hf-transfer==0.1.9",
        "pydantic==2.13.5",
    )
    .env(_COMMON_ENV)
    .add_local_python_source("modal_common")
)

app = modal.App(APP_NAME)
