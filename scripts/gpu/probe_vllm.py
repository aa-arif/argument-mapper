"""Check vLLM's structured-output API and architecture support.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_vllm.py

Two things have to be true before milestone 4 is worth starting, and both are
cheaper to check on CPU than to discover after a GPU has spun up and pulled
weights:

1. vLLM knows the `qwen3_5` architecture. It is a hybrid linear-attention
   model (DECISIONS D17), not a stock transformer, so support is not a given.
2. The name of the structured-output parameter class. vLLM renamed
   `GuidedDecodingParams` at some point; guessing wrong fails at request time,
   after the engine has loaded.
"""

from __future__ import annotations

import json

from modal_common import SERVE_IMAGE, app


@app.function(image=SERVE_IMAGE, timeout=60 * 15)
def probe() -> dict[str, object]:
    import vllm

    report: dict[str, object] = {"vllm_version": vllm.__version__}

    # Which structured-output parameter class exists in this release?
    candidates = ("StructuredOutputsParams", "GuidedDecodingParams")
    found: dict[str, bool] = {}
    for name in candidates:
        try:
            import vllm.sampling_params as sp

            found[name] = hasattr(sp, name)
        except Exception as exc:
            found[name] = False
            report["sampling_params_import_error"] = f"{type(exc).__name__}: {exc}"
    report["structured_output_classes"] = found

    # Which SamplingParams field carries it? SamplingParams is a msgspec
    # Struct in current vLLM, not a dataclass, so `dataclasses.fields` raises.
    try:
        from vllm import SamplingParams

        names = sorted(getattr(SamplingParams, "__struct_fields__", ()) or ())
        if not names:
            names = sorted(k for k in vars(SamplingParams) if not k.startswith("_"))
        report["sampling_params_field_count"] = len(names)
        report["sampling_params_fields"] = [
            f for f in names if "structur" in f or "guided" in f or "logprob" in f
        ]
    except Exception as exc:
        report["sampling_params_error"] = f"{type(exc).__name__}: {exc}"

    # Is the architecture registered?
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        archs = sorted(ModelRegistry.get_supported_archs())
        report["qwen_architectures"] = [a for a in archs if "Qwen3" in a]
        report["supports_qwen3_5"] = any("Qwen3_5" in a for a in archs)
        report["architecture_count"] = len(archs)
    except Exception as exc:
        report["registry_error"] = f"{type(exc).__name__}: {exc}"

    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main() -> None:
    report = probe.remote()
    print(json.dumps(report, indent=2))
