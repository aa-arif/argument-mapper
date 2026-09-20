"""Check how vLLM's LLM.chat accepts a LoRA request.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_chat_sig.py

All nine adapters produced byte-identical output, which means either the
adapters are inert or they were never applied. Before re-training anything,
confirm that `lora_request` is a parameter `chat` actually honours rather than
one it silently swallows through **kwargs.
"""

from __future__ import annotations

import json

from modal_common import SERVE_IMAGE, app


@app.function(image=SERVE_IMAGE, timeout=60 * 10)
def probe() -> dict[str, object]:
    import inspect

    from vllm import LLM

    report: dict[str, object] = {}
    for name in ("chat", "generate"):
        fn = getattr(LLM, name, None)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        report[f"{name}_params"] = [str(p) for p in sig.parameters.values()]
        report[f"{name}_has_lora_request"] = "lora_request" in sig.parameters
        report[f"{name}_has_var_kwargs"] = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main() -> None:
    r = probe.remote()
    for key in sorted(r):
        if key.endswith("_params"):
            print(f"{key}:")
            for p in r[key]:
                print(f"    {p}")
        else:
            print(f"{key}: {r[key]}")
