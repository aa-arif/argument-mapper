"""Inspect saved LoRA adapter weights.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_adapter.py

All nine adapters produced byte-identical generations. vLLM does accept the
`lora_request` parameter and did JIT its LoRA kernels, so the next suspect is
the weights themselves: a LoRA whose B matrix never moved off its zero
initialisation is a no-op no matter how correctly it is loaded.

Reports per-adapter tensor counts and norms, and whether two adapters differ
from each other.
"""

from __future__ import annotations

import json

from modal_common import MODELS_DIR, MODELS_VOLUME, TRAIN_IMAGE, app

ADAPTERS = [
    f"{MODELS_DIR}/adapters/Qwen__Qwen3.5-2B/seed0/epoch1",
    f"{MODELS_DIR}/adapters/Qwen__Qwen3.5-2B/seed0/epoch3",
    f"{MODELS_DIR}/adapters/Qwen__Qwen3.5-2B/seed2/epoch3",
]


@app.function(image=TRAIN_IMAGE, volumes={MODELS_DIR: MODELS_VOLUME}, timeout=60 * 15)
def probe() -> dict[str, object]:
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    out: dict[str, object] = {}
    loaded: dict[str, dict[str, torch.Tensor]] = {}

    for path in ADAPTERS:
        f = Path(path) / "adapter_model.safetensors"
        if not f.exists():
            out[path] = {"error": "missing"}
            continue
        tensors = load_file(str(f))
        loaded[path] = tensors

        a_norms = [t.float().norm().item() for k, t in tensors.items() if "lora_A" in k]
        b_norms = [t.float().norm().item() for k, t in tensors.items() if "lora_B" in k]
        out[path] = {
            "file_mb": round(f.stat().st_size / 1e6, 2),
            "tensors": len(tensors),
            "lora_A_count": len(a_norms),
            "lora_B_count": len(b_norms),
            # B starts at zero by design; if it is still zero, the adapter is
            # mathematically a no-op regardless of how it is served.
            "lora_B_all_zero": all(n == 0.0 for n in b_norms) if b_norms else None,
            "lora_B_mean_norm": round(sum(b_norms) / len(b_norms), 6) if b_norms else None,
            "lora_A_mean_norm": round(sum(a_norms) / len(a_norms), 6) if a_norms else None,
            "sample_keys": sorted(tensors)[:3],
        }

    # Do two different checkpoints actually hold different weights?
    keys = list(loaded)
    if len(keys) >= 2:
        a, b = loaded[keys[0]], loaded[keys[-1]]
        shared = sorted(set(a) & set(b))
        diffs = [float((a[k].float() - b[k].float()).abs().max()) for k in shared[:50]]
        out["first_vs_last"] = {
            "shared_tensors": len(shared),
            "max_abs_difference": round(max(diffs), 8) if diffs else None,
            "identical": (max(diffs) == 0.0) if diffs else None,
        }

    return json.loads(json.dumps(out, default=str))


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), indent=2))
