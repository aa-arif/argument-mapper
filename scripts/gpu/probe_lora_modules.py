"""Why did vLLM accept the adapter and serve base weights?

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_lora_modules.py

DECISIONS D27 established *that* vLLM applied nothing while reporting success,
and merging the adapter routed around it. This asks *why*, because "we merged
and it went away" is a workaround, not a diagnosis, and the next person to hit
it deserves the actual cause.

The hypothesis to test first is a name mismatch. PEFT writes LoRA weights under
the module paths it found in the transformers model; vLLM re-implements the
architecture and registers its own set of LoRA-capable modules. If the two sets
do not intersect, every layer vLLM looks for is absent from the adapter and
every layer the adapter carries is invisible to vLLM -- and nothing in that
sequence is an error, which is exactly what makes it silent.

Qwen3.5-2B makes this plausible: it is a hybrid linear-attention model, so only
6 of its 24 layers carry the `q/k/v/o_proj` modules a LoRA usually targets, and
the other 18 use a different block entirely.

Three questions, all answered from the artefacts themselves:

1. What module paths does the adapter actually carry weights for?
2. What does vLLM's model class declare it supports, and does it declare LoRA
   support at all?
3. When vLLM's own loader reads this adapter, what does it keep?

(3) is the decisive one: it runs the real code path with the real adapter and
reports what survived. Everything it records is a module path or a count --
no model output, so nothing here can carry corpus text.
"""

from __future__ import annotations

import json

from modal_common import GPU_SERVE, MODELS_DIR, MODELS_VOLUME, SERVE_IMAGE, app

MODEL = "Qwen/Qwen3.5-2B"


@app.function(
    gpu=GPU_SERVE,
    image=SERVE_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 30,
)
def probe(adapter: str) -> dict[str, object]:
    import traceback

    try:
        return _probe(adapter)
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }


def _module_stem(key: str) -> str:
    """`base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight` -> `q_proj`.

    The leaf module name is what both sides key on, so it is the comparable
    unit. The full path is kept separately for the layer-index histogram.
    """
    parts = key.split(".")
    for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        if marker in parts:
            return parts[parts.index(marker) - 1]
    return parts[-1]


def _probe(adapter: str) -> dict[str, object]:
    import collections
    import pathlib

    report: dict[str, object] = {"adapter": adapter, "model": MODEL}

    # ---- 1. what the adapter carries ---------------------------------------
    adapter_dir = pathlib.Path(adapter)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    report["adapter_config"] = {
        key: config.get(key)
        for key in ("peft_type", "r", "lora_alpha", "target_modules", "modules_to_save")
    }

    from safetensors import safe_open

    weight_file = next(iter(adapter_dir.glob("adapter_model.safetensors")), None)
    tensor_keys: list[str] = []
    if weight_file is not None:
        with safe_open(str(weight_file), framework="pt") as handle:  # type: ignore[no-untyped-call]
            tensor_keys = list(handle.keys())

    stems = collections.Counter(_module_stem(k) for k in tensor_keys if "lora" in k)
    layers = sorted({int(p) for k in tensor_keys for p in k.split(".") if p.isdigit()})
    report["adapter_weights"] = {
        "tensors": len(tensor_keys),
        "module_stems": dict(sorted(stems.items())),
        "layers_touched": layers,
        "n_layers_touched": len(layers),
    }

    # ---- 2. what vLLM's model class declares -------------------------------
    #
    # vLLM moves these APIs between releases, so nothing below assumes a
    # location: the probe reports what it found and how, and records the
    # failures, because "the attribute moved" and "the attribute is empty" are
    # different answers and both matter.
    import importlib
    import pkgutil

    import vllm
    from transformers import AutoConfig

    report["vllm_version"] = getattr(vllm, "__version__", None)

    hf_config = AutoConfig.from_pretrained(MODEL)
    architectures = list(getattr(hf_config, "architectures", []) or [])
    report["architectures"] = architectures

    import vllm.lora

    report["vllm_lora_submodules"] = sorted(
        m.name for m in pkgutil.iter_modules(vllm.lora.__path__)
    )

    notes: list[str] = []
    model_cls = None
    from vllm.model_executor.models.registry import ModelRegistry

    for name in architectures:
        # The registry's private map is the only lookup that does not need a
        # fully built ModelConfig, which a probe has no business constructing.
        try:
            spec = ModelRegistry.models.get(name)  # type: ignore[attr-defined]
            if spec is not None:
                model_cls = spec.load_model_cls()
                break
        except Exception as exc:
            notes.append(f"registry.models[{name}]: {type(exc).__name__}: {exc}")
        try:
            module = importlib.import_module("vllm.model_executor.models.qwen3_5")
            model_cls = next(
                (obj for attr in dir(module) if attr == name for obj in [getattr(module, attr)]),
                None,
            )
            if model_cls is not None:
                break
        except Exception as exc:
            notes.append(f"import qwen3_5: {type(exc).__name__}: {exc}")

    report["resolve_notes"] = notes

    if model_cls is None:
        report["vllm_model_class"] = None
    else:
        report["vllm_model_class"] = model_cls.__name__
        try:
            from vllm.model_executor.models.interfaces import supports_lora

            report["declares_lora_support"] = bool(supports_lora(model_cls))
        except Exception as exc:
            report["declares_lora_support"] = f"unavailable: {exc}"
        report["packed_modules_mapping"] = getattr(model_cls, "packed_modules_mapping", None)
        # Older vLLM exposed an explicit allowlist; newer versions infer it.
        report["supported_lora_modules"] = getattr(model_cls, "supported_lora_modules", None)
        report["embedding_modules"] = getattr(model_cls, "embedding_modules", None)
        report["mro"] = [c.__name__ for c in type.mro(model_cls)]

    # ---- 3. what vLLM's own loader keeps -----------------------------------
    #
    # The decisive test. Everything above is inference from declarations; this
    # runs the loader on the real adapter and counts what came out the far end.
    loader_report: dict[str, object] = {}
    lora_model_cls = None
    for module_name in ("vllm.lora.models", "vllm.lora.lora_model", "vllm.lora.request"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            loader_report[f"import {module_name}"] = f"{type(exc).__name__}: {exc}"
            continue
        candidate = getattr(module, "LoRAModel", None)
        if candidate is not None:
            lora_model_cls = candidate
            loader_report["loader_module"] = module_name
            break

    if lora_model_cls is None:
        loader_report["error"] = "LoRAModel not found in any known vllm.lora module"
    else:
        try:
            from vllm.lora.peft_helper import PEFTHelper

            helper = PEFTHelper.from_local_dir(str(adapter_dir), max_position_embeddings=5120)
            # What vLLM's worker computes: every supported module name, with
            # packed groups expanded back into the names PEFT would have used.
            packed = getattr(model_cls, "packed_modules_mapping", None) or {}
            expected = sorted({name for group in packed.values() for name in group} | set(packed))
            lora = lora_model_cls.from_local_checkpoint(
                str(adapter_dir),
                expected_lora_modules=expected,
                peft_helper=helper,
                lora_model_id=1,
                device="cpu",
            )
            kept = sorted(lora.loras)
            loader_report.update(
                {
                    "expected_lora_modules": expected,
                    "modules_kept": len(kept),
                    "module_stems_kept": dict(
                        sorted(collections.Counter(_module_stem(k) for k in kept).items())
                    ),
                    "sample_module_names": kept[:8],
                }
            )
        except BaseException as exc:
            loader_report["error"] = f"{type(exc).__name__}: {exc}"

    report["vllm_loader"] = loader_report

    # ---- 4. do the loaded names match the model's own module names? --------
    #
    # Loading is not applying. vLLM looks each LoRA up *by module name* when it
    # activates an adapter, and a name it cannot find is skipped without a
    # word. So the last question is whether the names the loader produced are
    # names this model actually has -- and for a multimodal wrapper class they
    # may not be, because the language model sits under a prefix that a
    # causal-LM checkpoint has never heard of.
    naming: dict[str, object] = {}
    mapper = getattr(model_cls, "hf_to_vllm_mapper", None) if model_cls is not None else None
    naming["has_hf_to_vllm_mapper"] = mapper is not None
    if mapper is not None:
        for attr in ("orig_to_new_prefix", "orig_to_new_substr", "orig_to_new_suffix"):
            value = getattr(mapper, attr, None)
            if value:
                naming[attr] = value
    try:
        import inspect

        from vllm.lora.utils import parse_fine_tuned_lora_name

        # Record the signature. "Every key failed to parse" and "the probe
        # called the function wrongly" produce the same empty list, and only
        # one of them is a finding.
        naming["parse_signature"] = str(inspect.signature(parse_fine_tuned_lora_name))

        def parse_all(with_mapper: bool) -> dict[str, object]:
            names: list[str] = []
            failures: list[str] = []
            for key in tensor_keys:
                try:
                    result = parse_fine_tuned_lora_name(key, mapper if with_mapper else None)
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    continue
                names.append(str(result[0]))
            return {
                "parsed": len(names),
                "failed": len(failures),
                "distinct_names": len(set(names)),
                "sample_names": sorted(set(names))[:6],
                "prefixes": sorted({n.split(".")[0] for n in names}),
                "sample_failures": sorted(set(failures))[:3],
            }

        naming["with_mapper"] = parse_all(with_mapper=True)
        naming["without_mapper"] = parse_all(with_mapper=False)
    except Exception as exc:
        naming["error"] = f"{type(exc).__name__}: {exc}"

    report["module_naming"] = naming
    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main(adapter: str = "Qwen__Qwen3.5-2B/e10/seed0/epoch10") -> None:
    import pathlib

    adapter_path = adapter if adapter.startswith(MODELS_DIR) else f"{MODELS_DIR}/adapters/{adapter}"
    report = probe.remote(adapter=adapter_path)
    if "error" in report and "adapter_weights" not in report:
        print(report["error"])
        print(report.get("traceback", ""))
        raise SystemExit(1)

    root = pathlib.Path(__file__).resolve().parents[2]
    out = root / "results" / "diagnostics" / "lora_modules.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    weights = report.get("adapter_weights", {})
    print(f"adapter tensors: {weights.get('tensors')}")
    print(f"  module stems : {weights.get('module_stems')}")
    print(f"  layers       : {weights.get('n_layers_touched')} -> {weights.get('layers_touched')}")
    print()
    print(f"vLLM class          : {report.get('vllm_model_class')}")
    print(f"declares LoRA support: {report.get('declares_lora_support')}")
    print(f"packed_modules_mapping: {report.get('packed_modules_mapping')}")
    print(f"supported_lora_modules: {report.get('supported_lora_modules')}")
    print()
    print(f"vLLM loader: {report.get('vllm_loader')}")
    print(f"\nwrote {out}")
