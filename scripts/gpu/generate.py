"""Batched generation with vLLM, with and without JSON-schema constraints.

Serves three milestones from one function:

* **3** -- generating validation predictions for every saved checkpoint, so a
  checkpoint is selected on validation F1 rather than on loss.
* **4** -- the constrained-vs-unconstrained comparison. Passing `json_schema`
  switches on vLLM's structured-output decoding; running the same prompts both
  ways gives the invalid-output rate and the F1 difference.
* **5** -- `mean_logprob` per generation is the cascade's routing signal.

Adapters are applied through vLLM's LoRA support rather than merged, so
comparing nine checkpoints does not mean first writing nine full copies of a
2B model to a volume. The winner is merged once, for serving.

`generate` takes a *list* of adapters and reuses one engine across all of them.
Loading vLLM costs one to two minutes; paying that nine times would dominate
the actual work.
"""

from __future__ import annotations

import json
import time

from modal_common import GPU_SERVE, MODELS_DIR, MODELS_VOLUME, SERVE_IMAGE, app

#: Deterministic decoding. Sampling would make a checkpoint comparison partly a
#: comparison of luck.
GREEDY = {"temperature": 0.0, "top_p": 1.0}


@app.function(
    gpu=GPU_SERVE,
    image=SERVE_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 90,
)
def generate(
    model_path: str,
    prompts: list[dict[str, object]],
    adapters: list[str] | None = None,
    json_schema: dict[str, object] | None = None,
    # Headroom for the densest plausible answer under the bounded schema:
    # ~30 components at ~50 tokens plus ~22 relations at ~25 is ~2,050, so
    # 2,048 left no room for the grammar to close the object.
    max_tokens: int = 3072,
    max_model_len: int = 5120,
    max_lora_rank: int = 32,
    out_path: str | None = None,
) -> dict[str, object]:
    """Generate completions for every prompt, once per adapter.

    `prompts` are `{"doc_id": str, "messages": [...]}` records -- the training
    chat shape minus the assistant turn. `adapters` may be empty or None to run
    the base model alone.

    Errors are returned as text rather than raised. A vLLM exception cannot be
    unpickled on a machine without vLLM installed, so raising it produces
    "Deserialization failed because the 'vllm' module is not available" and the
    real cause -- which vLLM had already logged clearly -- is lost.
    """
    import traceback

    try:
        return _generate(
            model_path,
            prompts,
            adapters,
            json_schema,
            max_tokens,
            max_model_len,
            max_lora_rank,
            out_path,
        )
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
            "model_path": model_path,
            "adapters": list(adapters or []),
        }


def _generate(
    model_path: str,
    prompts: list[dict[str, object]],
    adapters: list[str] | None,
    json_schema: dict[str, object] | None,
    max_tokens: int,
    max_model_len: int,
    max_lora_rank: int,
    out_path: str | None,
) -> dict[str, object]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_list = list(adapters or [])

    started = time.perf_counter()
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        enable_lora=bool(adapter_list),
        max_lora_rank=max_lora_rank,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    load_seconds = time.perf_counter() - started

    sampling_kwargs: dict[str, object] = {
        **GREEDY,
        "max_tokens": max_tokens,
        # One logprob per generated token is all the cascade needs; asking for
        # more multiplies the payload for no gain.
        "logprobs": 0,
    }
    if json_schema is not None:
        from vllm.sampling_params import StructuredOutputsParams

        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)

    params = SamplingParams(**sampling_kwargs)  # type: ignore[arg-type]
    conversations = [p["messages"] for p in prompts]

    runs: list[dict[str, object]] = []
    for index, adapter in enumerate(adapter_list or [None]):  # type: ignore[list-item]
        lora = LoRARequest(f"adapter{index}", index + 1, adapter) if adapter else None

        began = time.perf_counter()
        outputs = llm.chat(conversations, params, lora_request=lora)
        generate_seconds = time.perf_counter() - began

        results: list[dict[str, object]] = []
        for prompt, output in zip(prompts, outputs, strict=True):
            completion = output.outputs[0]
            logprobs: list[float] = []
            for entry in completion.logprobs or []:
                if entry:
                    logprobs.append(max(lp.logprob for lp in entry.values()))

            results.append(
                {
                    "doc_id": prompt["doc_id"],
                    "text": completion.text,
                    "output_tokens": len(completion.token_ids),
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "finish_reason": completion.finish_reason,
                    # The cascade routes on this: a low mean token logprob is
                    # the model reporting it was unsure, which is when
                    # escalating to the API pays for itself.
                    "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
                    "min_logprob": min(logprobs) if logprobs else None,
                }
            )

        runs.append(
            {
                "adapter": adapter,
                "generate_seconds": round(generate_seconds, 2),
                "documents": len(results),
                "throughput_docs_per_second": (
                    round(len(results) / generate_seconds, 3) if generate_seconds else 0.0
                ),
                "results": results,
            }
        )

    payload = {
        "model_path": model_path,
        "constrained": json_schema is not None,
        "load_seconds": round(load_seconds, 2),
        "gpu": GPU_SERVE,
        "runs": runs,
    }
    # JSON round-trip: vLLM returns its own types, and unpickling them locally
    # would require vllm installed on a machine that has no GPU.
    payload = json.loads(json.dumps(payload, default=str))

    if out_path is None:
        return payload

    # Generated text for a dozen checkpoints runs to several megabytes, which
    # is large enough that returning it over gRPC drops the stream
    # ("StreamTerminatedError: Connection lost") *after* the GPU work is done.
    # Writing to the volume and returning a summary keeps the result durable
    # and the response small.
    from pathlib import Path

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    MODELS_VOLUME.commit()

    return {
        "written_to": out_path,
        "bytes": target.stat().st_size,
        "model_path": model_path,
        "constrained": json_schema is not None,
        "load_seconds": payload["load_seconds"],
        "runs": [{k: v for k, v in run.items() if k != "results"} for run in runs],
    }
