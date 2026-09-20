"""Batched generation with vLLM, with and without JSON-schema constraints.

Serves three milestones from one function:

* **3** -- generating validation predictions for each saved checkpoint, so a
  checkpoint can be selected on validation F1 rather than on loss.
* **4** -- the constrained-vs-unconstrained comparison. Passing
  `json_schema` switches on vLLM's structured-output decoding, and running the
  same prompts both ways gives the invalid-output rate and the F1 difference.
* **5** -- `mean_logprob` per generation is the cascade's routing signal.

Adapters are applied through vLLM's LoRA support rather than merged, so
selecting among nine checkpoints does not mean writing nine full copies of a
2B model to a volume first. The winner is merged once, for serving.
"""

from __future__ import annotations

import json

from modal_common import GPU_SERVE, MODELS_DIR, MODELS_VOLUME, SERVE_IMAGE, app

#: Deterministic decoding. Sampling would make a checkpoint comparison partly
#: a comparison of luck.
GREEDY = {"temperature": 0.0, "top_p": 1.0}


@app.function(
    gpu=GPU_SERVE,
    image=SERVE_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 60,
)
def generate(
    model_path: str,
    prompts: list[dict[str, object]],
    *,
    adapter_path: str | None = None,
    json_schema: dict[str, object] | None = None,
    max_tokens: int = 2048,
    max_model_len: int = 4096,
) -> list[dict[str, object]]:
    """Generate one completion per prompt.

    `prompts` are `{"doc_id": str, "messages": [...]}` records -- the same chat
    shape used for training, minus the assistant turn.
    """
    import time

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    started = time.perf_counter()
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        enable_lora=adapter_path is not None,
        max_lora_rank=64,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    load_seconds = time.perf_counter() - started

    sampling_kwargs: dict[str, object] = {
        **GREEDY,
        "max_tokens": max_tokens,
        # One logprob per generated token is all the cascade needs, and asking
        # for more multiplies the response payload for no gain.
        "logprobs": 0,
    }
    if json_schema is not None:
        from vllm.sampling_params import GuidedDecodingParams

        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)

    params = SamplingParams(**sampling_kwargs)  # type: ignore[arg-type]
    lora = LoRARequest("adapter", 1, adapter_path) if adapter_path else None

    conversations = [p["messages"] for p in prompts]

    started = time.perf_counter()
    outputs = llm.chat(conversations, params, lora_request=lora)
    generate_seconds = time.perf_counter() - started

    results: list[dict[str, object]] = []
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output.outputs[0]
        logprobs = []
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
                # The cascade routes on this: a low mean token logprob is the
                # model saying it was unsure, which is when escalating pays.
                "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
                "min_logprob": min(logprobs) if logprobs else None,
            }
        )

    return json.loads(
        json.dumps(
            {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "constrained": json_schema is not None,
                "load_seconds": round(load_seconds, 2),
                "generate_seconds": round(generate_seconds, 2),
                "documents": len(results),
                "throughput_docs_per_second": (
                    round(len(results) / generate_seconds, 3) if generate_seconds else 0.0
                ),
                "results": results,
            },
            default=str,
        )
    )
