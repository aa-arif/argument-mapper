"""Load-test the local extraction route with Locust.

    PYTHONIOENCODING=utf-8 uv run --with modal modal run scripts/gpu/loadtest.py

Locust runs **inside the same container** as the vLLM server, driving it over
localhost. Driving it from a laptop would have measured a home broadband link
and the round trip to Modal's region; what is wanted is the serving system's
throughput and latency, so the client sits next to the server.

Locust runs headless with zero wait time, which makes each run a closed-loop
test at a fixed concurrency: `--users N` keeps N requests in flight, so the
sweep reports what the server does at each level rather than what an arrival
rate does to it.

Only the local route is tested. Driving concurrent load at the Anthropic API
would spend real money to measure someone else's infrastructure, and
per-request latency for that route is already recorded from the baseline runs.

The GPU and the sampler are both reported: flashinfer's sampler is disabled in
this image (DECISIONS D21), and which kernels produced a throughput number is
part of the number.
"""

from __future__ import annotations

import json
import time

from modal_common import GPU_SERVE, MODELS_DIR, MODELS_VOLUME, app, serve_image_with

BASE_MODEL = "Qwen/Qwen3.5-2B"

#: Concurrency levels to sweep. One request at a time is the latency figure a
#: single user sees; the higher levels show where batching pays and where the
#: GPU saturates.
CONCURRENCY = (1, 2, 4, 8, 16, 32)

DURATION = 45
WARMUP = 15

LOADTEST_IMAGE = serve_image_with("locust==2.42.2")

#: Written into the container at run time. Kept as a string rather than a file
#: in the repo because it is meaningless outside this harness -- it needs the
#: prompt pool and the server the harness starts.
LOCUSTFILE = """
import json
import random

from locust import HttpUser, constant, task

with open("/tmp/prompts.json", encoding="utf-8") as fh:
    PROMPTS = json.load(fh)

MAX_TOKENS = int(open("/tmp/max_tokens", encoding="utf-8").read().strip())


class ExtractUser(HttpUser):
    # No think time: this is a closed-loop capacity test, so each simulated
    # user issues its next request the moment the previous one returns.
    wait_time = constant(0)

    @task
    def extract(self):
        prompt = random.choice(PROMPTS)
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": prompt["messages"],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
            name="chat/completions",
            timeout=600,
        )
"""


@app.function(
    gpu=GPU_SERVE,
    image=LOADTEST_IMAGE,
    volumes={MODELS_DIR: MODELS_VOLUME},
    timeout=60 * 90,
)
def loadtest(
    adapter: str,
    prompts: list[dict[str, object]],
    concurrency: list[int] | None = None,
    duration: int = DURATION,
    max_tokens: int = 3072,
) -> dict[str, object]:
    """Serve the model, then drive it with Locust at each concurrency level."""
    import traceback

    try:
        return _loadtest(adapter, prompts, concurrency, duration, max_tokens)
    except BaseException as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
        }


def _start_server(adapter: str, port: int):  # noqa: ANN202 - subprocess handle
    import subprocess

    # Fixed argv, no shell.
    return subprocess.Popen(
        [
            "vllm",
            "serve",
            BASE_MODEL,
            "--port",
            str(port),
            "--max-model-len",
            "5120",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.90",
            "--enable-lora",
            "--max-lora-rank",
            "32",
            "--lora-modules",
            f"local={adapter}",
        ],
    )


def _wait_healthy(base: str, server, timeout: int = 900) -> float:  # noqa: ANN001
    import httpx

    started = time.perf_counter()
    with httpx.Client(timeout=5.0) as probe:
        while time.perf_counter() - started < timeout:
            if server.poll() is not None:
                msg = f"vllm serve exited with code {server.returncode}"
                raise RuntimeError(msg)
            try:
                if probe.get(f"{base}/health").status_code == 200:
                    return time.perf_counter() - started
            except Exception:
                time.sleep(3)
    msg = f"vllm serve did not become healthy within {timeout}s"
    raise RuntimeError(msg)


def _run_locust(base: str, users: int, seconds: int, tag: str) -> dict[str, object]:
    """One headless Locust run, read back from its CSV stats."""
    import csv
    import subprocess
    from pathlib import Path

    prefix = f"/tmp/locust-{tag}"
    # Fixed argv, no shell.
    subprocess.run(
        [
            "locust",
            "-f",
            "/tmp/locustfile.py",
            "--headless",
            "--host",
            base,
            "--users",
            str(users),
            # Spawn everyone at once: a slow ramp would spend part of a short
            # run at the wrong concurrency.
            "--spawn-rate",
            str(users),
            "--run-time",
            f"{seconds}s",
            "--csv",
            prefix,
            "--only-summary",
        ],
        check=False,
        capture_output=True,
    )

    stats_path = Path(f"{prefix}_stats.csv")
    if not stats_path.exists():
        return {"concurrency": users, "error": "locust produced no stats"}

    with stats_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    aggregated = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    if aggregated is None:
        return {"concurrency": users, "error": "no aggregated row"}

    def number(key: str, default: float = 0.0) -> float:
        try:
            return float(aggregated.get(key) or default)
        except ValueError:
            return default

    return {
        "concurrency": users,
        "requests": int(number("Request Count")),
        "failures": int(number("Failure Count")),
        "throughput_rps": round(number("Requests/s"), 3),
        # Locust reports milliseconds.
        "latency_s": {
            "p50": round(number("50%") / 1000, 3),
            "p95": round(number("95%") / 1000, 3),
            "p99": round(number("99%") / 1000, 3),
            "mean": round(number("Average Response Time") / 1000, 3),
            "max": round(number("Max Response Time") / 1000, 3),
        },
    }


def _loadtest(
    adapter: str,
    prompts: list[dict[str, object]],
    concurrency: list[int] | None,
    duration: int,
    max_tokens: int,
) -> dict[str, object]:
    from pathlib import Path

    levels = concurrency or list(CONCURRENCY)
    port = 8000
    base = f"http://127.0.0.1:{port}"

    Path("/tmp/prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    Path("/tmp/max_tokens").write_text(str(max_tokens), encoding="utf-8")
    Path("/tmp/locustfile.py").write_text(LOCUSTFILE, encoding="utf-8")

    server = _start_server(adapter, port)
    try:
        startup_seconds = _wait_healthy(base, server)

        # Warm up before measuring: the first requests pay for CUDA graph
        # capture and kernel autotuning, which would otherwise land entirely in
        # the concurrency-1 numbers and make them look far worse than steady
        # state.
        _run_locust(base, 4, WARMUP, "warmup")

        results = [_run_locust(base, level, duration, str(level)) for level in levels]
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()

    return json.loads(
        json.dumps(
            {
                "model": BASE_MODEL,
                "adapter": adapter,
                "gpu": GPU_SERVE,
                "sampler": "pytorch-native (flashinfer sampler disabled)",
                "driver": "locust 2.42.2, headless, zero wait time, in-container",
                "max_tokens": max_tokens,
                "documents_in_pool": len(prompts),
                "server_startup_seconds": round(startup_seconds, 1),
                "warmup_seconds": WARMUP,
                "duration_per_level_seconds": duration,
                "levels": results,
            },
            default=str,
        )
    )


@app.local_entrypoint()
def main(
    adapter: str = "Qwen__Qwen3.5-2B/e10/seed0/epoch10",
    limit: int = 24,
    duration: int = DURATION,
) -> None:
    import pathlib

    from argmap.cli.run_baseline import load_split
    from argmap.train_data import to_messages

    adapter_path = adapter if adapter.startswith(MODELS_DIR) else f"{MODELS_DIR}/adapters/{adapter}"

    root = pathlib.Path(__file__).resolve().parents[2]
    # Drawn from validation: this measures throughput, and reusing test
    # documents for it would touch the test split for a non-test purpose.
    docs = load_split(root, "aae-v2", "val")[:limit]
    prompts = [{"doc_id": d.doc_id, "messages": to_messages(d, include_answer=False)} for d in docs]

    print(f"load-testing {adapter_path} with {len(prompts)} documents")
    report = loadtest.remote(adapter=adapter_path, prompts=prompts, duration=duration)

    if "error" in report:
        print(f"FAILED: {report['error']}")
        print(report.get("traceback", ""))
        raise SystemExit(1)

    out = root / "results" / "serving" / "loadtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{report['gpu']}, {report['sampler']}")
    print(f"driver: {report['driver']}")
    print(f"server startup {report['server_startup_seconds']}s\n")
    print(f"  {'conc':>5}{'req/s':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'reqs':>7}{'fail':>6}")
    for level in report["levels"]:
        if "error" in level:
            print(f"  {level['concurrency']:>5}  {level['error']}")
            continue
        lat = level["latency_s"]
        print(
            f"  {level['concurrency']:>5}{level['throughput_rps']:>9.3f}"
            f"{lat['p50']:>9.2f}{lat['p95']:>9.2f}{lat['p99']:>9.2f}"
            f"{level['requests']:>7}{level['failures']:>6}"
        )
    print(f"\nwrote {out}")
