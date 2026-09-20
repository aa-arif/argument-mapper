"""Locate nvcc and the CUDA headers inside the serving image.

    PYTHONIOENCODING=utf-8 uv run modal run scripts/gpu/probe_cuda.py

vLLM's flashinfer backend JIT-compiles kernels at engine start and looks for a
CUDA toolkit at `$CUDA_HOME`, `$CUDA_PATH`, or `/usr/local/cuda`. A slim base
image has none of those, so the engine dies after the model has already
loaded. The toolkit *is* present -- pip pulls it in as `nvidia-cuda-nvcc` and
friends -- just not where flashinfer looks.

Finds the real paths so `CUDA_HOME` can be set correctly, instead of switching
to a multi-gigabyte CUDA devel base image on a hunch.
"""

from __future__ import annotations

import json

from modal_common import SERVE_IMAGE, app


@app.function(image=SERVE_IMAGE, timeout=60 * 10)
def probe() -> dict[str, object]:
    import os
    import shutil
    from pathlib import Path

    report: dict[str, object] = {
        "which_nvcc": shutil.which("nvcc"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "usr_local_cuda_exists": Path("/usr/local/cuda").exists(),
    }

    # Where did pip put the toolkit?
    roots = [Path(p) for p in ("/usr/local/lib/python3.11/site-packages/nvidia",)]
    found_nvcc: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        report["nvidia_packages"] = sorted(p.name for p in root.iterdir() if p.is_dir())
        found_nvcc.extend(str(p) for p in root.rglob("bin/nvcc"))
    report["nvcc_candidates"] = found_nvcc

    # Header locations matter as much as the compiler.
    headers: list[str] = []
    for root in roots:
        if root.exists():
            headers.extend(str(p.parent) for p in root.rglob("include/cuda_runtime.h"))
    report["cuda_runtime_header_dirs"] = headers

    # Some wheels ship a consolidated layout under a `cuda_toolkit` package.
    for candidate in ("/usr/local/cuda-13", "/usr/local/cuda-12", "/opt/cuda"):
        if Path(candidate).exists():
            report.setdefault("other_toolkit_dirs", []).append(candidate)  # type: ignore[attr-defined]

    return json.loads(json.dumps(report, default=str))


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), indent=2))
