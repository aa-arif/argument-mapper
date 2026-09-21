"""FastAPI service exposing every extraction route behind one endpoint.

    uv run uvicorn argmap.serve.app:app --reload

Routes are `claude`, `local` and `cascade`. They share the request shape, the
response schema, the result cache and the metrics, so the front end picks a
route by name and nothing else changes -- which is what makes the
side-by-side comparison in the UI a fair one.

Two things are deliberately *not* here. There is no JSON-repair path on the
local route, because constrained decoding measured a 0% invalid rate against
100% without it (milestone 4). And the Claude route does not fall back to the
local model on error: a silent downgrade would make the latency and quality
numbers mean nothing.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argmap.schema import Component, Prediction, Relation

Route = Literal["claude", "local", "cascade"]

#: Result cache size. Entries are small (a graph, not a document), and the
#: cache is keyed by content so repeated analysis of the same text is free.
CACHE_ENTRIES = 512

MAX_DOCUMENT_CHARS = 50_000


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    route: Route = "claude"
    model: str | None = Field(
        default=None,
        description="Override the model for the claude route, e.g. claude-sonnet-5.",
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    dollars: float = 0.0


class ExtractResponse(BaseModel):
    """The unified graph, plus what producing it cost."""

    components: list[Component]
    relations: list[Relation]
    route: Route
    model: str
    cached: bool
    latency_ms: float
    usage: Usage = Field(default_factory=Usage)
    #: Present on the cascade route: whether this document was escalated.
    escalated: bool | None = None
    #: Mean token logprob where the backend reports one.
    confidence: float | None = None


class ResultCache:
    """Content-hashed LRU over extraction results.

    Keyed on the text *and* the route, because the same passage analysed by
    two different routes is two different answers -- which is exactly what the
    UI's comparison view asks for.
    """

    def __init__(self, capacity: int = CACHE_ENTRIES) -> None:
        self._entries: OrderedDict[str, ExtractResponse] = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str, route: str, model: str | None) -> str:
        digest = hashlib.sha256(f"{route}\x00{model or ''}\x00{text}".encode()).hexdigest()
        return digest

    def get(self, key: str) -> ExtractResponse | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, key: str, value: ExtractResponse) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry and reset counters.

        Public because tests need a clean cache between cases, and
        reaching into the internals from outside the class is worse.
        """
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


cache = ResultCache()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    yield


app = FastAPI(title="Argument Mapper", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing(
    request: Request,
    call_next: Callable[[Request], Awaitable[object]],
) -> object:
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    headers = getattr(response, "headers", None)
    if headers is not None:
        headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness plus which routes are actually configured.

    Reports per-route availability rather than a bare `ok`, so the front end
    can grey out a route instead of offering one that will fail.
    """
    return {
        "status": "ok",
        "version": "2.0",
        "routes": {
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "local": bool(os.environ.get("ARGMAP_VLLM_URL")),
            "cascade": bool(os.environ.get("ANTHROPIC_API_KEY"))
            and bool(os.environ.get("ARGMAP_VLLM_URL")),
        },
        "cache": cache.stats(),
    }


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return {"cache": cache.stats()}


def prediction_to_response(
    prediction: Prediction,
    *,
    route: Route,
    model: str,
    latency_ms: float,
    cached: bool = False,
    usage: Usage | None = None,
    escalated: bool | None = None,
    confidence: float | None = None,
) -> ExtractResponse:
    return ExtractResponse(
        components=prediction.components,
        relations=prediction.relations,
        route=route,
        model=model,
        cached=cached,
        latency_ms=round(latency_ms, 2),
        usage=usage or Usage(),
        escalated=escalated,
        confidence=confidence,
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    key = ResultCache.key(request.text, request.route, request.model)
    hit = cache.get(key)
    if hit is not None:
        return hit.model_copy(update={"cached": True})

    started = time.perf_counter()
    try:
        response = _dispatch(request, started)
    except HTTPException:
        raise
    except Exception as exc:  # backend failures become 502, not 500
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    cache.put(key, response)
    return response


def _dispatch(request: ExtractRequest, started: float) -> ExtractResponse:
    """Route to a backend. Split out so tests can exercise it directly."""
    if request.route == "claude":
        return _extract_claude(request, started)
    detail = f"route {request.route!r} is not configured on this deployment"
    raise HTTPException(status_code=503, detail=detail)


def _extract_claude(request: ExtractRequest, started: float) -> ExtractResponse:
    from pathlib import Path

    from argmap.cost import SpendLedger
    from argmap.extractors.cache import ResponseCache
    from argmap.extractors.claude import ClaudeExtractor, load_exemplar_file
    from argmap.schema import Document

    model = request.model or os.environ.get("ARGMAP_CLAUDE_MODEL", "claude-haiku-4-5")
    root = Path(os.environ.get("ARGMAP_ROOT", "."))

    extractor = ClaudeExtractor(
        model=model,
        ledger=SpendLedger(root / "results" / "cost" / "ledger.jsonl"),
        cache=ResponseCache(root / ".cache" / "api"),
        exemplars=load_exemplar_file(root / "prompts" / "exemplars.json"),
    )

    doc = Document(doc_id="request", source="aae-v2", split="test", text=request.text)
    result = extractor.extract(doc)
    if result.error:
        raise HTTPException(status_code=502, detail=result.error)

    return prediction_to_response(
        result.prediction,
        route="claude",
        model=model,
        latency_ms=(time.perf_counter() - started) * 1000,
        usage=Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            dollars=round(result.dollars, 6),
        ),
    )
