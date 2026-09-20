"""Claude extractor, using structured outputs and prompt caching.

Design notes:

* **Structured outputs, not prompt-and-repair.** `output_config.format` pins
  the response to the schema, so the v1 truncated-JSON repair path is not
  carried over here. Comparing a *constrained* local model against an
  *unconstrained, repaired* API baseline would charge the repair path's
  failures to the API rather than to the prompt. See DECISIONS.md D6.
* **The model returns text, not offsets.** Spans are recovered by the fuzzy
  matcher, whose error is measured in `results/alignment/`.
* **Caching is on the prefix**, which is the system prompt plus the worked
  examples. Those are identical across all 402 documents, so after the first
  call they are billed at a tenth of the input rate.
* **Every call is metered** through `SpendLedger`, which refuses to issue a
  request that would breach the cap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from argmap.cost import LedgerEntry, SpendLedger, Usage, price
from argmap.extractors.assemble import graph_to_prediction
from argmap.extractors.base import ExtractionResult
from argmap.extractors.cache import ResponseCache, cache_key
from argmap.prompts import (
    SYSTEM_PROMPT,
    RawGraph,
    build_user_prompt,
    exemplar_turns,
    prompt_fingerprint,
)
from argmap.schema import Document, Prediction

#: Generous enough that a dense essay's graph is never truncated. A 20-component
#: essay renders to roughly 2.5k tokens of JSON.
DEFAULT_MAX_TOKENS = 8000


def _as_int(value: object) -> int:
    """Coerce a value out of a loosely-typed cached payload."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, int | float) else 0.0


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


@dataclass
class ClaudeExtractor:
    """Extract argument graphs with a Claude model."""

    model: str
    ledger: SpendLedger
    cache: ResponseCache
    exemplars: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Set when the caller wants failures to surface rather than be recorded.
    raise_on_error: bool = False
    _client: Any = None

    def __post_init__(self) -> None:
        self._exemplar_ids = [str(e.get("doc_id", "")) for e in self.exemplars]
        # Fingerprints the rendered exemplars, not just which documents they
        # came from, so a change in how they render misses the cache.
        self._fingerprint = prompt_fingerprint(self.exemplars)

    @property
    def name(self) -> str:
        return self.model

    @property
    def prompt_version(self) -> str:
        return self._fingerprint

    # -- client ------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    # -- request shape -----------------------------------------------------

    def _system(self) -> list[dict[str, Any]]:
        # The whole system block is stable across every document, so it is the
        # natural cache breakpoint.
        return [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _messages(self, doc: Document) -> list[dict[str, Any]]:
        turns = exemplar_turns(self.exemplars)
        if turns:
            # Extend the cached prefix through the worked examples: they are
            # identical for every document and are the bulk of the input.
            last = turns[-1]
            last["content"] = [
                {
                    "type": "text",
                    "text": str(last["content"]),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return [*turns, {"role": "user", "content": build_user_prompt(doc.text)}]

    def _cache_key(self, doc: Document) -> str:
        return cache_key(
            model=self.model,
            prompt_version=self._fingerprint,
            document_text=doc.text,
            params={"max_tokens": self.max_tokens, "schema": "RawGraph"},
        )

    # -- extraction --------------------------------------------------------

    def extract(self, doc: Document) -> ExtractionResult:
        key = self._cache_key(doc)
        hit = self.cache.get(key)
        if hit is not None:
            return self._result_from_payload(doc, hit, cached=True)

        # Project the spend before issuing anything. Input is known; output is
        # bounded by max_tokens, so this is a worst case, which is the right
        # side to err on for a cap.
        projected = price(
            Usage(input_tokens=len(doc.text) // 3, output_tokens=self.max_tokens),
            self.model,
        )
        self.ledger.reserve(projected, note=f"{self.model}/{doc.doc_id}")

        client = self._get_client()
        started = time.perf_counter()
        try:
            response = client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system(),
                messages=self._messages(doc),
                output_format=RawGraph,
            )
        except Exception as exc:
            # Broad on purpose: one document failing must not abandon a 400-doc
            # sweep. The error is recorded on the result and surfaces in the
            # run report rather than being swallowed.
            if self.raise_on_error:
                raise
            latency = time.perf_counter() - started
            return ExtractionResult(
                prediction=Prediction(doc_id=doc.doc_id),
                latency_s=latency,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = time.perf_counter() - started

        payload = self._payload_from_response(response, latency)
        self.cache.put(key, payload)

        result = self._result_from_payload(doc, payload, cached=False)
        self.ledger.record(
            LedgerEntry(
                timestamp=datetime.now(UTC).isoformat(),
                model=self.model,
                doc_id=doc.doc_id,
                usage=result.usage,
                dollars=result.dollars,
                latency_s=result.latency_s,
                batch=False,
                cached=False,
            )
        )
        return result

    # -- response handling -------------------------------------------------

    def _payload_from_response(self, response: Any, latency: float) -> dict[str, object]:
        usage = response.usage
        parsed = response.parsed_output
        return {
            "graph": parsed.model_dump() if parsed is not None else None,
            "raw_text": next(
                (b.text for b in response.content if getattr(b, "type", "") == "text"), ""
            ),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
            "latency_s": latency,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": self.model,
        }

    def _result_from_payload(
        self,
        doc: Document,
        payload: dict[str, object],
        *,
        cached: bool,
    ) -> ExtractionResult:
        # The payload is this module's own cache blob, so its shape is known;
        # `cast` records that rather than threading Unknown through.
        raw_usage = payload.get("usage")
        counts: dict[str, object] = (
            cast("dict[str, object]", raw_usage) if isinstance(raw_usage, dict) else {}
        )
        usage = Usage(
            input_tokens=_as_int(counts.get("input_tokens")),
            output_tokens=_as_int(counts.get("output_tokens")),
            cache_read_tokens=_as_int(counts.get("cache_read_tokens")),
            cache_write_tokens=_as_int(counts.get("cache_write_tokens")),
        )
        graph = payload.get("graph")

        if isinstance(graph, dict):
            prediction, dropped = self._to_prediction(doc, cast("dict[str, Any]", graph))
        else:
            prediction, dropped = Prediction(doc_id=doc.doc_id), 0
        prediction.meta["unalignable_components"] = dropped
        prediction.meta["cached"] = cached

        return ExtractionResult(
            prediction=prediction,
            usage=usage,
            # A cache hit costs nothing; reporting the original price here
            # would double-count it in any aggregate.
            dollars=0.0 if cached else price(usage, self.model),
            latency_s=_as_float(payload.get("latency_s")),
            cached=cached,
            invalid_output=not isinstance(graph, dict),
            raw_text=_as_str(payload.get("raw_text")),
        )

    def _to_prediction(self, doc: Document, graph: dict[str, Any]) -> tuple[Prediction, int]:
        """Delegate to the shared assembler so every route parses identically."""
        prediction, stats = graph_to_prediction(doc, graph)
        return prediction, stats.unalignable


def load_exemplar_file(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data: list[dict[str, object]] = json.load(fh)
    return data
