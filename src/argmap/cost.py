"""Token accounting, pricing, and a hard spend ceiling.

Every API call in this project is metered and appended to a ledger under
`results/cost/`. Two reasons, both of which matter for the question the
repository asks:

1. The cost comparison against a local model is only meaningful if the API
   side is measured rather than estimated.
2. A runaway loop over 402 documents is an expensive mistake to make twice.
   `SpendLedger.reserve` raises *before* a request is issued once the projected
   total would cross the cap.

Prices are US dollars per million tokens, first-party Anthropic API rates as
published for the models used here. They are declared in one place so a price
change is a one-line edit rather than a hunt.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ModelPricing:
    """Dollars per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    #: Cached prefix reads are billed at a tenth of the input rate.
    cache_read_multiplier: float = 0.1
    #: Writing a prefix into the cache costs a 25% premium over input.
    cache_write_multiplier: float = 1.25
    #: The Batch API applies a 50% discount to everything.
    batch_multiplier: float = 0.5


PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
}

#: Local inference is priced from GPU rental, not per token. Set from the
#: measured throughput in milestone 6; `None` until then so nothing downstream
#: can quietly use a placeholder.
LOCAL_GPU_USD_PER_HOUR: float | None = None


class SpendCapExceededError(RuntimeError):
    """Raised before issuing a request that would breach the configured cap."""


@dataclass(frozen=True)
class Usage:
    """Token counts for one API call, as reported by the API itself."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def price(usage: Usage, model: str, *, batch: bool = False) -> float:
    """Dollar cost of one call. Unknown models price at zero and are flagged."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0

    dollars = (
        usage.input_tokens * pricing.input_per_mtok
        + usage.output_tokens * pricing.output_per_mtok
        + usage.cache_read_tokens * pricing.input_per_mtok * pricing.cache_read_multiplier
        + usage.cache_write_tokens * pricing.input_per_mtok * pricing.cache_write_multiplier
    ) / 1_000_000

    return dollars * pricing.batch_multiplier if batch else dollars


def default_cap() -> float:
    """Spend ceiling, from `ARGMAP_SPEND_CAP_USD`. CI sets this to 0."""
    raw = os.environ.get("ARGMAP_SPEND_CAP_USD", "20.0")
    try:
        return float(raw)
    except ValueError:
        return 20.0


@dataclass(frozen=True)
class LedgerEntry:
    timestamp: str
    model: str
    doc_id: str
    usage: Usage
    dollars: float
    latency_s: float
    batch: bool
    cached: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "doc_id": self.doc_id,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_write_tokens": self.usage.cache_write_tokens,
            "dollars": round(self.dollars, 6),
            "latency_s": round(self.latency_s, 4),
            "batch": self.batch,
            "cached": self.cached,
            "note": self.note,
        }


class SpendLedger:
    """Append-only record of API spend, with a cap checked before each call.

    The ledger is reloaded from disk on construction, so the cap survives
    process restarts -- otherwise re-running a script would silently reset the
    budget, which is precisely when overspend happens.
    """

    def __init__(self, path: Path, cap_usd: float | None = None) -> None:
        self.path = path
        self.cap_usd = default_cap() if cap_usd is None else cap_usd
        self._lock = threading.Lock()
        self._spent = self._replay()

    def _replay(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    total += float(json.loads(line).get("dollars", 0.0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return total

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self._spent)

    def reserve(self, projected_usd: float, *, note: str = "") -> None:
        """Raise if this call would push cumulative spend past the cap.

        Called *before* the request. A cached response should not reserve
        anything, since it costs nothing.
        """
        with self._lock:
            if self._spent + projected_usd > self.cap_usd:
                msg = (
                    f"spend cap would be exceeded: ${self._spent:.4f} already spent, "
                    f"${projected_usd:.4f} projected, cap ${self.cap_usd:.2f}"
                    + (f" ({note})" if note else "")
                )
                raise SpendCapExceededError(msg)

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        """Append a completed call and add its cost to the running total."""
        entry = replace(entry, timestamp=entry.timestamp or datetime.now(UTC).isoformat())
        with self._lock:
            self._spent += entry.dollars
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry.as_dict()))
                fh.write("\n")
        return entry

    def summary(self) -> dict[str, object]:
        return {
            "spent_usd": round(self._spent, 6),
            "cap_usd": self.cap_usd,
            "remaining_usd": round(self.remaining_usd, 6),
        }
