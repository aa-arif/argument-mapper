"""The `Extractor` interface every extraction backend implements.

Claude, the fine-tuned local model, and the cascade are interchangeable behind
this protocol, which is what lets one evaluation harness score all three and
one serving route dispatch to any of them.

An extractor returns not just a prediction but what producing it cost: tokens,
dollars, and wall-clock latency. Those travel with the result rather than being
reconstructed later, because the comparison this repository exists to make is
quality *per dollar* and *per second*, and a number reconstructed after the
fact is a number nobody can check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from argmap.cost import Usage
from argmap.schema import Document, Prediction


@dataclass(frozen=True)
class ExtractionResult:
    """A prediction plus everything measured about producing it."""

    prediction: Prediction
    usage: Usage = field(default_factory=Usage)
    dollars: float = 0.0
    latency_s: float = 0.0
    #: True when served from the on-disk response cache, in which case
    #: `latency_s` and `dollars` describe the cache hit, not the original call.
    cached: bool = False
    #: True when the raw output failed schema validation and was repaired or
    #: dropped. Tracked so milestone 4 can report an invalid-output rate.
    invalid_output: bool = False
    #: Mean token logprob where the backend exposes it; the cascade's routing
    #: signal in milestone 5. `None` for backends that do not report logprobs.
    mean_logprob: float | None = None
    raw_text: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns a document into an argument graph."""

    @property
    def name(self) -> str:
        """Stable identifier used in result filenames and tables."""
        ...

    def extract(self, doc: Document) -> ExtractionResult:
        """Extract the argument graph for one document."""
        ...


class BatchExtractor(Protocol):
    """An extractor that can also run a whole corpus more cheaply.

    Separate from `Extractor` because the Anthropic Batch API halves cost but
    makes latency meaningless -- a batched job may sit queued for minutes. The
    evaluation harness therefore uses batch mode for scoring and synchronous
    calls on a subsample for the latency table.
    """

    @property
    def name(self) -> str: ...

    def extract(self, doc: Document) -> ExtractionResult: ...

    def extract_many(self, docs: list[Document]) -> dict[str, ExtractionResult]:
        """Extract for many documents, keyed by `doc_id`."""
        ...
