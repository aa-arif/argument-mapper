"""On-disk cache for model responses.

Re-running an evaluation should cost nothing. The cache key covers everything
that could change an output -- model, prompt version, decoding parameters, and
the document text itself -- so a cache hit is genuinely the same call and a
prompt edit correctly misses.

Keyed by content rather than by document id on purpose: two runs over the same
document with different prompts must not collide, and the same document
appearing in two corpora must hit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def cache_key(
    *,
    model: str,
    prompt_version: str,
    document_text: str,
    params: dict[str, object] | None = None,
) -> str:
    """Stable SHA-256 over everything that can change the response."""
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "document": document_text,
        # sort_keys so an unordered dict cannot produce two keys for one call
        "params": params or {},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResponseCache:
    """A directory of JSON blobs, one per cached response.

    Sharded one level by key prefix: a flat directory of several hundred
    thousand files is slow to list on every filesystem that matters.
    """

    root: Path

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, object] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                data: dict[str, object] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A truncated blob from an interrupted write is a miss, not a crash.
            return None
        return data

    def put(self, key: str, value: dict[str, object]) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run cannot leave a half-written
        # blob that later reads as a valid cache hit.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(value, fh, ensure_ascii=False)
        tmp.replace(path)

    def __contains__(self, key: str) -> bool:
        return self.path_for(key).exists()

    def size(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0
