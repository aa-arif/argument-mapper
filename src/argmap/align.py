"""Aligning generated component text back to character spans.

A model that emits argument components as *text* has not said where in the
document they are. Every span-based metric needs offsets, so the generated text
has to be located in the source -- and the error that alignment introduces is a
ceiling on the F1 any generate-then-align system can reach, regardless of how
good its extraction is.

`multi_scale_match` is the v1 matcher, ported unchanged in behaviour so the
measurement in `scripts/eval_alignment.py` describes the system that actually
shipped. `refine_span` is an optional second pass that fixes its known
weakness: the coarse search steps in quarter-window increments, so a returned
span is quantised to that step even when the true span sits between two
positions. Both are measured; see `results/alignment/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

#: Window scales tried, relative to the length of the text being located.
SCALES: tuple[float, ...] = (1.0, 1.2, 0.8, 1.5, 0.6)

#: Similarity below which the matcher reports failure rather than a bad guess.
MIN_RATIO = 0.4

#: Smallest window the coarse search will consider, in characters.
MIN_WINDOW = 10


@dataclass(frozen=True)
class Alignment:
    """Where a piece of generated text was located, and how confidently."""

    start: int
    end: int
    ratio: float
    exact: bool

    @property
    def found(self) -> bool:
        return self.end > self.start


def multi_scale_match(needle: str, haystack: str) -> Alignment | None:
    """Locate `needle` in `haystack`, exactly if possible and fuzzily otherwise.

    Behaviourally identical to the v1 implementation: an exact substring search
    first, then a sliding window at several scales scored by
    `difflib.SequenceMatcher`, stepping a quarter of the window each time.
    Returns `None` when nothing clears `MIN_RATIO`.
    """
    if not needle or not haystack:
        return None

    exact = haystack.find(needle)
    if exact != -1:
        return Alignment(start=exact, end=exact + len(needle), ratio=1.0, exact=True)

    needle_lower = needle.lower()
    haystack_lower = haystack.lower()
    needle_len = len(needle)

    best_ratio = 0.0
    best_start = 0
    best_end = min(needle_len, len(haystack))

    for scale in SCALES:
        window = max(MIN_WINDOW, int(needle_len * scale))
        step = max(1, window // 4)
        for i in range(0, max(1, len(haystack) - window + 1), step):
            candidate = haystack_lower[i : i + window]
            ratio = SequenceMatcher(None, needle_lower, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = min(i + window, len(haystack))

    if best_ratio <= MIN_RATIO:
        return None
    return Alignment(start=best_start, end=best_end, ratio=best_ratio, exact=False)


def refine_span(
    needle: str,
    haystack: str,
    alignment: Alignment,
    *,
    radius: int = 24,
) -> Alignment:
    """Hill-climb the boundaries of a coarse alignment one character at a time.

    The coarse search only ever returns starts that are multiples of its step,
    so a span can be off by up to a quarter-window even when the right answer is
    nearby. This walks both edges within `radius` characters, keeping whatever
    improves the similarity score.
    """
    if alignment.exact:
        return alignment

    needle_lower = needle.lower()
    haystack_lower = haystack.lower()
    n = len(haystack)

    def score(start: int, end: int) -> float:
        if end <= start:
            return 0.0
        return SequenceMatcher(None, needle_lower, haystack_lower[start:end]).ratio()

    start, end = alignment.start, alignment.end
    best = score(start, end)

    for _ in range(2):  # a second sweep lets the edges settle against each other
        for delta in range(-radius, radius + 1):
            candidate = start + delta
            if 0 <= candidate < end:
                value = score(candidate, end)
                if value > best:
                    best, start = value, candidate
        for delta in range(-radius, radius + 1):
            candidate = end + delta
            if start < candidate <= n:
                value = score(start, candidate)
                if value > best:
                    best, end = value, candidate

    return Alignment(start=start, end=end, ratio=best, exact=False)


def align(needle: str, haystack: str, *, refine: bool = True) -> Alignment | None:
    """Locate `needle`, optionally refining the coarse result."""
    coarse = multi_scale_match(needle, haystack)
    if coarse is None:
        return None
    return refine_span(needle, haystack, coarse) if refine else coarse
