"""Guard against ever committing licensed corpus data.

Argument Annotated Essays v2 is distributed under an agreement whose clause 2.2
forbids publishing, redistributing or displaying the data in any form, and this
repository is public. `.gitignore` is a convention that a single `git add -f`
defeats; these tests are checked in CI and fail the build instead.

They inspect what git actually tracks, not what is on disk, so a developer with
the corpora present locally still passes.

Three layers, deliberately overlapping, because the first two each missed a
real leak once (DECISIONS D30):

1. **paths** -- no corpus file may be tracked. Runs everywhere.
2. **shape** -- no tracked JSON under `results/` may contain a long free-text
   string. Runs everywhere, needs no corpus, and is the layer that would have
   caught a model generation being written into a results file.
3. **content** -- no tracked file may contain any span of the corpus. Needs the
   corpus, so it only runs where the data is present.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import cast

import pytest

#: Paths and patterns that would indicate corpus data has been committed.
_FORBIDDEN_SUFFIXES = (".ann",)
_FORBIDDEN_SUBSTRINGS = (
    "argumentannotatedessays",
    "brat-project",
    "arg-microtexts",
)
_FORBIDDEN_PREFIXES = ("data/",)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git repository")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_no_corpus_files_are_tracked() -> None:
    offenders = [
        path
        for path in _tracked_files()
        if path.lower().endswith(_FORBIDDEN_SUFFIXES)
        or any(s in path.lower() for s in _FORBIDDEN_SUBSTRINGS)
        or path.lower().startswith(_FORBIDDEN_PREFIXES)
    ]
    assert not offenders, (
        "Licensed corpus files are tracked by git, which the Argument Annotated "
        f"Essays license forbids: {offenders}"
    )


# ---------------------------------------------------------------------------
# Layer 2: no free text in results files. Needs no corpus, so it runs in CI.
# ---------------------------------------------------------------------------

#: Longest string a results file may hold outside the allowlist. Every
#: legitimate value in `results/` is an identifier, a label or a short
#: description; anything longer is prose, and prose in a results file is either
#: a model generation or a corpus excerpt.
_MAX_FREE_TEXT_WORDS = 8

#: Keys whose values are written by hand, in this repository, to describe a
#: run. They are ours to publish. Anything not named here is data.
_DESCRIPTION_KEYS = frozenset(
    {
        "description",
        "driver",
        "error",
        "note",
        "notes",
        "sampler",
        "selection_scalar",
        "served",
        "traceback",
    }
)


def _long_strings(node: object, key: str | None, found: list[tuple[str, int]]) -> None:
    """Collect (key, word count) for every over-long string, recursively.

    The offending text is deliberately never collected. Putting it in an
    assertion message would print corpus content into a public CI log, which is
    the same disclosure the test exists to prevent.
    """
    if isinstance(node, dict):
        for child_key, value in cast("dict[object, object]", node).items():
            _long_strings(value, str(child_key), found)
    elif isinstance(node, list):
        for value in cast("list[object]", node):
            # A list inherits its parent's key: {"heads": ["...", "..."]}
            # must not launder text past the allowlist check.
            _long_strings(value, key, found)
    elif isinstance(node, str):
        if key in _DESCRIPTION_KEYS:
            return
        words = len(node.split())
        if words > _MAX_FREE_TEXT_WORDS:
            found.append((key or "<root>", words))


def _tracked_result_documents() -> list[tuple[str, object]]:
    root = _repo_root()
    documents: list[tuple[str, object]] = []
    for path in _tracked_files():
        if not path.startswith("results/"):
            continue
        if not path.endswith((".json", ".jsonl")):
            continue
        text = (root / path).read_text(encoding="utf-8")
        if path.endswith(".jsonl"):
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    documents.append((f"{path}:{number}", json.loads(line)))
        else:
            documents.append((path, json.loads(text)))
    return documents


def test_results_files_contain_no_free_text() -> None:
    """Results files hold measurements, not prose.

    This is the layer that was missing when `probe_adapter_effect.py` wrote
    model generations -- which quote the essays verbatim -- into
    `results/diagnostics/adapter_effect.json` on a public repository. A probe
    may record what a generation *was like*; it may not record what it said.
    """
    offenders: list[str] = []
    for label, payload in _tracked_result_documents():
        found: list[tuple[str, int]] = []
        _long_strings(payload, None, found)
        if found:
            keys = sorted({key for key, _ in found})
            longest = max(words for _, words in found)
            offenders.append(
                f"{label}: {len(found)} string(s) under {keys}, longest {longest} words"
            )

    assert not offenders, (
        "Free text found in results files. Results record measurements; model "
        "generations and corpus excerpts must be reduced to lengths, counts and "
        "hashes before they are written (DECISIONS D30). Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Layer 3: no corpus content anywhere. Needs the corpus.
# ---------------------------------------------------------------------------

#: Length of the token window compared against the corpus. Eight consecutive
#: words matching an essay exactly is not a coincidence; it is a quotation.
_SHINGLE_WORDS = 8

_WORD = re.compile(r"[a-z0-9']+")


def _shingles(text: str) -> set[str]:
    """Every window of `_SHINGLE_WORDS` consecutive words, normalised.

    Normalising away case, punctuation and line breaks means a leak survives
    being reformatted, re-wrapped, or embedded in a JSON string with escaped
    newlines -- all of which a plain substring search misses.
    """
    words = _WORD.findall(text.lower())
    if len(words) < _SHINGLE_WORDS:
        return set()
    return {" ".join(words[i : i + _SHINGLE_WORDS]) for i in range(len(words) - _SHINGLE_WORDS + 1)}


def _corpus_shingles() -> set[str]:
    """Every window in every processed document, both corpora.

    The previous version of this test sampled 40 component strings out of
    6,089 and matched them as exact substrings. It therefore checked 0.7% of
    the corpus for one specific kind of leak, and missed the leak that
    happened. This checks all of it, including the document text around the
    components, and catches partial quotations rather than only whole ones.
    """
    processed = _repo_root() / "data" / "processed"
    if not processed.exists():
        return set()

    shingles: set[str] = set()
    for path in sorted(processed.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                shingles |= _shingles(str(doc.get("text", "")))
                for component in doc.get("components", []):
                    shingles |= _shingles(str(component.get("text", "")))
    return shingles


@pytest.mark.corpus
def test_no_corpus_text_appears_in_tracked_files() -> None:
    """Belt-and-braces net for corpus text in source, fixtures, docs or results.

    Needs the real corpus to know what to look for, so it is skipped where the
    data is absent. The path and shape guards above run everywhere.
    """
    corpus = _corpus_shingles()
    if not corpus:
        pytest.skip("corpus not built; run `uv run argmap-build-datasets`")

    root = _repo_root()
    offenders: list[str] = []

    for path in _tracked_files():
        full = root / path
        if not full.is_file() or full.suffix.lower() in {".png", ".jpg", ".pdf", ".zip"}:
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        overlap = _shingles(content) & corpus
        if overlap:
            # Count only. Printing a match would leak the very text the test
            # is protecting, into a log that is often public.
            offenders.append(f"{path} ({len(overlap)} matching {_SHINGLE_WORDS}-word spans)")

    assert not offenders, (
        "Corpus text found in tracked files, which the Argument Annotated Essays "
        f"license forbids publishing: {offenders}"
    )


def test_split_files_carry_ids_only() -> None:
    """Split files are committed on purpose; they must contain no corpus text."""

    splits_dir = _repo_root() / "splits"
    if not splits_dir.exists():
        pytest.skip("splits not built yet")

    for path in splits_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) <= {"train", "val", "test"}
        for ids in payload.values():
            for doc_id in ids:
                assert " " not in doc_id, f"{path.name}: {doc_id!r} looks like text, not an ID"
                assert len(doc_id) < 64, f"{path.name}: {doc_id!r} is too long to be an ID"
