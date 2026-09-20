"""Guard against ever committing licensed corpus data.

Argument Annotated Essays v2 is distributed under an agreement whose clause 2.2
forbids publishing, redistributing or displaying the data in any form, and this
repository is public. `.gitignore` is a convention that a single `git add -f`
defeats; this test is checked in CI and fails the build instead.

It inspects what git actually tracks, not what is on disk, so a developer with
the corpora present locally still passes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

#: Paths and patterns that would indicate corpus data has been committed.
_FORBIDDEN_SUFFIXES = (".ann",)
_FORBIDDEN_SUBSTRINGS = (
    "argumentannotatedessays",
    "brat-project",
    "arg-microtexts",
)
_FORBIDDEN_PREFIXES = ("data/",)

#: How many distinct sentences to sample from the corpus as leak sentinels.
_SENTINEL_COUNT = 40

#: Minimum sentinel length, so short generic clauses do not false-positive.
_SENTINEL_MIN_CHARS = 45


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


def _sentinels() -> list[str]:
    """Sample real corpus sentences to search for.

    Deliberately derived from the corpus at runtime rather than hardcoded. A
    hardcoded sentinel would itself be corpus text living in a tracked file --
    which is both the thing being guarded against and a guaranteed self-match.
    """
    processed = _repo_root() / "data" / "processed" / "aae.jsonl"
    if not processed.exists():
        return []

    phrases: list[str] = []
    with processed.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            for component in doc.get("components", []):
                text = str(component.get("text", "")).strip()
                if len(text) >= _SENTINEL_MIN_CHARS:
                    phrases.append(text)
                    break
            if len(phrases) >= _SENTINEL_COUNT:
                break
    return phrases


@pytest.mark.corpus
def test_no_corpus_text_appears_in_tracked_files() -> None:
    """Belt-and-braces net for corpus text pasted into source, fixtures or docs.

    Needs the real corpus to know what to look for, so it is skipped where the
    data is absent. The path-based guard above, and the equivalent CI job, run
    everywhere and do not need the data.
    """
    sentinels = _sentinels()
    if not sentinels:
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
        if any(phrase in content for phrase in sentinels):
            offenders.append(path)

    assert not offenders, f"corpus text found in tracked files: {offenders}"


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
