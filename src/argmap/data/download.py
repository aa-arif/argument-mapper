"""Fetching the corpora.

The two corpora need very different handling.

**arg-microtexts** is a public GitHub repository under CC BY-NC-SA 4.0 and can
be fetched directly.

**Argument Annotated Essays v2** cannot. TUdatalib sits behind an Anubis
proof-of-work challenge that returns an HTML page instead of the archive to any
non-browser client, so there is no headless download path -- with or without a
browser User-Agent, the response is a 7.5 KB challenge page. The corpus also
ships with a signature-form license restricting it to academic use and
forbidding redistribution, which makes mirroring it here the wrong answer even
if it were technically possible. A human fetches it once in a browser; this
module verifies what they dropped in place.

Nothing here writes outside `data/`, which is gitignored.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: SHA-256 of the distributed ArgumentAnnotatedEssays-2.0.zip.
AAE_SHA256 = "7d592ccdebfcce580b2d3d9bb53c93da691cbb3ac2741e8d403a7d1599e60de7"

AAE_FILENAME = "ArgumentAnnotatedEssays-2.0.zip"
AAE_LANDING_PAGE = "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422"

MICROTEXTS_URL = "https://github.com/peldszus/arg-microtexts/archive/refs/heads/master.zip"
MICROTEXTS_FILENAME = "arg-microtexts-master.zip"

_AAE_INSTRUCTIONS = f"""
The Argument Annotated Essays v2 archive is not present and cannot be
downloaded automatically.

  1. Open {AAE_LANDING_PAGE}
  2. Clear the "Making sure you're not a bot!" challenge
  3. Download {AAE_FILENAME}
  4. Place it at: {{target}}

It is then verified against SHA-256 {AAE_SHA256}.

The corpus is licensed for academic use and may not be redistributed, so it is
never committed to this repository and never fetched from a mirror.
""".strip()


class CorpusUnavailableError(RuntimeError):
    """Raised when a corpus is missing and cannot be fetched automatically."""


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    sha256: str
    downloaded: bool


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_aae(raw_dir: Path) -> CorpusFile:
    """Verify a human-supplied AAE archive, with actionable instructions if absent."""
    target = raw_dir / AAE_FILENAME
    if not target.exists():
        raise CorpusUnavailableError(_AAE_INSTRUCTIONS.format(target=target.resolve()))

    digest = sha256_of(target)
    if digest != AAE_SHA256:
        msg = (
            f"{target} does not match the expected distribution.\n"
            f"  expected SHA-256 {AAE_SHA256}\n"
            f"  found    SHA-256 {digest}\n"
            "Re-download it from the landing page rather than using this file."
        )
        raise CorpusUnavailableError(msg)

    return CorpusFile(path=target, sha256=digest, downloaded=False)


def ensure_microtexts(raw_dir: Path, *, force: bool = False) -> CorpusFile:
    """Download the arg-microtexts repository archive if it is not already present."""
    target = raw_dir / MICROTEXTS_FILENAME
    downloaded = False

    if force or not target.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(MICROTEXTS_URL, timeout=120) as response:
                payload = response.read()
        except OSError as exc:
            msg = f"could not download arg-microtexts from {MICROTEXTS_URL}: {exc}"
            raise CorpusUnavailableError(msg) from exc

        if not payload.startswith(b"PK"):
            msg = f"{MICROTEXTS_URL} did not return a zip archive"
            raise CorpusUnavailableError(msg)

        target.write_bytes(payload)
        downloaded = True

    return CorpusFile(path=target, sha256=sha256_of(target), downloaded=downloaded)
