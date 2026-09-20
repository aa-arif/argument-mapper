"""Shared fixtures.

Every fixture here is synthetic. The real corpora are licensed and gitignored,
so CI must be able to run the whole suite without them; tests that genuinely
need real data are marked `corpus` and skipped when it is absent.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from argmap.schema import Component, Document, Relation

# "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
#  0     6    11      18    24      32    38  42    48
# tokens:  0=Alpha 1=beta 2=gamma 3=Delta 4=epsilon 5=zeta 6=Eta 7=theta 8=iota
FIXTURE_TEXT = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."


@pytest.fixture
def fixture_text() -> str:
    return FIXTURE_TEXT


@pytest.fixture
def gold_doc() -> Document:
    """Three components, one relation. Spans verified against FIXTURE_TEXT."""
    return Document(
        doc_id="fixture001",
        source="aae-v2",
        split="train",
        text=FIXTURE_TEXT,
        components=[
            Component(id="T1", type="Premise", start=0, end=17, text="Alpha beta gamma."),
            Component(id="T2", type="Claim", start=18, end=37, text="Delta epsilon zeta."),
            Component(id="T3", type="MajorClaim", start=38, end=53, text="Eta theta iota."),
        ],
        relations=[Relation(id="R1", src="T1", tgt="T2", type="supports")],
    )


# ---------------------------------------------------------------------------
# Synthetic corpora, shaped exactly like the real distributions
# ---------------------------------------------------------------------------

_ESSAY_TEXT = "Title here\n\nAlpha beta gamma. Delta epsilon zeta. Eta theta iota.\n"
#              0123456789 10 11  12

_ESSAY_ANN = (
    "T1\tMajorClaim 12 29\tAlpha beta gamma.\n"
    "T2\tClaim 30 49\tDelta epsilon zeta.\n"
    "A1\tStance T2 For\n"
    "T3\tPremise 50 65\tEta theta iota.\n"
    "R1\tsupports Arg1:T3 Arg2:T2\t\n"
)


def _build_inner_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for essay_id in ("essay001", "essay002"):
            zf.writestr(f"brat-project-final/{essay_id}.txt", _ESSAY_TEXT)
            zf.writestr(f"brat-project-final/{essay_id}.ann", _ESSAY_ANN)
        # macOS resource forks appear in the real distribution and must be ignored
        zf.writestr("__MACOSX/brat-project-final/._essay001.txt", "junk")
    return buf.getvalue()


@pytest.fixture
def aae_zip(tmp_path: Path) -> Path:
    """A miniature stand-in for the Argument Annotated Essays distribution."""
    path = tmp_path / "ArgumentAnnotatedEssays-2.0.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "ArgumentAnnotatedEssays-2.0/train-test-split.csv",
            '"ID";"SET"\n"essay001";"TRAIN"\n"essay002";"TEST"\n',
        )
        zf.writestr("ArgumentAnnotatedEssays-2.0/brat-project-final.zip", _build_inner_zip())
    return path


MICROTEXT_XML = """<?xml version='1.0' encoding='UTF-8'?>
<arggraph id="micro_t001" topic_id="fixture" stance="pro">
  <edu id="e1"><![CDATA[Alpha beta gamma.]]></edu>
  <edu id="e2"><![CDATA[Delta epsilon zeta.]]></edu>
  <edu id="e3"><![CDATA[Eta theta iota.]]></edu>
  <adu id="a1" type="opp"/>
  <adu id="a2" type="pro"/>
  <adu id="a3" type="pro"/>
  <edge id="s1" src="e1" trg="a1" type="seg"/>
  <edge id="s2" src="e2" trg="a2" type="seg"/>
  <edge id="s3" src="e3" trg="a3" type="seg"/>
  <edge id="c1" src="a1" trg="a2" type="reb"/>
  <edge id="c2" src="a3" trg="c1" type="und"/>
</arggraph>
"""


@pytest.fixture
def microtext_xml() -> str:
    return MICROTEXT_XML
