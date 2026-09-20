"""
Philosophy Argument Mapper — Claude Sonnet Backend
Run with: python server.py
Requires: pip install fastapi uvicorn anthropic
Set ANTHROPIC_API_KEY in your environment.
"""

import difflib
import json
import os
import re
from typing import List, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ─── App Setup ───────────────────────────────────────────────────────────
app = FastAPI(title="Argument Mapper API")

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Claude Client ───────────────────────────────────────────────────────
MODEL_ID = "claude-sonnet-4-6"
client = anthropic.Anthropic()

# ─── Constants ───────────────────────────────────────────────────────────
VALID_CLAIM_TYPES = ("premise", "conclusion", "objection", "assumption", "weakness", "hidden_assumption")
VALID_REL_TYPES = ("supports", "opposes", "refines", "depends_on")
VALID_REASONING_TYPES = ("deductive", "inductive", "abductive", "analogical")
VALID_FRAMING_VALUES = ("neutral", "loaded", "one_sided", "balanced")

# ─── Data Models ─────────────────────────────────────────────────────────
class ExtractionRequest(BaseModel):
    text: str
    max_tokens: int = 8000

class Claim(BaseModel):
    id: str
    claim: str
    type: str
    sourceStart: int
    sourceEnd: int
    confidence: float = 0.5

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    strength: float = 0.5
    reasoning_type: str = "deductive"

class BiasAssessment(BaseModel):
    framing: str = "neutral"
    dominant_perspective: str = ""
    underrepresented_perspectives: List[str] = []

class ExtractionResponse(BaseModel):
    claims: List[Claim]
    relationships: List[Relationship]
    bias_assessment: Optional[BiasAssessment] = None

# ─── Extraction Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert philosophy argument analyst. Given a philosophical text, extract the argument structure by identifying:

1. **Claims**: Each distinct claim, premise, conclusion, objection, assumption, weakness, or hidden assumption in the text.
2. **Relationships**: How claims relate to each other (supports, opposes, refines, depends_on).
3. **Bias Assessment**: An overall assessment of argumentative bias in the text.

For each claim, you MUST provide:
- `id`: A short unique identifier (c1, c2, c3, etc.)
- `claim`: The claim stated clearly in one sentence
- `type`: One of "premise", "conclusion", "objection", "assumption", "weakness", or "hidden_assumption"
  - "weakness": a logical vulnerability or unsupported leap you identify in the argument
  - "hidden_assumption": a premise the text relies on but never states explicitly
- `sourceStart`: The character index where this claim begins in the original text
- `sourceEnd`: The character index where this claim ends in the original text
- `confidence`: A float from 0.0 to 1.0 representing how certain you are this claim is genuinely present and correctly typed

CRITICAL RULES for sourceStart and sourceEnd:
- These must be exact character positions in the original text
- The text between sourceStart and sourceEnd should be the passage that expresses this claim
- Source ranges must NOT overlap with each other
- Every range must be within the bounds of the text length
- For "hidden_assumption" and "weakness" types, use the source range of the most relevant passage

For each relationship:
- `source`: The id of the claim that does the supporting/opposing
- `target`: The id of the claim being supported/opposed
- `type`: One of "supports", "opposes", "refines", "depends_on"
- `strength`: A float from 0.0 to 1.0 indicating how strong this relationship is
- `reasoning_type`: One of "deductive", "inductive", "abductive", "analogical"

For the bias assessment:
- `framing`: One of "neutral", "loaded", "one_sided", "balanced"
- `dominant_perspective`: A one-sentence description of whose viewpoint is centered
- `underrepresented_perspectives`: A list of 0–3 strings naming perspectives absent from the argument

Respond with ONLY valid JSON in this exact format, no other text:
{
  "claims": [
    {"id": "c1", "claim": "...", "type": "premise", "sourceStart": 0, "sourceEnd": 100, "confidence": 0.9},
    {"id": "c2", "claim": "...", "type": "conclusion", "sourceStart": 101, "sourceEnd": 200, "confidence": 0.85}
  ],
  "relationships": [
    {"source": "c1", "target": "c2", "type": "supports", "strength": 0.8, "reasoning_type": "deductive"}
  ],
  "bias_assessment": {
    "framing": "neutral",
    "dominant_perspective": "The author argues from a Kantian deontological perspective.",
    "underrepresented_perspectives": ["consequentialist viewpoint", "virtue ethics perspective"]
  }
}"""

def build_user_prompt(text: str) -> str:
    return f"""Analyze the following philosophical text and extract its argument structure.

TEXT (total length: {len(text)} characters):
\"\"\"
{text}
\"\"\"

Extract all claims and relationships. Remember:
- sourceStart and sourceEnd must be valid character indices within the text
- Source ranges should not overlap
- Include confidence scores for each claim
- Include strength and reasoning_type for each relationship
- Include a bias_assessment for the overall text
- Identify any weaknesses or hidden assumptions in the argument
- Respond with ONLY the JSON object, nothing else"""

# ─── JSON Parsing Helper ─────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Try to extract JSON from the model's response, handling common issues."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: output was likely truncated mid-JSON. Strip any code-fence
    # opener, then walk back to the last complete claim/relationship and close
    # braces so we at least render a partial argument map.
    candidate = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
    last_obj_end = candidate.rfind('}')
    if last_obj_end != -1:
        truncated = candidate[:last_obj_end + 1]
        for suffix in (']}', '}]}', ']}]'):
            try:
                return json.loads(truncated + suffix)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not extract valid JSON from model response")

# ─── Fuzzy Source Matching ───────────────────────────────────────────────
def find_claim_position(claim_text, source_text):
    """Find the best matching position of claim_text within source_text."""
    pos = source_text.find(claim_text)
    if pos != -1:
        return pos, pos + len(claim_text)

    claim_len = len(claim_text)
    best_ratio = 0.0
    best_start = 0
    best_end = claim_len
    source_lower = source_text.lower()
    claim_lower = claim_text.lower()

    for scale in (1.0, 1.2, 0.8, 1.5, 0.6):
        ws = max(10, int(claim_len * scale))
        step = max(1, ws // 4)
        for i in range(0, max(1, len(source_text) - ws + 1), step):
            window = source_lower[i:i + ws]
            ratio = difflib.SequenceMatcher(None, claim_lower, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = min(i + ws, len(source_text))

    if best_ratio > 0.4:
        return best_start, best_end

    return None, None

# ─── Validation Helper ───────────────────────────────────────────────────
def validate_and_fix(data: dict, text: str) -> dict:
    """Validate and fix common issues with the model's output."""
    text_length = len(text)
    claims = data.get("claims", [])
    relationships = data.get("relationships", [])

    valid_claim_ids = set()
    for claim in claims:
        claim_text = claim.get("claim", "")

        matched_start, matched_end = find_claim_position(claim_text, text)
        if matched_start is not None:
            claim["sourceStart"] = matched_start
            claim["sourceEnd"] = matched_end
        else:
            claim["sourceStart"] = max(0, min(claim.get("sourceStart", 0), text_length - 1))
            claim["sourceEnd"] = max(claim["sourceStart"] + 1, min(claim.get("sourceEnd", text_length), text_length))

        valid_claim_ids.add(claim["id"])

        if claim.get("type") not in VALID_CLAIM_TYPES:
            claim["type"] = "premise"

        confidence = claim.get("confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            claim["confidence"] = 0.5
        else:
            claim["confidence"] = max(0.0, min(1.0, float(confidence)))

    valid_rels = []
    for rel in relationships:
        if rel.get("source") in valid_claim_ids and rel.get("target") in valid_claim_ids:
            if rel.get("type") not in VALID_REL_TYPES:
                rel["type"] = "supports"

            strength = rel.get("strength")
            if strength is None or not isinstance(strength, (int, float)):
                rel["strength"] = 0.5
            else:
                rel["strength"] = max(0.0, min(1.0, float(strength)))

            if rel.get("reasoning_type") not in VALID_REASONING_TYPES:
                rel["reasoning_type"] = "deductive"

            valid_rels.append(rel)

    bias = data.get("bias_assessment")
    if isinstance(bias, dict):
        if bias.get("framing") not in VALID_FRAMING_VALUES:
            bias["framing"] = "neutral"
        if not isinstance(bias.get("dominant_perspective"), str):
            bias["dominant_perspective"] = ""
        persp = bias.get("underrepresented_perspectives")
        if not isinstance(persp, list):
            bias["underrepresented_perspectives"] = []
        else:
            bias["underrepresented_perspectives"] = [str(p) for p in persp[:3]]
    else:
        bias = {
            "framing": "neutral",
            "dominant_perspective": "",
            "underrepresented_perspectives": [],
        }

    return {"claims": claims, "relationships": valid_rels, "bias_assessment": bias}

# ─── Routes ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/extract", response_model=ExtractionResponse)
def extract_arguments(req: ExtractionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(req.text) > 50000:
        raise HTTPException(status_code=400, detail="Text too long (max 50,000 characters)")

    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=req.max_tokens,
            temperature=0.2,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": build_user_prompt(req.text)}],
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Claude API error: {e.message}")
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Claude API: {e}")

    response_text = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    ).strip()

    try:
        parsed = extract_json(response_text)
        return validate_and_fix(parsed, req.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model output: {str(e)}\n\nRaw output:\n{response_text[:1000]}",
        )

# ─── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
