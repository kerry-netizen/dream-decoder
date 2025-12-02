import json
import os
import re
import difflib
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # Uses OPENAI_API_KEY


# ----------------------------------------------------
# Paths & Logging
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "dream_logs.jsonl")
LEXICON_PATH = os.path.join(BASE_DIR, "symbol_lexicon.json")


def append_log(record: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_logs() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        return []
    out = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# ----------------------------------------------------
# Lexicon Loader
# ----------------------------------------------------

def load_lexicon() -> Dict[str, Dict[str, Any]]:
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data:
                return data
    except Exception:
        pass
    return {}


DREAM_SYMBOL_LEXICON = load_lexicon()
if not DREAM_SYMBOL_LEXICON:
    raise RuntimeError("symbol_lexicon.json failed to load or is empty.")

DREAM_KEYWORDS: List[str] = sorted(DREAM_SYMBOL_LEXICON.keys())


# ----------------------------------------------------
# Synonyms & Normalization
# ----------------------------------------------------

SYNONYMS: Dict[str, str] = {
    "letters": "letter",
    "messages": "letter",
    "message": "letter",
    "starlit": "starlight",
    "stars": "starlight",
    "star": "starlight",
    "constellations": "constellation",
    "guides": "guide",
    "guardian": "guide",
    "mentor": "guide",
    "glass": "glass",
    "shattered": "cracked glass",
    "shatter": "cracked glass",
    "burn": "fire",
    "burned": "fire",
    "burning": "fire",
    "galaxy": "galaxy face",
    "cosmic": "galaxy face",
    "water": "ocean",
    "flood": "ocean",
    "fall": "falling",
    "falling": "falling",
    "tilt": "gravity shift",
    "tilting": "gravity shift",
    "gravity": "gravity shift",
    "museum": "museum",
    "bridge": "bridge"
}


def normalize_word(word: str) -> str:
    w = word.lower().strip()
    if w in SYNONYMS:
        return SYNONYMS[w]
    if w.endswith("s") and w[:-1] in DREAM_KEYWORDS:
        return w[:-1]
    return w


# ----------------------------------------------------
# Conceptual Motif Detector (Mode A)
# ----------------------------------------------------

def detect_keywords(text: str) -> List[str]:
    """
    High-sensitivity motif detector:
    - matches lexicon keys directly
    - normalizes synonyms & plurals
    - fuzzy matches similar terms
    - adds conceptual motifs from contextual cues
    """
    lowered = text.lower()
    found: List[str] = []
    seen: set[str] = set()

    # Tokenize
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    normalized_tokens = [normalize_word(tok) for tok in tokens]

    # 1) Direct matches via normalized tokens
    for nt in normalized_tokens:
        if nt in DREAM_KEYWORDS and nt not in seen:
            found.append(nt)
            seen.add(nt)

    # 2) Multiword motifs (exact phrase)
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered and kw not in seen:
            found.append(kw)
            seen.add(kw)

    # 3) Fuzzy lexical matches (spell / variant softness)
    for tok in tokens:
        close = difflib.get_close_matches(tok, DREAM_KEYWORDS, n=1, cutoff=0.82)
        if close:
            key = close[0]
            if key not in seen:
                found.append(key)
                seen.add(key)

    # 4) Conceptual cues (associative triggers)
    cues = {
        "fall": "falling",
        "tilt": "gravity shift",
        "shatter": "cracked glass",
        "collapse": "fire",
        "burn": "fire",
        "glow": "starlight",
        "star": "starlight",
        "mirror": "mirror",
        "child": "child",
        "door": "door",
        "guide": "guide",
        "galaxy": "galaxy face",
        "museum": "museum",
        "letter": "letter",
        "ocean": "ocean",
        "water": "ocean",
        "bridge": "bridge"
    }

    for cue, motif in cues.items():
        if cue in lowered and motif in DREAM_KEYWORDS and motif not in seen:
            found.append(motif)
            seen.add(motif)

    return found


# ----------------------------------------------------
# Symbol Candidates & Priority Scoring
# ----------------------------------------------------

def simple_candidate_symbols(text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    stop = {
        "the", "and", "this", "that", "from", "into", "onto",
        "just", "like", "your", "with", "have", "then"
    }

    counts: Dict[str, int] = {}
    for tok in tokens:
        if len(tok) < 4 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    items = [{"phrase": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: (-x["count"], -len(x["phrase"])))
    return items[:max_items]


def build_priority_symbols(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        phrase = c.get("phrase", "").lower()
        norm = normalize_word(phrase)
        lex = DREAM_SYMBOL_LEXICON.get(norm)
        score = c.get("count", 0) * 2 + (4 if lex else 0)

        scored.append({
            "symbol": norm,
            "original": phrase,
            "local_count": c.get("count", 0),
            "in_lexicon": bool(lex),
            "score": score,
            "lexicon_themes": (lex or {}).get("themes", [])
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:10]


# ----------------------------------------------------
# LLM Prompt
# ----------------------------------------------------

SYSTEM_PROMPT = """
You are Dream Decoder, an evidence-informed, non-mystical interpreter of dreams.

Your job is not just to list symbols, but to help the dreamer understand how
those symbols, emotions, and situations might fit together into a psychological
story about where they are right now.

You will receive:
- "dream_text"
- "felt_during", "felt_after"
- "life_context"
- "detected_keywords"
- "candidate_symbols"
- "priority_symbols"

Principles:
- Stay grounded in the dream's actual content and the life_context.
- Make interpretations tentative: use language like "may suggest" or "could reflect".
- Avoid superstition, fortune-telling, or claims about the future.
- Tie symbols to emotional and situational themes, not fixed meanings.
- Help the dreamer generate insight, not fear.

Return VALID JSON with:

{
  "micronarrative": "...",
  "summary": "...",
  "interpretive_narrative": "...",
  "key_symbols": [
    {
      "symbol": "string",
      "description": "string",
      "possible_meanings": ["string", ...],
      "confidence": 0.0
    }
  ],
  "emotional_profile": {
    "primary_emotions": [
      {"name": "string", "intensity": 0.0}
    ],
    "overall_tone": "string"
  },
  "emotional_arc": [
    {"stage": "beginning/middle/end", "emotion": "string", "intensity": 0.0}
  ],
  "narrative_pattern": {
    "pattern_name": "string",
    "description": "string",
    "related_themes": ["string", ...]
  },
  "symbol_relations": [
    {"source": "string", "target": "string", "relation": "string"}
  ],
  "reflection_prompts": ["string", ...],
  "cautions": ["string", ...]
}

All intensities are floats 0–1. Keep lists short (3–7 items).
"""


# ----------------------------------------------------
# LLM Execution (4o + silent fallback to 4o-mini)
# ----------------------------------------------------

def call_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call gpt-4o with timeout + silent fallback to gpt-4o-mini.
    Always returns something JSON-parseable or raises.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    # Primary attempt: gpt-4o
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.6,
            timeout=8,  # critical for Render free tier
        )
        raw = completion.choices[0].message.content
        return json.loads(raw)
    except Exception:
        # Silent fallback: gpt-4o-mini
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.6,
            timeout=8,
        )
        raw = completion.choices[0].message.content
        return json.loads(raw)


def analyze_dream(
    dream_text: str,
    title: str = "",
    felt_during: str = "",
    felt_after: str = "",
    life_context: str = ""
) -> Dict[str, Any]:

    detected = detect_keywords(dream_text)
    candidates = simple_candidate_symbols(dream_text)
    priority = build_priority_symbols(candidates)

    payload = {
        "dream_title": title,
        "dream_text": dream_text,
        "felt_during": felt_during,
        "felt_after": felt_after,
        "life_context": life_context,
        "detected_keywords": detected,
        "candidate_symbols": candidates,
        "priority_symbols": priority,
    }

    try:
        data = call_model(payload)
    except Exception:
        data = {
            "micronarrative": "",
            "summary": "There was an error contacting the model.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "emotional_arc": [],
            "narrative_pattern": {
                "pattern_name": "",
                "description": "",
                "related_themes": [],
            },
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": ["Model call failed."],
        }

    emotional_profile = data.get("emotional_profile", {}) or {}
    primary_emotions = emotional_profile.get("primary_emotions", []) or []
    overall_tone = emotional_profile.get("overall_tone", "unknown") or "unknown"

    analysis: Dict[str, Any] = {
        "micronarrative": data.get("micronarrative", "") or "",
        "summary": data.get("summary", "") or "",
        "interpretive_narrative": data.get("interpretive_narrative", "") or "",
        "key_symbols": data.get("key_symbols", []) or [],
        "emotional_profile_primary": primary_emotions,
        "emotional_profile_tone": overall_tone,
        "emotional_arc": data.get("emotional_arc", []) or [],
        "narrative_pattern": data.get("narrative_pattern", {}) or {
            "pattern_name": "",
            "description": "",
            "related_themes": [],
        },
        "symbol_relations": data.get("symbol_relations", []) or [],
        "reflection_prompts": data.get("reflection_prompts", []) or [],
        "cautions": data.get("cautions", []) or [],
        "detected_keywords": detected,
    }

    return analysis


# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return handle_decode()
    return render_template("index.html")


@app.route("/decode", methods=["GET", "POST"])
def decode():
    if request.method == "POST":
        return handle_decode()
    return redirect(url_for("index"))


def handle_decode():
    dream_text = request.form.get("dream_text", "").strip()
    dream_title = request.form.get("dream_title", "").strip()
    felt_during = request.form.get("felt_during", "").strip()
    felt_after = request.form.get("felt_after", "").strip()
    life_context = request.form.get("life_context", "").strip()

    if not dream_text:
        return render_template("index.html", error="Please paste a dream before decoding.")

    try:
        analysis = analyze_dream(
            dream_text=dream_text,
            title=dream_title,
            felt_during=felt_during,
            felt_after=felt_after,
            life_context=life_context,
        )
    except Exception as e:
        return f"<h1>Decode error</h1><p><strong>{type(e).__name__}</strong>: {e}</p>", 500

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": {
            "title": dream_title,
            "dream_text": dream_text,
            "felt_during": felt_during,
            "felt_after": felt_after,
            "life_context": life_context,
        },
        "analysis": analysis,
    }
    append_log(record)

    return render_template("result.html", analysis=analysis)


if __name__ == "__main__":
    app.run(debug=True)
