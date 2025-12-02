import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # Uses OPENAI_API_KEY

# ---------------------------
# Paths & Logging
# ---------------------------

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


# ---------------------------
# Load Lexicon
# ---------------------------

def load_lexicon() -> Dict[str, Dict[str, Any]]:
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


DREAM_SYMBOL_LEXICON = load_lexicon()

if not DREAM_SYMBOL_LEXICON:
    raise RuntimeError("symbol_lexicon.json failed to load or is empty.")


# Extract canonical motif keys
DREAM_KEYWORDS: List[str] = sorted(DREAM_SYMBOL_LEXICON.keys())


# ---------------------------
# Semantic Synonyms
# ---------------------------
# This improves detection for terms that appear in dreams but not as lexicon keys.
SYNONYMS: Dict[str, str] = {
    "letters": "letter",
    "message": "letter",
    "messages": "letter",
    "starlit": "starlight",
    "stars": "starlight",
    "star": "starlight",
    "guides": "guide",
    "guardian": "guide",
    "mentor": "guide",
    "museum": "museum",
    "glass": "glass",
    "cracked": "cracked glass",
    "alarm": "alarm",
    "bridge": "bridge",
    "burning bridge": "fire",
    "gravity": "gravity shift",
    "tilting": "gravity shift",
    "falling": "falling",
    "frozen": "child",
    "younger self": "child",
    "older self": "mirror",
    "galaxy": "galaxy face",
    "cosmic face": "galaxy face",
    "ocean": "ocean"
}


def normalize_word(word: str) -> str:
    """Apply synonym or plurals -> singular normalization."""
    w = word.lower().strip()

    # If it's an exact synonym, map it
    if w in SYNONYMS:
        return SYNONYMS[w]

    # Remove trailing plural 's'
    if w.endswith("s") and w[:-1] in DREAM_KEYWORDS:
        return w[:-1]

    return w


# ---------------------------
# Improved Semantic Keyword Detector
# ---------------------------

def detect_keywords(text: str) -> List[str]:
    """
    Powerful motif detector:
    - matches whole words OR semantic relatives
    - plural aware
    - multiword phrase support
    - lexicon-driven scanning
    """
    lowered = text.lower()
    found: List[str] = []

    # Token-level detection
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    for tok in tokens:
        key = normalize_word(tok)
        if key in DREAM_KEYWORDS:
            found.append(key)

    # Phrase-level detection (e.g., "burning bridge", "cracked glass")
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered:
            found.append(kw)

    # Deduplicate while preserving order
    seen = set()
    cleaned = []
    for k in found:
        if k not in seen:
            cleaned.append(k)
            seen.add(k)

    return cleaned


# ---------------------------
# Candidate Symbol Extraction
# ---------------------------

def simple_candidate_symbols(text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    stop = {
        "the", "and", "with", "this", "that", "from", "into", "onto", "then", "were",
        "have", "about", "while", "when", "just", "like", "over", "under", "through",
        "as", "for", "but", "not", "you", "your", "they", "them"
    }

    counts: Dict[str, int] = {}
    for tok in tokens:
        if len(tok) < 4 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    items = [{"phrase": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: (-x["count"], -len(x["phrase"])))
    return items[:max_items]


# ---------------------------
# Priority Symbol Builder
# ---------------------------

def build_priority_symbols(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        phrase = c.get("phrase", "")
        normalized = normalize_word(phrase)
        lex = DREAM_SYMBOL_LEXICON.get(normalized)

        in_lex = 1 if lex else 0
        local_count = c.get("count", 0)
        score = local_count * 2 + in_lex * 4

        scored.append({
            "symbol": normalized,
            "original": phrase,
            "local_count": local_count,
            "in_lexicon": bool(in_lex),
            "score": score,
            "lexicon_themes": (lex or {}).get("themes", [])
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:10]


# ---------------------------
# Model Prompt
# ---------------------------

SYSTEM_PROMPT = """
You are Dream Decoder, an evidence-informed, non-mystical interpreter of dreams.

Your job is not just to list symbols, but to help the dreamer understand how
those symbols, emotions, and situations might fit together into a psychological
story about where they are right now.

You will receive:
- dream_text
- felt_during, felt_after
- life_context
- detected_keywords
- candidate_symbols
- priority_symbols

Principles:
- Stay grounded in the dream's actual content.
- Make interpretations tentative ("may suggest", "could reflect").
- Avoid superstition or fortune-telling.
- Focus on psychological meaning, emotional patterns, and internal dynamics.
- Help generate insight, not fear.

Return VALID JSON with:
{
  "micronarrative": "...",
  "summary": "...",
  "interpretive_narrative": "...",
  "key_symbols": [...],
  "emotional_profile": {...},
  "emotional_arc": [...],
  "narrative_pattern": {...},
  "symbol_relations": [...],
  "reflection_prompts": [...],
  "cautions": [...]
}
"""


# ---------------------------
# Model Execution
# ---------------------------

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
        "priority_symbols": priority
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ],
        response_format={"type": "json_object"},
        temperature=0.6
    )

    raw = completion.choices[0].message.content

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "micronarrative": "",
            "summary": "There was an error parsing model output.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "emotional_arc": [],
            "narrative_pattern": {"pattern_name": "", "description": "", "related_themes": []},
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": ["Model output could not be parsed."]
        }

    emotional_profile = data.get("emotional_profile", {}) or {}
    primary_emotions = emotional_profile.get("primary_emotions", []) or []
    tone = emotional_profile.get("overall_tone", "unknown") or "unknown"

    analysis = {
        "micronarrative": data.get("micronarrative", "") or "",
        "summary": data.get("summary", "") or "",
        "interpretive_narrative": data.get("interpretive_narrative", "") or "",
        "key_symbols": data.get("key_symbols", []) or [],
        "emotional_profile_primary": primary_emotions,
        "emotional_profile_tone": tone,
        "emotional_arc": data.get("emotional_arc", []) or [],
        "narrative_pattern": data.get("narrative_pattern", {}) or {},
        "symbol_relations": data.get("symbol_relations", []) or [],
        "reflection_prompts": data.get("reflection_prompts", []) or [],
        "cautions": data.get("cautions", []) or [],
        "detected_keywords": detected
    }

    return analysis


# ---------------------------
# Routes
# ---------------------------

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
            life_context=life_context
        )
    except Exception as e:
        return f"<h1>Decode Error</h1><p><b>{type(e).__name__}</b>: {e}</p>", 500

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": {
            "title": dream_title,
            "dream_text": dream_text,
            "felt_during": felt_during,
            "felt_after": felt_after,
            "life_context": life_context
        },
        "analysis": analysis
    }

    append_log(record)
    return render_template("result.html", analysis=analysis)


if __name__ == "__main__":
    app.run(debug=True)
