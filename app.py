import json
import os
import re
import difflib
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()


# ----------------------------------------------------
# Paths & Logging
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "dream_logs.jsonl")
LEXICON_PATH = os.path.join(BASE_DIR, "symbol_lexicon.json")


def append_log(record: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------
# Lexicon Loader
# ----------------------------------------------------

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
DREAM_KEYWORDS = sorted(DREAM_SYMBOL_LEXICON.keys())


# ----------------------------------------------------
# Synonyms & Normalization
# ----------------------------------------------------

SYNONYMS = {
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
# Conceptual Motif Detector
# ----------------------------------------------------

def detect_keywords(text: str) -> List[str]:
    lowered = text.lower()
    found = []
    seen = set()

    tokens = re.findall(r"[a-zA-Z']+", lowered)
    normalized_tokens = [normalize_word(tok) for tok in tokens]

    # 1) Direct matches
    for nt in normalized_tokens:
        if nt in DREAM_KEYWORDS and nt not in seen:
            found.append(nt)
            seen.add(nt)

    # 2) Multiword motifs
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered and kw not in seen:
            found.append(kw)
            seen.add(kw)

    # 3) Fuzzy matches
    for tok in tokens:
        close = difflib.get_close_matches(tok, DREAM_KEYWORDS, n=1, cutoff=0.82)
        if close:
            key = close[0]
            if key not in seen:
                found.append(key)
                seen.add(key)

    # 4) Conceptual cues
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
# Symbol Candidate Extraction & Priority Scoring
# ----------------------------------------------------

def simple_candidate_symbols(text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    stop = {"the", "and", "this", "that", "from", "into", "onto", "just", "like"}

    counts = {}
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
        phrase = c["phrase"].lower()
        norm = normalize_word(phrase)
        lex = DREAM_SYMBOL_LEXICON.get(norm)
        score = c["count"] * 2 + (4 if lex else 0)

        scored.append({
            "symbol": norm,
            "original": phrase,
            "local_count": c["count"],
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
... (prompt unchanged for brevity)
"""


# ----------------------------------------------------
# LLM Execution
# ----------------------------------------------------

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
            "summary": "Parsing error.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "emotional_arc": [],
            "narrative_pattern": {},
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": []
        }

    profile = data.get("emotional_profile", {})
    primary = profile.get("primary_emotions", [])
    tone = profile.get("overall_tone", "unknown")

    return {
        "micronarrative": data.get("micronarrative", ""),
        "summary": data.get("summary", ""),
        "interpretive_narrative": data.get("interpretive_narrative", ""),
        "key_symbols": data.get("key_symbols", []),
        "emotional_profile_primary": primary,
        "emotional_profile_tone": tone,
        "emotional_arc": data.get("emotional_arc", []),
        "narrative_pattern": data.get("narrative_pattern", {}),
        "symbol_relations": data.get("symbol_relations", []),
        "reflection_prompts": data.get("reflection_prompts", []),
        "cautions": data.get("cautions", []),
        "detected_keywords": detected
    }


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
        return render_template("index.html", error="Enter a dream to decode.")

    try:
        analysis = analyze_dream(
            dream_text=dream_text,
            title=dream_title,
            felt_during=felt_during,
            felt_after=felt_after,
            life_context=life_context
        )
    except Exception as e:
        return f"<h1>Error</h1><p>{e}</p>", 500

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
