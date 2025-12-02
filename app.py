import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # uses OPENAI_API_KEY


# ----------------------------------------------------
# Paths & Logging
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "dream_logs.jsonl")


def append_log(record: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------
# Built-in Symbol Lexicon (no external JSON required)
# ----------------------------------------------------

DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
    "cat": {"themes": ["independence", "intuition"], "notes": "Emotional independence."},
    "dog": {"themes": ["loyalty", "support"], "notes": "Trust and companionship."},
    "snake": {"themes": ["transformation", "fear"], "notes": "Unsettling change."},
    "spider": {"themes": ["entrapment"], "notes": "Feeling caught."},
    "rat": {"themes": ["disgust", "hidden problems"], "notes": "Something unwanted surfacing."},
    "wolf": {"themes": ["instinct"], "notes": "Primal drive or threat."},
    "bear": {"themes": ["power"], "notes": "Emotional or territorial defense."},
    "lion": {"themes": ["pride", "authority"], "notes": "Leadership or dominance."},
    "tiger": {"themes": ["anger"], "notes": "Intense emotion."},
    "bird": {"themes": ["freedom"], "notes": "Desire to rise above issues."},
    "fish": {"themes": ["emotion"], "notes": "Submerged feelings."},
    "shark": {"themes": ["threat"], "notes": "Danger beneath the surface."},
    "horse": {"themes": ["drive"], "notes": "Vitality and momentum."},
    "unicorn": {"themes": ["idealism"], "notes": "Wishfulness."},
    "baby": {"themes": ["new beginnings"], "notes": "Something new forming."},
    "child": {"themes": ["innocence", "past self"], "notes": "Younger aspects of you."},
    "mother": {"themes": ["care"], "notes": "Support or control."},
    "father": {"themes": ["authority"], "notes": "Judgment or structure."},
    "stranger": {"themes": ["unknown"], "notes": "Unknown situations or self-aspects."},
    "guide": {"themes": ["intuition"], "notes": "Inner direction."},
    "faceless guide": {"themes": ["mystery"], "notes": "Guidance without clarity."},
    "house": {"themes": ["self"], "notes": "Your inner life or structure."},
    "forest": {"themes": ["mystery"], "notes": "Exploring the unconscious."},
    "ocean": {"themes": ["depth"], "notes": "Overwhelming emotion."},
    "river": {"themes": ["flow"], "notes": "Life direction."},
    "lake": {"themes": ["containment"], "notes": "Still emotion."},
    "mountain": {"themes": ["aspiration"], "notes": "Challenges and goals."},
    "car": {"themes": ["control"], "notes": "Direction of life."},
    "train": {"themes": ["path"], "notes": "Momentum or routine."},
    "plane": {"themes": ["transition"], "notes": "Big changes."},
    "stairs": {"themes": ["progress"], "notes": "Moving between psychological layers."},
    "door": {"themes": ["opportunity"], "notes": "Thresholds and choices."},
    "window": {"themes": ["perspective"], "notes": "Desire to see differently."},
    "bridge": {"themes": ["transition"], "notes": "Crossing between states."},
    "storm": {"themes": ["conflict"], "notes": "Emotional turbulence."},
    "fire": {"themes": ["destruction", "purification"], "notes": "Burning away the old."},
    "flood": {"themes": ["overwhelm"], "notes": "Emotion rising uncontrollably."},
    "mirror": {"themes": ["identity"], "notes": "Self-reflection."},
    "lost": {"themes": ["confusion"], "notes": "Searching for direction."},
    "trapped": {"themes": ["pressure"], "notes": "Feeling constrained."},

    # Dream Decoder additions
    "museum": {"themes": ["memory"], "notes": "Archive of the self."},
    "letter": {"themes": ["unfinished business"], "notes": "Unspoken communication."},
    "starlight": {"themes": ["guidance"], "notes": "Light in darkness."},
    "constellation": {"themes": ["connection"], "notes": "Seeing patterns."},
    "glass": {"themes": ["clarity"], "notes": "Transparency, vulnerability."},
    "cracked glass": {"themes": ["instability"], "notes": "Strain or breakthrough."},
    "alarm": {"themes": ["urgency"], "notes": "Awareness rising."},
    "gravity shift": {"themes": ["imbalance"], "notes": "Perspective changing."},
    "galaxy face": {"themes": ["mystery"], "notes": "Identity in transformation."}
}

DREAM_KEYWORDS = sorted(DREAM_SYMBOL_LEXICON.keys())


# ----------------------------------------------------
# Synonym Normalization
# ----------------------------------------------------

SYNONYMS = {
    "stars": "starlight",
    "star": "starlight",
    "letters": "letter",
    "messages": "letter",
    "burning": "fire",
    "water": "ocean",
    "flood": "ocean",
    "tilt": "gravity shift",
    "gravity": "gravity shift",
    "shatter": "cracked glass",
    "shattered": "cracked glass",
}


def normalize(word: str) -> str:
    w = word.lower()
    if w in SYNONYMS:
        return SYNONYMS[w]
    if w.endswith("s") and w[:-1] in DREAM_KEYWORDS:
        return w[:-1]
    return w


# ----------------------------------------------------
# High-Sensitivity Motif Detector (Mode A)
# ----------------------------------------------------

def detect_keywords(text: str) -> List[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    found = []
    seen = set()

    # 1. Token-level matching
    for tok in tokens:
        nt = normalize(tok)
        if nt in DREAM_KEYWORDS and nt not in seen:
            found.append(nt)
            seen.add(nt)

    # 2. Phrase-level matching
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered and kw not in seen:
            found.append(kw)
            seen.add(kw)

    # 3. Contextual cues
    cues = {
        "mirror": "mirror",
        "child": "child",
        "door": "door",
        "guide": "guide",
        "museum": "museum",
        "letter": "letter",
        "bridge": "bridge",
        "star": "starlight",
        "glow": "starlight",
        "alarm": "alarm",
        "collapse": "fire",
        "glass": "glass",
    }

    for cue, motif in cues.items():
        if cue in lowered and motif not in seen and motif in DREAM_KEYWORDS:
            found.append(motif)
            seen.add(motif)

    return found


# ----------------------------------------------------
# Candidate Symbols & Priority
# ----------------------------------------------------

def simple_candidate_symbols(text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    stop = {"the", "and", "this", "that", "from", "just", "like", "then", "with"}

    counts = {}
    for tok in tokens:
        if len(tok) < 4 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    items = [{"phrase": k, "count": v} for k, v in counts.items()]
    return sorted(items, key=lambda x: (-x["count"], -len(x["phrase"])))[:max_items]


def build_priority_symbols(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    for c in candidates:
        phrase = c["phrase"].lower()
        norm = normalize(phrase)
        lex = DREAM_SYMBOL_LEXICON.get(norm)
        score = c["count"] * 2 + (4 if lex else 0)

        scored.append({
            "symbol": norm,
            "original": phrase,
            "local_count": c["count"],
            "in_lexicon": bool(lex),
            "score": score,
            "lexicon_themes": lex.get("themes", []) if lex else []
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:10]


# ----------------------------------------------------
# Rich Interpretation Prompt
# ----------------------------------------------------

SYSTEM_PROMPT = """
You are Dream Decoder, an evidence-informed, non-mystical interpreter of dreams.

Your job is to weave symbols, emotions, and situations into a psychologically
meaningful narrative.

Return VALID JSON with:

- micronarrative: 2–4 vivid sentences retelling the dream.
- summary: 3–5 sentences explaining the psychological frame.
- interpretive_narrative: 3–7 paragraphs (400–700+ words) integrating symbols,
  emotions, and context into a cohesive, reflective interpretation.
- key_symbols: 3–7 symbols with explanations + possible meanings + confidence.
- emotional_profile: primary emotions (0–1 intensities).
- emotional_arc: beginning → middle → end.
- narrative_pattern: overarching psychological pattern.
- symbol_relations: how symbols interact.
- reflection_prompts: 3–6 thoughtful questions.
- cautions: gentle reminders to avoid literal interpretation.

Your tone: warm, reflective, insightful, psychologically deep.
Avoid fortune-telling or absolutes. Use phrases like “may reflect” or “could suggest.”
"""


# ----------------------------------------------------
# LLM Call (timeout + silent fallback)
# ----------------------------------------------------

def call_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout=8,
        )
        return json.loads(response.choices[0].message.content)

    except Exception:
        # Silent fallback
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout=8,
        )
        return json.loads(response.choices[0].message.content)


# ----------------------------------------------------
# Dream Analysis
# ----------------------------------------------------

def analyze_dream(
    dream_text: str, title: str = "", felt_during: str = "",
    felt_after: str = "", life_context: str = ""
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
        data = {"summary": "Model error.", "detected_keywords": detected}

    profile = data.get("emotional_profile", {})
    tone = profile.get("overall_tone", "unknown")

    return {
        "micronarrative": data.get("micronarrative", ""),
        "summary": data.get("summary", ""),
        "interpretive_narrative": data.get("interpretive_narrative", ""),
        "key_symbols": data.get("key_symbols", []),
        "emotional_profile_primary": profile.get("primary_emotions", []),
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
        return render_template("index.html", error="Please paste a dream before decoding.")

    analysis = analyze_dream(
        dream_text=dream_text,
        title=dream_title,
        felt_during=felt_during,
        felt_after=felt_after,
        life_context=life_context,
    )

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
