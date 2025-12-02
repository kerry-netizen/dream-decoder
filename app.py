import json
import os
import re
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


def append_log(record: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------
# Built-in Symbol Lexicon (self-contained)
# ----------------------------------------------------

DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
    "cat": {"themes": ["independence", "intuition"], "notes": "Emotional independence, sensitivity to boundaries."},
    "dog": {"themes": ["loyalty", "support"], "notes": "Trust, companionship, desire for support."},
    "snake": {"themes": ["transformation", "fear"], "notes": "Unsettling change, hidden emotion."},
    "spider": {"themes": ["entrapment"], "notes": "Feeling caught in a situation or web of obligations."},
    "rat": {"themes": ["disgust", "hidden problems"], "notes": "Something unwanted creeping into awareness."},
    "wolf": {"themes": ["instinct", "threat"], "notes": "Primal drive or sense of being hunted/targeted."},
    "bear": {"themes": ["power", "rest"], "notes": "Big emotion, territorial defense, need to hibernate."},
    "lion": {"themes": ["pride", "authority"], "notes": "Leadership, dominance, or pride issues."},
    "tiger": {"themes": ["anger"], "notes": "Intense emotion you’re wary of (especially anger)."},
    "bird": {"themes": ["freedom", "perspective"], "notes": "Desire to rise above problems or see broadly."},
    "fish": {"themes": ["emotion"], "notes": "Submerged feelings, intuitions just below awareness."},
    "shark": {"themes": ["threat"], "notes": "Perceived predators, ruthless competition."},
    "horse": {"themes": ["drive"], "notes": "Vitality, sexual energy, forward momentum."},
    "unicorn": {"themes": ["idealism"], "notes": "Fantasy, impossible or idealized desires."},
    "baby": {"themes": ["new beginnings"], "notes": "Something new, vulnerable, needing care."},
    "child": {"themes": ["innocence", "past self"], "notes": "Younger aspects of you, potential and vulnerability."},
    "mother": {"themes": ["care"], "notes": "Nurture, emotional holding, or smothering."},
    "father": {"themes": ["authority"], "notes": "Rules, expectations, internal critic."},
    "stranger": {"themes": ["unknown"], "notes": "Unknown parts of self, new situations."},
    "guide": {"themes": ["intuition"], "notes": "Inner compass, subtle guidance."},
    "faceless guide": {"themes": ["mystery"], "notes": "Guidance without clear identity; you feel led but not sure by whom."},
    "house": {"themes": ["self"], "notes": "Your inner world or life structure."},
    "forest": {"themes": ["mystery"], "notes": "Exploration of the unconscious."},
    "ocean": {"themes": ["depth"], "notes": "Vast, overwhelming emotion or unconscious material."},
    "river": {"themes": ["flow"], "notes": "Direction and momentum of your life."},
    "lake": {"themes": ["containment"], "notes": "Still emotion, reflection, containment."},
    "mountain": {"themes": ["challenge"], "notes": "Obstacles, lofty goals, perspective."},
    "car": {"themes": ["control"], "notes": "Sense of control over your path."},
    "train": {"themes": ["path"], "notes": "Life track, momentum, routines."},
    "plane": {"themes": ["transition"], "notes": "Major changes or ambitions."},
    "stairs": {"themes": ["progress"], "notes": "Moving between levels of insight or emotion."},
    "door": {"themes": ["opportunity"], "notes": "Thresholds, decisions, new phases."},
    "window": {"themes": ["perspective"], "notes": "How you see a situation, or longing."},
    "bridge": {"themes": ["transition"], "notes": "Crossing between states or roles."},
    "storm": {"themes": ["conflict"], "notes": "Emotional turmoil, brewing conflict."},
    "fire": {"themes": ["destruction", "purification"], "notes": "Burning away the old, passion or anger."},
    "flood": {"themes": ["overwhelm"], "notes": "Emotions rising too fast to manage."},
    "mirror": {"themes": ["identity"], "notes": "Self-image, reflection, seeing yourself clearly or distorted."},
    "lost": {"themes": ["confusion"], "notes": "Searching for direction or role."},
    "trapped": {"themes": ["pressure"], "notes": "Feeling stuck or constrained."},

    # Dream-Decoder-specific symbols
    "museum": {"themes": ["memory"], "notes": "Life review, curated memories, unresolved moments."},
    "letter": {"themes": ["unfinished business"], "notes": "Unspoken or unresolved communication."},
    "starlight": {"themes": ["guidance"], "notes": "Subtle insight or hope."},
    "constellation": {"themes": ["connection"], "notes": "Seeing patterns in events or memories."},
    "glass": {"themes": ["clarity"], "notes": "Transparency, vulnerability, seeing through a barrier."},
    "cracked glass": {"themes": ["instability"], "notes": "Something under strain or about to change."},
    "alarm": {"themes": ["urgency"], "notes": "Inner warning, rising awareness."},
    "gravity shift": {"themes": ["imbalance"], "notes": "Perspective or stability changing."},
    "galaxy face": {"themes": ["mystery"], "notes": "Identity in flux, feeling part of something larger."},
}

DREAM_KEYWORDS: List[str] = sorted(DREAM_SYMBOL_LEXICON.keys())


# ----------------------------------------------------
# Synonym Normalization
# ----------------------------------------------------

SYNONYMS: Dict[str, str] = {
    "stars": "starlight",
    "star": "starlight",
    "letters": "letter",
    "messages": "letter",
    "message": "letter",
    "burning": "fire",
    "burned": "fire",
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
# High-Sensitivity Motif Detector
# ----------------------------------------------------

def detect_keywords(text: str) -> List[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    found: List[str] = []
    seen = set()

    # token-level matches
    for tok in tokens:
        nt = normalize(tok)
        if nt in DREAM_KEYWORDS and nt not in seen:
            found.append(nt)
            seen.add(nt)

    # multiword motifs
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered and kw not in seen:
            found.append(kw)
            seen.add(kw)

    # contextual cues
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
        "glass": "glass",
        "collapse": "fire",
    }
    for cue, motif in cues.items():
        if cue in lowered and motif in DREAM_KEYWORDS and motif not in seen:
            found.append(motif)
            seen.add(motif)

    return found


# ----------------------------------------------------
# Candidate Symbols & Priority
# ----------------------------------------------------

def simple_candidate_symbols(text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    stop = {"the", "and", "this", "that", "from", "just", "like", "then", "with", "your"}

    counts: Dict[str, int] = {}
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

You will receive:
- dream_text
- felt_during, felt_after
- life_context
- detected_keywords
- candidate_symbols
- priority_symbols

Principles:
- Stay grounded in the dream's actual content and the life_context.
- Make interpretations tentative: use language like "may suggest" or "could reflect".
- Avoid superstition, fortune-telling, or claims about the future.
- Focus on psychological meaning, emotional patterns, and inner conflicts.
- Help the dreamer generate insight, not fear.
- It is okay to be rich and detailed in your explanations.

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

Formatting / richness:
- "micronarrative": 2–4 sentences retelling the dream vividly but concisely.
- "summary": 3–5 sentences explaining the main psychological themes.
- "interpretive_narrative": your main canvas. Write 3–7 short paragraphs
  (around 400–700 words) weaving symbols, emotions, and life_context into a
  cohesive psychological story.
- "key_symbols": 3–7 key symbols with 2–3 possible meanings and a confidence
  value between 0 and 1.
- "emotional_profile": 3–5 primary emotions with intensities 0–1 and an overall tone.
- "reflection_prompts": 3–6 thoughtful, gentle questions inviting self-reflection.
Keep JSON structure strict, but allow prose fields to be rich and textured.
"""


# ----------------------------------------------------
# LLM Call (mini-only, 30-second timeout)
# ----------------------------------------------------

def call_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
        timeout=30,
    )
    return json.loads(response.choices[0].message.content)


# ----------------------------------------------------
# Helper: Normalize emotional structures for templates
# ----------------------------------------------------

def normalize_emotional_profile(raw_profile: Any) -> (List[Dict[str, Any]], str):
    profile = raw_profile or {}
    if not isinstance(profile, dict):
        profile = {}
    primary_raw = profile.get("primary_emotions", []) or []
    overall_tone = profile.get("overall_tone", "unknown") or "unknown"

    normalized_primary: List[Dict[str, Any]] = []
    for item in primary_raw:
        if isinstance(item, dict):
            name = item.get("name") or item.get("emotion") or str(item)
            intensity = item.get("intensity", 0.0)
        else:
            name = str(item)
            intensity = 0.0
        if not isinstance(intensity, (int, float)):
            intensity = 0.0
        normalized_primary.append({"name": name, "intensity": float(intensity)})

    return normalized_primary, overall_tone


def normalize_emotional_arc(raw_arc: Any) -> List[Dict[str, Any]]:
    arc = raw_arc or []
    normalized_arc: List[Dict[str, Any]] = []

    if not isinstance(arc, list):
        return normalized_arc

    for st in arc:
        if isinstance(st, dict):
            stage = st.get("stage") or ""
            emotion = st.get("emotion") or st.get("name") or ""
            intensity = st.get("intensity", 0.0)
        else:
            stage = ""
            emotion = str(st)
            intensity = 0.0
        if not isinstance(intensity, (int, float)):
            intensity = 0.0
        normalized_arc.append(
            {"stage": stage, "emotion": emotion, "intensity": float(intensity)}
        )

    return normalized_arc


# ----------------------------------------------------
# Dream Analysis
# ----------------------------------------------------

def analyze_dream(
    dream_text: str,
    title: str = "",
    felt_during: str = "",
    felt_after: str = "",
    life_context: str = "",
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
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("MODEL ERROR:", error_msg, flush=True)
        data = {
            "micronarrative": "",
            "summary": f"There was an error contacting the model: {error_msg}",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "emotional_arc": [],
            "narrative_pattern": {},
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": ["Model call failed.", error_msg],
        }

    raw_profile = data.get("emotional_profile", {}) or {}
    normalized_primary, tone = normalize_emotional_profile(raw_profile)
    normalized_arc = normalize_emotional_arc(data.get("emotional_arc", []))

    return {
        "micronarrative": data.get("micronarrative", "") or "",
        "summary": data.get("summary", "") or "",
        "interpretive_narrative": data.get("interpretive_narrative", "") or "",
        "key_symbols": data.get("key_symbols", []) or [],
        "emotional_profile_primary": normalized_primary,
        "emotional_profile_tone": tone,
        "emotional_arc": normalized_arc,
        "narrative_pattern": data.get("narrative_pattern", {}) or {},
        "symbol_relations": data.get("symbol_relations", []) or [],
        "reflection_prompts": data.get("reflection_prompts", []) or [],
        "cautions": data.get("cautions", []) or [],
        "detected_keywords": detected,
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
