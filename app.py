import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # Uses OPENAI_API_KEY


# ---------------------------
# Paths & logging
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
# Symbol lexicon + keywords
# ---------------------------

# Fallback lexicon (used only if JSON file is missing or invalid)
DEFAULT_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
    "cat": {
        "themes": ["independence", "intuition", "feminine energy"],
        "notes": "Cats can reflect autonomy, emotional distance, or a need for comfort on your own terms.",
    },
    "snake": {
        "themes": ["threat", "transformation", "hidden fear"],
        "notes": "Snakes often represent sources of anxiety, betrayal, or powerful but unsettling change.",
    },
    "water": {
        "themes": ["emotion", "unconscious", "change"],
        "notes": "Water usually points to emotional states, shifts, or things 'beneath the surface.'",
    },
    "river": {
        "themes": ["life path", "flow", "transition"],
        "notes": "Rivers often symbolize the direction and momentum of one’s life or a specific transition.",
    },
    "house": {
        "themes": ["self", "identity", "boundaries"],
        "notes": "Houses can mirror parts of the self, personal history, or different 'rooms' of your life.",
    },
    "door": {
        "themes": ["opportunity", "barrier", "choice"],
        "notes": "Doors often represent thresholds, decisions, or access to new phases of life.",
    },
    "car": {
        "themes": ["agency", "direction", "control"],
        "notes": "Cars can symbolize how in control you feel about where your life is heading.",
    },
    "falling": {
        "themes": ["loss of control", "vulnerability"],
        "notes": "Falling often appears when people feel overwhelmed or afraid of failure.",
    },
    "teeth": {
        "themes": ["appearance", "communication", "power"],
        "notes": "Teeth dreams can relate to self-image, aging, or fear about how you come across.",
    },
    "flight": {
        "themes": ["freedom", "escape", "perspective"],
        "notes": "Flying can express relief, new perspective, or desire to escape constraints.",
    },
    "school": {
        "themes": ["evaluation", "learning", "performance anxiety"],
        "notes": "School dreams often arise around feeling tested, judged, or unprepared.",
    },
    "baby": {
        "themes": ["new beginnings", "vulnerability", "responsibility"],
        "notes": "Babies can symbolize fragile new projects, relationships, or parts of yourself.",
    },
    "forest": {
        "themes": ["unknown", "exploration", "overwhelm"],
        "notes": "Forests may represent confusion or the need to explore something unclear in your life.",
    },
    "road": {
        "themes": ["direction", "progress", "choices"],
        "notes": "Roads often mirror the path you feel you’re on and how straightforward or confusing it seems.",
    },
    "bridge": {
        "themes": ["transition", "connection", "risk"],
        "notes": "Bridges show up when people are in-between stages or trying to connect parts of life.",
    },
    "storm": {
        "themes": ["intense emotion", "conflict", "release"],
        "notes": "Storms often track with emotional upheaval or brewing conflict.",
    },
    "animal": {
        "themes": ["instinct", "drives", "natural self"],
        "notes": "Animals can reflect instinctive reactions or unpolished feelings.",
    },
    "shadow": {
        "themes": ["hidden self", "denied feelings"],
        "notes": "Shadows point toward parts of yourself you’re not fully looking at yet.",
    },
    "mirror": {
        "themes": ["self-reflection", "identity"],
        "notes": "Mirrors tend to show up when you’re asking who you are or how you’re changing.",
    },
    "boat": {
        "themes": ["navigation", "emotional journey"],
        "notes": "Boats can mirror how you’re moving across emotional terrain or life changes.",
    },
    "milk": {
        "themes": ["nurturing", "comfort", "need for care"],
        "notes": "Milk often touches themes of nourishment, dependency, or needing support.",
    },
    "knight": {
        "themes": ["authority", "judgment", "duty"],
        "notes": "Knights can represent external or internal critics, rules, or expectations.",
    },
    "armor": {
        "themes": ["defense", "protection", "emotional guard"],
        "notes": "Armor often mirrors how defended or guarded you feel in waking life.",
    },
}

# Try to load the rich JSON lexicon; fall back to the in-code one if needed
try:
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        loaded_lex = json.load(f)
    if isinstance(loaded_lex, dict) and loaded_lex:
        DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = loaded_lex
    else:
        DREAM_SYMBOL_LEXICON = DEFAULT_SYMBOL_LEXICON
except Exception:
    DREAM_SYMBOL_LEXICON = DEFAULT_SYMBOL_LEXICON

# Keywords are now derived directly from the active lexicon
DREAM_KEYWORDS: List[str] = sorted(DREAM_SYMBOL_LEXICON.keys())


def detect_keywords(text: str) -> List[str]:
    """
    Detect motifs using the active DREAM_KEYWORDS list.

    - Handles single words and multi-word phrases.
    - Case-insensitive.
    - Allows simple plural 's' on single-word entries.
    """
    lowered = text.lower()
    found: List[str] = []

    for kw in DREAM_KEYWORDS:
        kw_lower = kw.lower()
        if " " in kw_lower:
            # Multi-word phrase, match as-is with word boundaries on both sides
            pattern = r"(?<!\w)" + re.escape(kw_lower) + r"(?!\w)"
        else:
            # Single word: allow optional trailing 's'
            pattern = r"\b" + re.escape(kw_lower) + r"s?\b"

        if re.search(pattern, lowered):
            found.append(kw)

    # de-duplicate, keep order
    return list(dict.fromkeys(found))


def simple_candidate_symbols(text: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """Very simple candidate symbol extractor based on word frequency."""
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    stop = {
        "the",
        "and",
        "with",
        "this",
        "that",
        "from",
        "into",
        "onto",
        "then",
        "were",
        "have",
        "about",
        "while",
        "when",
        "just",
        "like",
        "over",
        "under",
        "through",
    }

    counts: Dict[str, int] = {}
    for tok in tokens:
        if len(tok) < 4 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    items = [{"phrase": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: (-x["count"], -len(x["phrase"])))
    return items[:max_items]


def build_priority_symbols(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score symbols based on local frequency + presence in the lexicon.
    Larger lexicon → richer priority mapping.
    """
    scored = []
    for c in candidates:
        phrase = c.get("phrase", "")
        local_count = c.get("count", 0)
        lex = DREAM_SYMBOL_LEXICON.get(phrase.lower())
        in_lex = 1 if lex else 0
        score = local_count * 2 + in_lex * 3
        scored.append(
            {
                "symbol": phrase,
                "local_count": local_count,
                "in_lexicon": bool(in_lex),
                "score": score,
                "lexicon_themes": (lex or {}).get("themes", []),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:8]


# ---------------------------
# Model prompt
# ---------------------------

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

    completion = client.chat.completions.create(
        model="gpt-41",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )

    raw = completion.choices[0].message.content

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "micronarrative": "",
            "summary": "There was an error parsing the model output.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {
                "primary_emotions": [],
                "overall_tone": "unknown",
            },
            "emotional_arc": [],
            "narrative_pattern": {
                "pattern_name": "Unknown",
                "description": "",
                "related_themes": [],
            },
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": ["Model output could not be parsed."],
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
        "narrative_pattern": data.get("narrative_pattern", {})
        or {
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


# ---------------------------
# Routes – robust against 405
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle decode directly from /
        return handle_decode()
    return render_template("index.html")


@app.route("/decode", methods=["GET", "POST"])
def decode():
    if request.method == "POST":
        # Handle decode from /decode
        return handle_decode()
    # GET /decode → send them to the main form
    return redirect(url_for("index"))


def handle_decode():
    dream_text = request.form.get("dream_text", "").strip()
    dream_title = request.form.get("dream_title", "").strip()
    felt_during = request.form.get("felt_during", "").strip()
    felt_after = request.form.get("felt_after", "").strip()
    life_context = request.form.get("life_context", "").strip()

    if not dream_text:
        return render_template(
            "index.html",
            error="Please paste a dream before decoding.",
        )

    try:
        analysis = analyze_dream(
            dream_text=dream_text,
            title=dream_title,
            felt_during=felt_during,
            felt_after=felt_after,
            life_context=life_context,
        )
    except Exception as e:
        return (
            f"<h1>Decode error</h1><p><strong>{type(e).__name__}</strong>: {e}</p>",
            500,
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
