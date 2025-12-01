import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, abort
from openai import OpenAI

# ---------------------------
# App + OpenAI client
# ---------------------------

app = Flask(__name__)
client = OpenAI()  # Uses OPENAI_API_KEY from environment

# ---------------------------
# Logging setup
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "dream_logs.jsonl")


def append_log(record: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_logs() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# ---------------------------
# Symbol lexicon + keywords
# ---------------------------

DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
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

DREAM_KEYWORDS: List[str] = sorted(
    {
        "cat",
        "snake",
        "water",
        "river",
        "ocean",
        "lake",
        "house",
        "door",
        "window",
        "car",
        "train",
        "bus",
        "falling",
        "teeth",
        "flight",
        "flying",
        "school",
        "exam",
        "baby",
        "forest",
        "road",
        "bridge",
        "storm",
        "animal",
        "shadow",
        "mirror",
        "boat",
        "milk",
        "knight",
        "armor",
    }
)

# ---------------------------
# Helpers
# ---------------------------


def detect_keywords(text: str) -> List[str]:
    """
    Detect simple dream motifs using word boundaries, allowing plural for single words.
    """
    lowered = text.lower()
    found: List[str] = []
    for kw in DREAM_KEYWORDS:
        if " " in kw:
            pattern = r"(?<!\\w)" + re.escape(kw) + r"(?!\\w)"
        else:
            pattern = r"\\b" + re.escape(kw) + r"s?\\b"
        if re.search(pattern, lowered):
            found.append(kw)
    # preserve order of first appearance
    return list(dict.fromkeys(found))


def simple_candidate_symbols(text: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Very simple candidate symbol extractor:
    - 1-word tokens longer than 3 chars, not obvious stopwords
    - counted by frequency
    """
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
    Lightweight priority scoring: local frequency + lexicon presence.
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
# System prompt for the model
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
- "detected_keywords": motifs auto-detected in the text
- "candidate_symbols": phrases with counts that look symbolically important
- "priority_symbols": symbols scored higher based on usage + lexicon

Principles:
- Stay grounded in the dream's actual content and the life_context.
- Make interpretations tentative: use language like "may suggest" or "could reflect".
- Avoid superstition, fortune-telling, or claims about the future.
- Tie symbols to emotional and situational themes, not fixed meanings.
- Help the dreamer generate insight, not fear.

OUTPUT must be VALID JSON with these top-level keys:

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

All float values must be between 0 and 1. Keep lists reasonably short (3–7 items).
"""


# ---------------------------
# Core analysis function
# ---------------------------

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
        model="gpt-4o-mini",
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

    # Normalize and enrich a bit for the template
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


# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/decode", methods=["POST"])
def decode():
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


# ---------------------------
# Run (local dev)
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)
