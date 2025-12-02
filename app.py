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


def read_logs() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        return []
    out: List[Dict[str, Any]] = []
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
# Built-in Symbol Lexicon (no external JSON)
# ----------------------------------------------------

DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
    "cat": {
        "themes": ["independence", "intuition", "boundaries"],
        "notes": "Cats can reflect emotional independence or sensitivity to personal space."
    },
    "dog": {
        "themes": ["loyalty", "companionship", "protection"],
        "notes": "Dogs often relate to trusted relationships or a desire for support."
    },
    "snake": {
        "themes": ["transformation", "danger", "hidden emotion"],
        "notes": "Snakes can represent change, fear, or something emerging from the unconscious."
    },
    "spider": {
        "themes": ["entrapment", "patience", "plans"],
        "notes": "Spiders may point to feeling caught in a situation or weaving long-term plans."
    },
    "rat": {
        "themes": ["disgust", "survival", "hidden problems"],
        "notes": "Rats can symbolize anxiety about betrayal or contamination."
    },
    "wolf": {
        "themes": ["instinct", "pack dynamics", "threat"],
        "notes": "Wolves may reflect primal instincts or fear of being targeted."
    },
    "bear": {
        "themes": ["power", "rest", "territory"],
        "notes": "Bears can represent big emotion or a need for hibernation/rest."
    },
    "lion": {
        "themes": ["courage", "authority", "pride"],
        "notes": "Lions often relate to personal power or confronting dominance."
    },
    "tiger": {
        "themes": ["anger", "wild energy", "fear"],
        "notes": "Tigers may point to intense, wary emotions."
    },
    "bird": {
        "themes": ["freedom", "perspective", "messages"],
        "notes": "Birds can symbolize the desire to rise above problems."
    },
    "fish": {
        "themes": ["unconscious", "emotion", "intuition"],
        "notes": "Fish often relate to feelings just below awareness."
    },
    "shark": {
        "themes": ["predatory threat", "fear", "competition"],
        "notes": "Sharks may represent threats or ruthless people/situations."
    },
    "horse": {
        "themes": ["vitality", "drive", "freedom"],
        "notes": "Horses reflect energy, sexual vitality, or forward motion."
    },
    "unicorn": {
        "themes": ["idealism", "fantasy", "purity"],
        "notes": "Unicorns point to magical or idealized desires."
    },
    "baby": {
        "themes": ["new beginnings", "vulnerability", "responsibility"],
        "notes": "Babies symbolize new projects or delicate parts of self."
    },
    "child": {
        "themes": ["innocence", "potential", "past self"],
        "notes": "Children often reflect younger you or something developing."
    },
    "mother": {
        "themes": ["nurture", "care", "control"],
        "notes": "Mothers can represent emotional support or smothering."
    },
    "father": {
        "themes": ["authority", "structure", "judgment"],
        "notes": "Fathers often relate to rules, expectations, or self-critique."
    },
    "stranger": {
        "themes": ["unknown self", "new situation", "uncertainty"],
        "notes": "Strangers stand in for parts of you or life you don't yet know."
    },
    "guide": {
        "themes": ["intuition", "inner wisdom", "transition"],
        "notes": "A guide figure often represents your inner compass or readiness to change."
    },
    "faceless guide": {
        "themes": ["mystery", "ambiguous help", "identity in flux"],
        "notes": "A faceless guide suggests guidance without a clear identity."
    },
    "intruder": {
        "themes": ["boundary violation", "fear", "unwanted influence"],
        "notes": "Intruders mirror anxiety about privacy or safety."
    },
    "house": {
        "themes": ["self", "psyche", "life structure"],
        "notes": "Houses represent your inner world or life situation."
    },
    "childhood home": {
        "themes": ["past influences", "family patterns", "nostalgia"],
        "notes": "Points to old emotional patterns resurfacing."
    },
    "school": {
        "themes": ["evaluation", "learning", "comparison"],
        "notes": "School dreams often involve feeling tested or judged."
    },
    "office": {
        "themes": ["work", "obligation", "performance"],
        "notes": "Offices echo concerns about productivity or status."
    },
    "hospital": {
        "themes": ["healing", "vulnerability", "crisis"],
        "notes": "Hospitals signal something needing care/repair."
    },
    "forest": {
        "themes": ["unknown", "mystery", "inner exploration"],
        "notes": "Forests represent entering the depths of your psyche."
    },
    "ocean": {
        "themes": ["deep emotion", "vastness", "overwhelm"],
        "notes": "Oceans symbolize emotional depths or overwhelm."
    },
    "river": {
        "themes": ["life flow", "transition", "direction"],
        "notes": "Rivers reflect how your life is moving."
    },
    "lake": {
        "themes": ["still emotion", "reflection", "containment"],
        "notes": "Lakes point to contained feelings or reflection."
    },
    "mountain": {
        "themes": ["challenge", "aspiration", "perspective"],
        "notes": "Mountains represent big goals or obstacles."
    },
    "car": {
        "themes": ["agency", "direction", "control"],
        "notes": "Cars reflect how in control you feel of your path."
    },
    "train": {
        "themes": ["life path", "routine", "collective journey"],
        "notes": "Trains suggest being on set tracks or shared paths."
    },
    "plane": {
        "themes": ["ambition", "transition", "risk"],
        "notes": "Planes symbolize bigger shifts or risks."
    },
    "stairs": {
        "themes": ["progress", "regression", "movement between levels"],
        "notes": "Stairs reflect moving up/down emotionally or psychologically."
    },
    "door": {
        "themes": ["opportunity", "threshold", "choice"],
        "notes": "Doors represent transitions and decisions."
    },
    "window": {
        "themes": ["perspective", "distance", "longing"],
        "notes": "Windows show how you view a situation."
    },
    "bridge": {
        "themes": ["transition", "connection", "risk"],
        "notes": "Bridges mark crossings between phases."
    },
    "storm": {
        "themes": ["emotional turmoil", "conflict", "release"],
        "notes": "Storms mirror intense emotion or conflict."
    },
    "fire": {
        "themes": ["passion", "destruction", "purification"],
        "notes": "Fire can burn away what no longer serves or express anger."
    },
    "flood": {
        "themes": ["emotional overwhelm", "loss of control", "cleansing"],
        "notes": "Floods depict feelings rising too fast to handle."
    },
    "exam": {
        "themes": ["evaluation", "self-judgment", "performance anxiety"],
        "notes": "Exam dreams are classic for fearing exposure as unprepared."
    },
    "teeth falling out": {
        "themes": ["vulnerability", "appearance", "loss of power"],
        "notes": "Teeth dreams concern aging, power, or being seen."
    },
    "flying": {
        "themes": ["freedom", "escape", "perspective"],
        "notes": "Flying can feel liberating or like rising above problems."
    },
    "falling": {
        "themes": ["loss of control", "fear of failure", "instability"],
        "notes": "Falling reflects anxiety about losing safety or status."
    },
    "naked in public": {
        "themes": ["exposure", "shame", "vulnerability"],
        "notes": "These dreams reflect fear of being too visible or judged."
    },
    "chase": {
        "themes": ["avoidance", "fear", "unresolved issues"],
        "notes": "Being chased points to something you’re avoiding."
    },
    "death": {
        "themes": ["change", "ending", "transition"],
        "notes": "Dream death rarely predicts literal death; it reflects endings."
    },
    "funeral": {
        "themes": ["closure", "mourning", "letting go"],
        "notes": "Funerals represent saying goodbye to roles or relationships."
    },
    "wedding": {
        "themes": ["union", "commitment", "integration"],
        "notes": "Weddings echo commitments or parts of life coming together."
    },
    "pregnancy": {
        "themes": ["potential", "growth", "creative process"],
        "notes": "Pregnancy often signals something new developing within you."
    },
    "blood": {
        "themes": ["life force", "injury", "sacrifice"],
        "notes": "Blood points to strong emotion or vital stakes."
    },
    "mirror": {
        "themes": ["self-image", "identity", "self-reflection"],
        "notes": "Mirrors invite you to look at how you see yourself."
    },
    "phone": {
        "themes": ["communication", "connection", "missed messages"],
        "notes": "Phones reflect how heard/connected you feel."
    },
    "lost": {
        "themes": ["confusion", "search for direction", "identity"],
        "notes": "Feeling lost mirrors uncertainty about path or role."
    },
    "trapped": {
        "themes": ["stuckness", "constraint", "pressure"],
        "notes": "Being trapped echoes feeling constrained or powerless."
    },
    "museum": {
        "themes": ["memory", "life review", "reflection"],
        "notes": "Museums are curated spaces of remembered or unresolved moments."
    },
    "letter": {
        "themes": ["communication", "unfinished business", "messages"],
        "notes": "Unopened letters symbolize unresolved communication."
    },
    "starlight": {
        "themes": ["guidance", "hope", "cosmic perspective"],
        "notes": "Starlight suggests insight emerging in the dark."
    },
    "constellation": {
        "themes": ["pattern recognition", "connection", "destiny"],
        "notes": "Constellations are about seeing patterns in scattered points."
    },
    "glass": {
        "themes": ["transparency", "fragility", "clarity"],
        "notes": "Glass reflects vulnerability or seeing clearly through something."
    },
    "cracked glass": {
        "themes": ["instability", "breakthrough", "fragility"],
        "notes": "Cracks suggest something under strain or about to change."
    },
    "alarm": {
        "themes": ["urgency", "warning", "awakening"],
        "notes": "Alarms point to rising internal pressure or awareness."
    },
    "gravity shift": {
        "themes": ["imbalance", "change", "perspective shift"],
        "notes": "Shifting gravity mirrors feeling off-balance or destabilized."
    },
    "galaxy face": {
        "themes": ["cosmic identity", "mystery", "inner truth"],
        "notes": "A face made of stars suggests identity in flux and depth."
    }
}

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
    "glass": "glass",
    "shattered": "cracked glass",
    "shatter": "cracked glass",
    "burn": "fire",
    "burned": "fire",
    "burning": "fire",
    "water": "ocean",
    "flood": "ocean",
    "fall": "falling",
    "falling": "falling",
    "tilt": "gravity shift",
    "tilting": "gravity shift",
    "gravity": "gravity shift"
}


def normalize_word(word: str) -> str:
    w = word.lower().strip()
    if w in SYNONYMS:
        return SYNONYMS[w]
    if w.endswith("s") and w[:-1] in DREAM_KEYWORDS:
        return w[:-1]
    return w


# ----------------------------------------------------
# Conceptual Motif Detector (high sensitivity)
# ----------------------------------------------------

def detect_keywords(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    seen: set[str] = set()

    tokens = re.findall(r"[a-zA-Z']+", lowered)
    normalized_tokens = [normalize_word(tok) for tok in tokens]

    # 1) Direct matches
    for nt in normalized_tokens:
        if nt in DREAM_KEYWORDS and nt not in seen:
            found.append(nt)
            seen.add(nt)

    # 2) Multi-word motifs by literal phrase (e.g., "cracked glass", "gravity shift")
    for kw in DREAM_KEYWORDS:
        if " " in kw and kw in lowered and kw not in seen:
            found.append(kw)
            seen.add(kw)

    # 3) Conceptual cues: weaker hints that still map to motifs
    cues = {
        "mirror": "mirror",
        "child": "child",
        "door": "door",
        "guide": "guide",
        "museum": "museum",
        "letter": "letter",
        "ocean": "ocean",
        "water": "ocean",
        "bridge": "bridge",
        "star": "starlight",
        "glow": "starlight",
        "collapse": "fire",
        "alarm": "alarm",
        "glass": "glass"
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
        "just", "like", "your", "with", "have", "then", "were"
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
# LLM Calls – 4o + fallback to 4o-mini
# ----------------------------------------------------

def call_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    # Primary: gpt-4o
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.6,
            timeout=8,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception:
        # Silent fallback: gpt-4o-mini
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.6,
            timeout=8,
        )
        return json.loads(completion.choices[0].message.content)


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
