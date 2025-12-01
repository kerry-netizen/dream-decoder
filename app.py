import json
import os
import re
from dataclasses import dataclass
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
# Dream keyword lexicon
# ---------------------------

# Simple symbol lexicon for now. In the future, this can be expanded or moved to JSON.
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

# A simple list of keywords we might want to detect in dream text.
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
# Utility + data classes
# ---------------------------


def load_logs_path() -> str:
    """
    Path to the local JSONL log file.
    """
    return os.environ.get("DREAM_LOG_PATH", "dream_logs.jsonl")


def read_logs() -> List[Dict[str, Any]]:
    path = load_logs_path()
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records


def append_log(record: Dict[str, Any]) -> None:
    path = load_logs_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_occurrences(text: str, phrase: str) -> int:
    """
    Count non-overlapping case-insensitive occurrences of phrase in text.
    """
    if not phrase:
        return 0
    pattern = re.escape(phrase)
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches)


@dataclass
class SymbolMeaning:
    symbol: str
    description: str
    possible_meanings: List[str]
    confidence: float
    local_count: int
    global_count: int


@dataclass
class EmotionalProfile:
    name: str
    intensity: float


@dataclass
class EmotionalStage:
    stage: str
    emotion: str
    intensity: float


@dataclass
class SymbolRelation:
    source: str
    target: str
    relation: str


@dataclass
class NarrativePattern:
    pattern_name: str
    description: str
    related_themes: List[str]


@dataclass
class DreamAnalysis:
    micronarrative: str
    summary: str
    interpretive_narrative: str
    key_symbols: List[SymbolMeaning]
    emotional_profile_primary: List[EmotionalProfile]
    emotional_profile_tone: str
    emotional_arc: List[EmotionalStage]
    narrative_pattern: NarrativePattern
    symbol_relations: List[SymbolRelation]
    reflection_prompts: List[str]
    cautions: List[str]
    detected_keywords: List[str]


# ---------------------------
# SYSTEM PROMPT
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
- "lexicon_entries": themes and notes for some symbols from a dream symbol lexicon
- "priority_symbols": a short list of symbols with importance scores, based on:
    * frequency within this dream,
    * known symbolic weight (from lexicon),
    * and whether they overlap with detected_keywords

Your task is to build a structured, psychologically grounded interpretation.

Principles:
- Stay grounded in the dream's actual content and the life_context.
- Make interpretations tentative: use language like "may suggest" or "could reflect".
- Avoid superstition, fortune-telling, or claims about the future.
- Tie symbols to emotional and situational themes, not fixed meanings.
- Help the dreamer generate insight, not fear.

OUTPUT STRUCTURE (high-level):

1) A BRIEF MICRONARRATIVE
   - 2–4 sentences retelling the dream as a cohesive story, in the present tense.
   - No interpretation yet, just a vivid, concise narrative.

2) A SUMMARY
   - 2–4 sentences summarizing the dream and its possible psychological meaning.
   - This should be interpretive but gentle and non-dogmatic.

3) An EMOTIONAL MODEL
   - "emotional_profile": primary emotions + overall tone.
   - "emotional_arc": how emotion shifts across the dream, as a list of stages.
     Example:
     "emotional_arc": [
       {"stage": "beginning", "emotion": "curiosity", "intensity": 0.5},
       {"stage": "middle", "emotion": "fear", "intensity": 0.8},
       {"stage": "end", "emotion": "relief", "intensity": 0.6}
     ]

4) SYMBOL MODEL
   - "key_symbols": 3–7 symbols, chosen with strong reference to priority_symbols,
     candidate_symbols, lexicon_entries, and detected_keywords.
   - For each symbol, include description, possible meanings, and a confidence score.

5) SYMBOL RELATIONSHIPS
   - "symbol_relations": how key symbols relate to each other inside the dream narrative
     (e.g. "cat" protects dreamer from "snake"; "storm" interrupts "wedding").
   - This is where you connect symbols into mini-stories.

6) NARRATIVE PATTERN
   - Pick a simple pattern label such as "confrontation", "escape", "search/quest",
     "test/evaluation", "reunion", "loss/grief", or "transformation".
   - Briefly explain why this pattern fits and what themes it often carries.

7) INTERPRETIVE NARRATIVE
   - 1–3 short paragraphs integrating symbols, emotions, and life_context into a
     cohesive psychological interpretation.
   - This is where you gently explore:
       * what tensions the dreamer may be wrestling with,
       * what conflicting desires or fears show up,
       * and what this dream might be asking them to reflect on.
   - Always present this as possibilities, not as fact.

8) REFLECTION PROMPTS & CAUTIONS
   - "reflection_prompts": 2–5 questions that the dreamer could journal about.
   - "cautions": any notes about not over-pathologizing,
     or about seeking professional help if the dream ties into severe distress.

Final output must be VALID JSON with the following top-level keys:

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

Make sure all floats are between 0 and 1, and keep lists reasonably short (3–7 items).
"""


# ---------------------------
# Keyword detection + symbol helpers
# ---------------------------


def detect_keywords(text: str) -> List[str]:
    """
    Detect keywords with proper word boundaries:
    - Multi-word phrases: strict phrase boundaries
    - Single words: strict word boundaries, with optional 's' plural
    """
    lowered = text.lower()
    found: List[str] = []

    for kw in DREAM_KEYWORDS:
        if " " in kw:
            pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
        else:
            pattern = r'\b' + re.escape(kw) + r's?\b'

        if re.search(pattern, lowered):
            found.append(kw)

    return list(dict.fromkeys(found))


def extract_candidate_symbols(text: str, max_phrases: int = 12) -> List[Dict[str, Any]]:
    """
    Very simple candidate symbol extractor:
    - Look for 1–3 word noun-like phrases:
      sequences with words and optional adjectives.
    - Count frequency case-insensitively.
    - Return top N by frequency, then length.
    """
    lowered = text.lower()

    # Tokenize to words only
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    phrases: Dict[str, int] = {}

    # Build 1–3 word sequences
    for i in range(len(tokens)):
        for length in (1, 2, 3):
            if i + length > len(tokens):
                continue
            phrase = " ".join(tokens[i : i + length])
            if len(phrase) < 3:
                continue
            phrases[phrase] = phrases.get(phrase, 0) + 1

    items = [{"phrase": p, "count": c} for p, c in phrases.items()]
    items.sort(key=lambda x: (-x["count"], -len(x["phrase"])))
    return items[:max_phrases]


def lookup_symbol_in_lexicon(symbol: str) -> Dict[str, Any]:
    """
    Look up a symbol (case-insensitive) in the lexicon.
    If not found, return {}.
    """
    key = symbol.lower().strip()
    return DREAM_SYMBOL_LEXICON.get(key, {})


def compute_symbol_stats_from_logs() -> Dict[str, int]:
    """
    Build a global frequency table of symbols from past dream logs.
    """
    records = read_logs()
    counts: Dict[str, int] = {}

    for rec in records:
        analysis = rec.get("analysis", {})
        key_syms = analysis.get("key_symbols", [])
        for sym in key_syms:
            label = sym.get("symbol", "").lower().strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1

    return counts


def compute_priority_symbols(
    candidate_symbols: List[Dict[str, Any]],
    global_symbol_stats: Dict[str, int],
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    """
    Decide which symbols to treat as highest priority for interpretation.
    Score based on:
    - local frequency
    - presence in lexicon
    - global dream frequency
    """
    scored = []
    for cs in candidate_symbols:
        phrase = cs.get("phrase", "")
        local_count = cs.get("count", 0)
        info = lookup_symbol_in_lexicon(phrase)
        in_lexicon = 1 if info else 0
        global_count = global_symbol_stats.get(phrase.lower(), 0)

        score = local_count * 2 + in_lexicon * 3 + min(global_count, 5)
        scored.append(
            {
                "symbol": phrase,
                "local_count": local_count,
                "global_count": global_count,
                "in_lexicon": bool(in_lexicon),
                "score": score,
                "lexicon_themes": info.get("themes", []),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_items]


# ---------------------------
# Core analysis function
# ---------------------------


def analyze_dream(
    dream_text: str,
    title: str = "",
    felt_during: str = "",
    felt_after: str = "",
    life_context: str = "",
) -> DreamAnalysis:
    detected = detect_keywords(dream_text)
    candidate_symbols = extract_candidate_symbols(dream_text)
    global_symbol_stats = compute_symbol_stats_from_logs()
    priority_symbols = compute_priority_symbols(candidate_symbols, global_symbol_stats)

    lex_entries = []
    for cs in candidate_symbols:
        phrase = cs.get("phrase", "")
        info = lookup_symbol_in_lexicon(phrase)
        if info:
            lex_entries.append(
                {
                    "symbol": phrase,
                    "themes": info.get("themes", []),
                    "notes": info.get("notes", "")
                }
            )
    lex_entries = lex_entries[:10]

    payload = {
        "dream_title": title,
        "dream_text": dream_text,
        "felt_during": felt_during,
        "felt_after": felt_after,
        "life_context": life_context,
        "detected_keywords": detected,
        "candidate_symbols": candidate_symbols,
        "lexicon_entries": lex_entries,
        "priority_symbols": priority_symbols,
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
            "summary": "Parsing error.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
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

    key_symbols: List[SymbolMeaning] = []
    global_symbol_stats = compute_symbol_stats_from_logs()
    for s in data.get("key_symbols", []):
        sym_text = s.get("symbol", "") or ""
        local_count = count_occurrences(dream_text, sym_text)
        global_count = global_symbol_stats.get(sym_text.lower(), 0)
        key_symbols.append(
            SymbolMeaning(
                symbol=sym_text,
                description=s.get("description", "") or "",
                possible_meanings=s.get("possible_meanings", []) or [],
                confidence=float(s.get("confidence", 0.0) or 0.0),
                local_count=local_count,
                global_count=global_count,
            )
        )

    emotional_profile_data = data.get("emotional_profile", {}) or {}
    primary_emotions_list = emotional_profile_data.get("primary_emotions", []) or []
    emotional_profile_primary = []
    for e in primary_emotions_list:
        name = e.get("name", "") or ""
        intensity = float(e.get("intensity", 0.0) or 0.0)
        emotional_profile_primary.append(EmotionalProfile(name=name, intensity=intensity))

    emotional_arc_data = []
    for st in data.get("emotional_arc", []):
        stage = st.get("stage", "") or ""
        emotion = st.get("emotion", "") or ""
        intensity = float(st.get("intensity", 0.0) or 0.0)
        emotional_arc_data.append(
            EmotionalStage(stage=stage, emotion=emotion, intensity=intensity)
        )

    narrative_pattern_data = data.get("narrative_pattern", {}) or {}
    narrative_pattern = NarrativePattern(
        pattern_name=narrative_pattern_data.get("pattern_name", "") or "",
        description=narrative_pattern_data.get("description", "") or "",
        related_themes=narrative_pattern_data.get("related_themes", []) or [],
    )

    symbol_relations_data = []
    for rel in data.get("symbol_relations", []):
        symbol_relations_data.append(
            SymbolRelation(
                source=rel.get("source", "") or "",
                target=rel.get("target", "") or "",
                relation=rel.get("relation", "") or "",
            )
        )

    return DreamAnalysis(
        micronarrative=data.get("micronarrative", "") or "",
        summary=data.get("summary", "") or "",
        interpretive_narrative=data.get("interpretive_narrative", "") or "",
        key_symbols=key_symbols,
        emotional_profile_primary=emotional_profile_primary,
        emotional_profile_tone=emotional_profile_data.get("overall_tone", "unknown")
        or "unknown",
        emotional_arc=emotional_arc_data,
        narrative_pattern=narrative_pattern,
        symbol_relations=symbol_relations_data,
        reflection_prompts=data.get("reflection_prompts", []) or [],
        cautions=data.get("cautions", []) or [],
        detected_keywords=detected,
    )


# ---------------------------
# Visual helpers for the UI
# ---------------------------


def build_emotion_bars(primary_emotions: List[EmotionalProfile]) -> List[Dict[str, Any]]:
    """
    Prepare data for bar visualization of primary emotions.
    """
    return [
        {"name": e.name, "intensity": e.intensity}
        for e in primary_emotions
        if e.name
    ]


def build_emotional_arc_timeline(
    emotional_arc: List[EmotionalStage],
) -> List[Dict[str, Any]]:
    """
    Convert emotional arc into a simple timeline structure.
    """
    timeline = []
    for idx, stage in enumerate(emotional_arc):
        timeline.append(
            {
                "index": idx,
                "stage": stage.stage,
                "emotion": stage.emotion,
                "intensity": stage.intensity,
            }
        )
    return timeline


def build_symbol_graph(
    key_symbols: List[SymbolMeaning],
    relations: List[SymbolRelation],
) -> Dict[str, Any]:
    """
    Build a simple graph structure of symbols and their relations.
    """
    nodes = []
    for s in key_symbols:
        nodes.append(
            {
                "id": s.symbol,
                "label": s.symbol,
                "weight": s.local_count + s.global_count,
            }
        )

    edges = []
    for r in relations:
        if not r.source or not r.target:
            continue
        edges.append(
            {
                "source": r.source,
                "target": r.target,
                "relation": r.relation,
            }
        )

    return {"nodes": nodes, "edges": edges}


# ---------------------------
# Search / history helpers
# ---------------------------


def search_logs(query: str) -> List[Dict[str, Any]]:
    """
    Very simple text search across logged dreams.
    Searches title, dream_text, life_context, detected_keywords,
    key_symbols, summary, interpretive_narrative, narrative_pattern.
    Returns newest-first matches with the same idx mapping as /history.
    """
    q = (query or "").strip()
    if not q:
        return []

    q_low = q.lower()
    records = read_logs()
    records = list(reversed(records))  # newest first

    results = []
    for idx, rec in enumerate(records):
        inp = rec.get("input", {})
        analysis = rec.get("analysis", {})

        haystack_parts = [
            inp.get("title", ""),
            inp.get("dream_text", ""),
            inp.get("life_context", ""),
            " ".join(inp.get("detected_keywords", [])),
            " ".join([s.get("symbol", "") for s in analysis.get("key_symbols", [])]),
            analysis.get("summary", ""),
            analysis.get("interpretive_narrative", ""),
            (analysis.get("narrative_pattern", {}) or {}).get("pattern_name", ""),
        ]

        haystack = " ".join(haystack_parts).lower()
        if q_low in haystack:
            rec_copy = dict(rec)
            rec_copy["_history_idx"] = idx
            results.append(rec_copy)

    return results


def build_symbol_frequency_from_logs(max_items: int = 20) -> List[Dict[str, Any]]:
    """
    Build a simple frequency list of symbols across all logged dreams.
    """
    records = read_logs()
    counts: Dict[str, int] = {}

    for rec in records:
        analysis = rec.get("analysis", {})
        key_syms = analysis.get("key_symbols", [])
        for sym in key_syms:
            label = sym.get("symbol", "").lower().strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1

    items = [{"label": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:max_items]


def build_visual_context_for_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare front-end-friendly visual structures from the existing analysis dict.
    - emotional_arc: list of {index, stage, emotion, intensity}
    - symbol_graph: {nodes, edges}
    """
    # Emotional arc → simple timeline
    emo_arc = analysis.get("emotional_arc", []) or []
    emotional_arc_visual = []
    for idx, st in enumerate(emo_arc):
        emotional_arc_visual.append(
            {
                "index": idx,
                "stage": st.get("stage", ""),
                "emotion": st.get("emotion", ""),
                "intensity": st.get("intensity", 0.0),
            }
        )

    # Symbol graph
    key_syms = analysis.get("key_symbols", []) or []
    relations = analysis.get("symbol_relations", []) or []

    nodes = []
    for s in key_syms:
        label = s.get("symbol", "")
        local_count = s.get("local_count", 0)
        global_count = s.get("global_count", 0)
        nodes.append(
            {
                "id": label,
                "label": label,
                "weight": local_count + global_count,
            }
        )

    edges = []
    for r in relations:
        src = r.get("source", "")
        tgt = r.get("target", "")
        rel = r.get("relation", "")
        if not src or not tgt:
            continue
        edges.append({"source": src, "target": tgt, "relation": rel})

    return {
        "emotional_arc_visual": emotional_arc_visual,
        "symbol_graph": {"nodes": nodes, "edges": edges},
    }


# ---------------------------
# Routes
# ---------------------------


@app.route("/", methods=["GET"])
def index():
    """
    Show the dream input form + possibly a single existing dream (for quick testing).
    """
    # Optionally allow ?prefill=N to load a past dream into the form
    prefill_idx = request.args.get("prefill")
    prefill_data = None
    if prefill_idx is not None:
        try:
            idx = int(prefill_idx)
        except ValueError:
            idx = None

        if idx is not None:
            records = read_logs()
            if 0 <= idx < len(records):
                rec = records[idx]
                inp = rec.get("input", {})
                prefill_data = {
                    "title": inp.get("title", ""),
                    "dream_text": inp.get("dream_text", ""),
                    "felt_during": inp.get("felt_during", ""),
                    "felt_after": inp.get("felt_after", ""),
                    "life_context": inp.get("life_context", ""),
                }

    return render_template("index.html", prefill=prefill_data)


@app.route("/decode", methods=["POST"])
def decode():
    """
    Handle form submission, run analysis, and show result.
    """
    dream_text = request.form.get("dream_text", "").strip()
    title = request.form.get("title", "").strip()
    felt_during = request.form.get("felt_during", "").strip()
    felt_after = request.form.get("felt_after", "").strip()
    life_context = request.form.get("life_context", "").strip()

    if not dream_text:
        abort(400, "Dream text is required.")

    try:
        analysis = analyze_dream(
            dream_text=dream_text,
            title=title,
            felt_during=felt_during,
            felt_after=felt_after,
            life_context=life_context,
        )
    except Exception as e:
        # TEMP: show the actual error instead of a blank 500.
        # Once things are stable, you can remove this and rely on normal error logging.
        return (
            f"<h1>Decode error</h1>"
            f"<p><strong>{type(e).__name__}</strong>: {e}</p>",
            500,
        )

    # Build a serializable dict for logging and template
    analysis_dict = {
        "micronarrative": analysis.micronarrative,
        "summary": analysis.summary,
        "interpretive_narrative": analysis.interpretive_narrative,
        "key_symbols": [
            {
                "symbol": s.symbol,
                "description": s.description,
                "possible_meanings": s.possible_meanings,
                "confidence": s.confidence,
                "local_count": s.local_count,
                "global_count": s.global_count,
            }
            for s in analysis.key_symbols
        ],
        "emotional_profile_primary": [
            {"name": e.name, "intensity": e.intensity}
            for e in analysis.emotional_profile_primary
        ],
        "emotional_profile_tone": analysis.emotional_profile_tone,
        "emotional_arc": [
            {
                "stage": st.stage,
                "emotion": st.emotion,
                "intensity": st.intensity,
            }
            for st in analysis.emotional_arc
        ],
        "narrative_pattern": {
            "pattern_name": analysis.narrative_pattern.pattern_name,
            "description": analysis.narrative_pattern.description,
            "related_themes": analysis.narrative_pattern.related_themes,
        },
        "symbol_relations": [
            {
                "source": r.source,
                "target": r.target,
                "relation": r.relation,
            }
            for r in analysis.symbol_relations
        ],
        "reflection_prompts": analysis.reflection_prompts,
        "cautions": analysis.cautions,
        "detected_keywords": analysis.detected_keywords,
    }

    # Log the dream + analysis
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": {
            "title": title,
            "dream_text": dream_text,
            "felt_during": felt_during,
            "felt_after": felt_after,
            "life_context": life_context,
            "detected_keywords": analysis.detected_keywords,
        },
        "analysis": analysis_dict,
    }
    append_log(record)

    # Attach visual context for the UI
    visual_ctx = build_visual_context_for_analysis(analysis_dict)

    return render_template(
        "result.html",
        analysis=analysis_dict,
        visual_ctx=visual_ctx,
    )


@app.route("/history")
def history():
    """
    Show a simple list of past dreams from the log (most recent first).
    """
    records = read_logs()
    records = list(reversed(records))  # newest first

    indexed = []
    for idx, rec in enumerate(records):
        inp = rec.get("input", {})
        indexed.append(
            {
                "idx": idx,
                "timestamp": rec.get("timestamp", ""),
                "title": inp.get("title", ""),
                "dream_text": inp.get("dream_text", ""),
                "felt_during": inp.get("felt_during", ""),
                "felt_after": inp.get("felt_after", ""),
                "life_context": inp.get("life_context", ""),
            }
        )

    symbol_freq = build_symbol_frequency_from_logs()

    return render_template(
        "history.html",
        records=indexed,
        symbol_freq=symbol_freq,
    )


@app.route("/history/<int:idx>")
def history_detail(idx: int):
    """
    Show details for a single logged dream.
    """
    records = read_logs()
    records = list(reversed(records))  # newest first

    if idx < 0 or idx >= len(records):
        abort(404, "Dream not found.")

    rec = records[idx]
    analysis = rec.get("analysis", {})
    visual_ctx = build_visual_context_for_analysis(analysis)

    return render_template(
        "result.html",
        analysis=analysis,
        visual_ctx=visual_ctx,
    )


@app.route("/search")
def search():
    """
    Search across dream history.
    """
    query = request.args.get("q", "").strip()
    results = search_logs(query)
    symbol_freq = build_symbol_frequency_from_logs()

    indexed = []
    for idx, rec in enumerate(results):
        inp = rec.get("input", {})
        indexed.append(
            {
                "idx": rec.get("_history_idx", idx),
                "timestamp": rec.get("timestamp", ""),
                "title": inp.get("title", ""),
                "dream_text": inp.get("dream_text", ""),
                "felt_during": inp.get("felt_during", ""),
                "felt_after": inp.get("felt_after", ""),
                "life_context": inp.get("life_context", ""),
            }
        )

    return render_template(
        "search.html",
        query=query,
        records=indexed,
        symbol_freq=symbol_freq,
    )


# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
