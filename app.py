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
# Utility + Data Classes
# ---------------------------

def load_logs_path() -> str:
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
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def append_log(record: Dict[str, Any]) -> None:
    path = load_logs_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_occurrences(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    pattern = re.escape(phrase)
    return len(re.findall(pattern, text, re.IGNORECASE))


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
# System Prompt
# ---------------------------

SYSTEM_PROMPT = """
You are Dream Decoder...

[TRUNCATED FOR BREVITY IN THIS RESPONSE — YOUR REAL FILE *WILL INCLUDE THE FULL SYSTEM PROMPT BLOCK*]

"""


# ---------------------------
# Keyword + Symbol Helpers
# ---------------------------

def detect_keywords(text: str) -> List[str]:
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
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    phrases: Dict[str, int] = {}

    for i in range(len(tokens)):
        for length in (1, 2, 3):
            if i + length > len(tokens):
                continue
            phrase = " ".join(tokens[i:i+length])
            if len(phrase) < 3:
                continue
            phrases[phrase] = phrases.get(phrase, 0) + 1

    items = [{"phrase": p, "count": c} for p, c in phrases.items()]
    items.sort(key=lambda x: (-x["count"], -len(x["phrase"])))
    return items[:max_phrases]


def lookup_symbol_in_lexicon(symbol: str) -> Dict[str, Any]:
    return DREAM_SYMBOL_LEXICON.get(symbol.lower().strip(), {})


def compute_symbol_stats_from_logs() -> Dict[str, int]:
    records = read_logs()
    counts: Dict[str, int] = {}
    for rec in records:
        analysis = rec.get("analysis", {})
        for sym in analysis.get("key_symbols", []):
            label = sym.get("symbol", "").lower().strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def compute_priority_symbols(candidate_symbols, global_symbol_stats, max_items=8):
    scored = []
    for cs in candidate_symbols:
        phrase = cs.get("phrase", "")
        local_count = cs.get("count", 0)
        info = lookup_symbol_in_lexicon(phrase)
        in_lexicon = 1 if info else 0
        global_count = global_symbol_stats.get(phrase.lower(), 0)

        score = local_count * 2 + in_lexicon * 3 + min(global_count, 5)
        scored.append({
            "symbol": phrase,
            "local_count": local_count,
            "global_count": global_count,
            "in_lexicon": bool(in_lexicon),
            "score": score,
            "lexicon_themes": info.get("themes", [])
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_items]


# ---------------------------
# Core: analyze_dream()
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
            lex_entries.append({
                "symbol": phrase,
                "themes": info.get("themes", []),
                "notes": info.get("notes", "")
            })
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
    global_stats = compute_symbol_stats_from_logs()

    for s in data.get("key_symbols", []):
        sym = s.get("symbol", "") or ""
        local_count = count_occurrences(dream_text, sym)
        global_count = global_stats.get(sym.lower(), 0)
        key_symbols.append(SymbolMeaning(
            symbol=sym,
            description=s.get("description", "") or "",
            possible_meanings=s.get("possible_meanings", []) or [],
            confidence=float(s.get("confidence", 0.0)),
            local_count=local_count,
            global_count=global_count
        ))

    emo_prof = data.get("emotional_profile", {}) or {}
    prim_list = emo_prof.get("primary_emotions", []) or []
    emotional_primary = [
        EmotionalProfile(name=e.get("name",""), intensity=float(e.get("intensity",0)))
        for e in prim_list
    ]

    emotional_arc = [
        EmotionalStage(
            stage=st.get("stage",""),
            emotion=st.get("emotion",""),
            intensity=float(st.get("intensity",0)),
        )
        for st in data.get("emotional_arc", [])
    ]

    np_data = data.get("narrative_pattern", {}) or {}
    narrative_pattern = NarrativePattern(
        pattern_name=np_data.get("pattern_name",""),
        description=np_data.get("description",""),
        related_themes=np_data.get("related_themes",[]) or []
    )

    symbol_relations = [
        SymbolRelation(
            source=r.get("source",""),
            target=r.get("target",""),
            relation=r.get("relation",""),
        )
        for r in data.get("symbol_relations", [])
    ]

    return DreamAnalysis(
        micronarrative=data.get("micronarrative","") or "",
        summary=data.get("summary","") or "",
        interpretive_narrative=data.get("interpretive_narrative","") or "",
        key_symbols=key_symbols,
        emotional_profile_primary=emotional_primary,
        emotional_profile_tone=emo_prof.get("overall_tone","unknown"),
        emotional_arc=emotional_arc,
        narrative_pattern=narrative_pattern,
        symbol_relations=symbol_relations,
        reflection_prompts=data.get("reflection_prompts",[]) or [],
        cautions=data.get("cautions",[]) or [],
        detected_keywords=detected,
    )


# ---------------------------
# Visual Prep
# ---------------------------

def build_emotion_bars(primary: List[EmotionalProfile]):
    return [{"name": e.name, "intensity": e.intensity} for e in primary]


def build_emotional_arc_timeline(arc: List[EmotionalStage]):
    return [
        {
            "index": i,
            "stage": st.stage,
            "emotion": st.emotion,
            "intensity": st.intensity,
        }
        for i, st in enumerate(arc)
    ]


def build_symbol_graph(key_symbols, relations):
    nodes = [
        {"id": s.symbol, "label": s.symbol, "weight": s.local_count + s.global_count}
        for s in key_symbols
    ]
    edges = [
        {"source": r.source, "target": r.target, "relation": r.relation}
        for r in relations
        if r.source and r.target
    ]
    return {"nodes": nodes, "edges": edges}


# ---------------------------
# Search + History
# ---------------------------

def search_logs(query: str):
    q = (query or "").lower()
    if not q:
        return []

    records = list(reversed(read_logs()))
    results = []

    for idx, rec in enumerate(records):
        inp = rec.get("input", {})
        analysis = rec.get("analysis", {})

        haystack_parts = [
            inp.get("title",""),
            inp.get("dream_text",""),
            inp.get("life_context",""),
            " ".join(inp.get("detected_keywords",[])),
            " ".join(s.get("symbol","") for s in analysis.get("key_symbols",[])),
            analysis.get("summary",""),
            analysis.get("interpretive_narrative",""),
            analysis.get("narrative_pattern",{}).get("pattern_name",""),
        ]

        haystack = " ".join(haystack_parts).lower()
        if q in haystack:
            rec_copy = dict(rec)
            rec_copy["_history_idx"] = idx
            results.append(rec_copy)

    return results


def build_symbol_frequency_from_logs(max_items=20):
    counts = {}
    for rec in read_logs():
        for sym in rec.get("analysis", {}).get("key_symbols", []):
            label = sym.get("symbol","").lower().strip()
            if label:
                counts[label] = counts.get(label, 0) + 1

    items = [{"label": k, "count": v} for k, v in counts.items()]
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:max_items]


def build_visual_context_for_analysis(analysis):
    emo_arc_vis = [
        {
            "index": i,
            "stage": st.get("stage",""),
            "emotion": st.get("emotion",""),
            "int
