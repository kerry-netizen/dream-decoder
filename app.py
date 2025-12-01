import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, abort
from openai import OpenAI

# Initialize Flask app and OpenAI client
app = Flask(__name__)
client = OpenAI()

# ---------------------------
# Logging config
# ---------------------------

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "dreams.jsonl")


def log_dream(input_payload: dict, analysis_payload: dict) -> None:
    """Append a single dream record to logs/dreams.jsonl."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input": input_payload,
            "analysis": analysis_payload,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging should never break the app
        pass


def read_logs(max_entries: int = 1000) -> List[dict]:
    """
    Read dream records from the JSONL log file.
    Returns up to max_entries, oldest to newest.
    """
    records: List[dict] = []
    if not os.path.exists(LOG_FILE):
        return records

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(data)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    if len(records) > max_entries:
        records = records[-max_entries:]
    return records


def compute_symbol_stats_from_logs() -> Dict[str, int]:
    """
    Build a simple global frequency table for symbols across all logged dreams.
    Key: symbol text (lowercased)
    Value: number of dreams that included that symbol.
    """
    stats: Dict[str, int] = {}
    records = read_logs()
    for rec in records:
        analysis = rec.get("analysis", {})
        key_syms = analysis.get("key_symbols", [])
        seen_in_this_dream = set()
        for s in key_syms:
            sym = (s.get("symbol") or "").strip()
            if not sym:
                continue
            key = sym.lower()
            if key not in seen_in_this_dream:
                seen_in_this_dream.add(key)
        for key in seen_in_this_dream:
            stats[key] = stats.get(key, 0) + 1
    return stats


# ---------------------------
# Phase 5 helpers (visuals)
# ---------------------------

def compute_motif_frequencies_from_logs(max_items: int = 12) -> List[Dict[str, Any]]:
    """
    Aggregate how often each detected keyword appears across all logged dreams.
    Used for the motif frequency chart.
    """
    records = read_logs()
    counts: Dict[str, int] = {}

    for rec in records:
        analysis = rec.get("analysis", {}) or {}
        kws = analysis.get("detected_keywords") or []
        for kw in kws:
            label = (kw or "").strip()
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
    # Emotional arc → simple sequence
    raw_arc = analysis.get("emotional_arc") or []
    emotional_arc = []
    for idx, stage in enumerate(raw_arc):
        emotional_arc.append(
            {
                "index": idx,
                "stage": stage.get("stage") or f"Stage {idx + 1}",
                "emotion": stage.get("emotion") or "",
                "intensity": float(stage.get("intensity") or 0.0),
            }
        )

    # Symbol graph nodes from key_symbols
    key_symbols = analysis.get("key_symbols") or []
    max_nodes = 10
    nodes = []
    for s in key_symbols[:max_nodes]:
        sym_name = (s.get("symbol") or "").strip()
        if not sym_name:
            continue
        importance = float(s.get("confidence", 0.0) or 0.0)
        nodes.append(
            {
                "id": sym_name,
                "label": sym_name,
                "importance": importance,
            }
        )

    node_ids = {n["id"] for n in nodes}

    # Edges from symbol_relations
    relations = analysis.get("symbol_relations") or []
    edges = []
    for rel in relations:
        src = (rel.get("source") or "").strip()
        tgt = (rel.get("target") or "").strip()
        if not src or not tgt:
            continue
        if src not in node_ids or tgt not in node_ids:
            continue
        edges.append(
            {
                "source": src,
                "target": tgt,
                "relation": rel.get("relation") or "",
            }
        )

    return {
        "emotional_arc": emotional_arc,
        "symbol_graph": {
            "nodes": nodes,
            "edges": edges,
        },
    }


# ---------------------------
# Phase 6 helper: search across logs
# ---------------------------

def search_records(query: str, max_results: int = 100) -> List[dict]:
    """
    Naive full-text search across logged dreams.
    Searches title, dream_text, life_context, detected_keywords,
    key_symbols, summary, interpretive_narrative, narrative_pattern.
    Returns newest-first matches with the same idx mapping as /history.
    """
    q = (query or "").strip()
    if not q:
        return []

    q_low = q.lower()
    records = read_logs()
    records = list(reversed(records))  # newest first, same as /history

    results: List[dict] = []
    for idx, rec in enumerate(records):
        inp = rec.get("input", {}) or {}
        analysis = rec.get("analysis", {}) or {}

        title = inp.get("title", "") or ""
        dream_text = inp.get("dream_text", "") or ""
        life_context = inp.get("life_context", "") or ""
        felt_during = inp.get("felt_during", "") or ""
        felt_after = inp.get("felt_after", "") or ""

        detected_keywords = " ".join(analysis.get("detected_keywords") or [])
        key_symbol_names = " ".join(
            (s.get("symbol") or "") for s in (analysis.get("key_symbols") or [])
        )
        summary = analysis.get("summary", "") or ""
        interpretive = analysis.get("interpretive_narrative", "") or ""
        pattern_name = (analysis.get("narrative_pattern", {}) or {}).get("pattern_name", "") or ""

        haystack = " ".join(
            [
                title,
                dream_text,
                life_context,
                felt_during,
                felt_after,
                detected_keywords,
                key_symbol_names,
                summary,
                interpretive,
                pattern_name,
            ]
        ).lower()

        if q_low in haystack:
            snippet = dream_text[:180]
            if len(dream_text) > 180:
                snippet += "…"
            results.append(
                {
                    "idx": idx,
                    "timestamp": rec.get("timestamp", ""),
                    "title": title or "(untitled)",
                    "felt_during": felt_during,
                    "felt_after": felt_after,
                    "snippet": snippet,
                }
            )

        if len(results) >= max_results:
            break

    return results


# ---------------------------
# Load symbol lexicon
# ---------------------------

def load_symbol_lexicon() -> Dict[str, Dict[str, Any]]:
    """
    Load symbol_lexicon.json from the project directory.
    Keys are lowercased.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "symbol_lexicon.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        lex = {}
        for k, v in raw.items():
            lex[k.lower()] = v
        return lex
    except Exception:
        return {}


LEXICON = load_symbol_lexicon()


def normalize_symbol_key(symbol: str) -> str:
    """Normalize a symbol for lexicon lookup."""
    return (symbol or "").strip().lower()


def lookup_symbol_in_lexicon(symbol: str) -> Dict[str, Any]:
    """
    Try to find symbol in lexicon, with fallbacks:
    - Exact phrase
    - Last word of multi-word phrases
    - Simple plural stripping
    """
    key = normalize_symbol_key(symbol)
    if not key:
        return {}
    if key in LEXICON:
        return LEXICON[key]

    parts = key.split()
    if len(parts) > 1:
        last = parts[-1]
        if last in LEXICON:
            return LEXICON[last]
        if last.endswith("s"):
            singular = last[:-1]
            if singular in LEXICON:
                return LEXICON[singular]

    if len(parts) == 1 and key.endswith("s"):
        singular = key[:-1]
        if singular in LEXICON:
            return LEXICON[singular]

    return {}


# ---------------------------
# Dream keyword parsing
# ---------------------------

DREAM_KEYWORDS = [
    # Situations
    "falling", "flying", "being chased", "chased", "running away", "lost",
    "trapped", "late for", "missing a test", "exam", "test", "naked in public",
    "embarrassed", "argument", "fight", "war", "accident", "car crash", "crash",
    "drowning", "suffocating", "unable to speak", "cannot move", "paralyzed",
    "searching", "looking for", "forgot something", "locked out",

    # Body / health
    "teeth falling out", "losing teeth", "hair falling out", "bleeding",
    "injury", "sick", "illness", "pregnant", "pregnancy", "baby", "birth",
    "dying", "death", "dead body", "funeral",

    # Animals
    "dog", "dogs", "cat", "cats", "black cat", "snake", "snakes",
    "spider", "spiders", "rat", "rats", "wolf", "wolves", "bear",
    "lion", "tiger", "bird", "birds", "fish", "shark", "insect", "bugs",

    # People
    "mother", "father", "mom", "dad", "child", "children",
    "teacher", "boss", "stranger", "intruder",

    # Places / vehicles
    "house", "home", "childhood home", "school", "office", "hospital",
    "church", "forest", "woods", "ocean", "sea", "lake", "river",
    "mountain", "stairs", "elevator", "car", "bus", "train", "plane",
    "airplane", "boat", "ship",

    # Events / objects
    "wedding", "marriage", "engagement", "party", "celebration",
    "storm", "tornado", "earthquake", "fire", "flood", "explosion",
    "weapon", "gun", "knife", "blood"
]


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


# ---------------------------
# Candidate symbols, priority, etc.
# ---------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "from", "with", "without", "into", "out", "by", "about", "as",
    "is", "was", "were", "are", "be", "been", "being", "that", "this",
    "it", "its", "he", "she", "they", "them", "we", "us", "you", "i",
    "my", "your", "our", "their", "his", "her"
}

COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "purple",
    "orange", "pink", "brown", "gray", "grey", "gold", "silver"
}


def extract_candidate_symbols(text: str) -> List[dict]:
    """
    Very basic noun-ish phrase extractor.
    """
    tokens = re.findall(r"[A-Za-z']+", text)
    lower = [t.lower() for t in tokens]
    candidates: Dict[str, int] = {}

    # Capitalized words (names, places)
    for i, tok in enumerate(tokens):
        lw = lower[i]
        if lw in STOPWORDS or len(lw) <= 2:
            continue
        if tok[0].isupper() and i != 0:
            candidates[tok] = candidates.get(tok, 0) + 1

    # Color + noun bigrams
    for i in range(len(tokens) - 1):
        lw1, lw2 = lower[i], lower[i + 1]
        if lw1 in COLORS and lw2 not in STOPWORDS:
            phrase = f"{tokens[i]} {tokens[i + 1]}"
            candidates[phrase] = candidates.get(phrase, 0) + 1

    # Add keywords as phrases with counts
    lowered_text = text.lower()
    for kw in DREAM_KEYWORDS:
        if " " in kw:
            pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
        else:
            pattern = r'\b' + re.escape(kw) + r's?\b'
        matches = re.findall(pattern, lowered_text)
        count = len(matches)
        if count > 0:
            candidates[kw] = candidates.get(kw, 0) + count

    out = [{"phrase": k, "count": v} for k, v in candidates.items()]
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:15]


def count_occurrences(text: str, phrase: str) -> int:
    """
    Slightly smarter occurrence count.
    """
    if not phrase:
        return 0
    t = text.lower()
    p = phrase.lower().strip()
    if not p:
        return 0

    exact = t.count(p)
    if exact > 0:
        return exact

    words = p.split()
    if len(words) > 1:
        rev = " ".join(reversed(words))
        rev_count = t.count(rev)
        if rev_count > 0:
            return rev_count

    content_words = [w for w in words if w not in STOPWORDS]
    if not content_words:
        return 0
    return sum(t.count(w) for w in content_words)


def compute_priority_symbols(
    candidate_symbols: List[dict],
    global_symbol_stats: Dict[str, int],
) -> List[dict]:
    """
    importance = local_count*2 + global_count*0.5 + lexicon_bonus
    """
    scored: List[dict] = []
    for cs in candidate_symbols:
        phrase = cs.get("phrase", "")
        local_count = cs.get("count", 0)
        key = phrase.lower()
        global_count = global_symbol_stats.get(key, 0)
        lex_info = lookup_symbol_in_lexicon(phrase)
        lex_bonus = 1.0 if lex_info else 0.0
        importance = local_count * 2.0 + global_count * 0.5 + lex_bonus
        scored.append(
            {
                "symbol": phrase,
                "local_count": local_count,
                "global_count": global_count,
                "importance": importance,
                "has_lexicon": bool(lex_info),
            }
        )

    scored.sort(key=lambda x: x["importance"], reverse=True)
    return scored[:8]


# ---------------------------
# Data Models
# ---------------------------

@dataclass
class SymbolMeaning:
    symbol: str
    description: str
    possible_meanings: List[str]
    confidence: float
    local_count: int = 0
    global_count: int = 0


@dataclass
class Emotion:
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
    summary: str
    micronarrative: str
    interpretive_narrative: str
    key_symbols: List[SymbolMeaning]
    emotional_profile_primary: List[Emotion]
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
[... trimmed for brevity in this explanation, keep your existing SYSTEM_PROMPT here unchanged ...]
"""

# (Keep your existing SYSTEM_PROMPT body exactly as before.)


# ---------------------------
# Analyzer
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
            "symbol_relations": [],
            "narrative_pattern": {
                "pattern_name": "Unknown",
                "description": "",
                "related_themes": [],
            },
            "reflection_prompts": [],
            "cautions": ["Model output could not be parsed."],
        }

    key_symbols: List[SymbolMeaning] = []
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

    emo_block = data.get("emotional_profile", {})
    emotions: List[Emotion] = []
    for e in emo_block.get("primary_emotions", []):
        emotions.append(
            Emotion(
                name=e.get("name", "") or "",
                intensity=float(e.get("intensity", 0.0) or 0.0),
            )
        )

    arc_block = data.get("emotional_arc", []) or []
    emotional_arc: List[EmotionalStage] = []
    for stage in arc_block:
        emotional_arc.append(
            EmotionalStage(
                stage=stage.get("stage", "") or "",
                emotion=stage.get("emotion", "") or "",
                intensity=float(stage.get("intensity", 0.0) or 0.0),
            )
        )

    rel_block = data.get("symbol_relations", []) or []
    symbol_relations: List[SymbolRelation] = []
    for rel in rel_block:
        symbol_relations.append(
            SymbolRelation(
                source=rel.get("source", "") or "",
                target=rel.get("target", "") or "",
                relation=rel.get("relation", "") or "",
            )
        )

    pattern_raw = data.get("narrative_pattern", {})
    narrative = NarrativePattern(
        pattern_name=pattern_raw.get("pattern_name", "") or "",
        description=pattern_raw.get("description", "") or "",
        related_themes=pattern_raw.get("related_themes", []) or [],
    )

    return DreamAnalysis(
        summary=data.get("summary", "") or "",
        micronarrative=data.get("micronarrative", "") or "",
        interpretive_narrative=data.get("interpretive_narrative", "") or "",
        key_symbols=key_symbols,
        emotional_profile_primary=emotions,
        emotional_profile_tone=emo_block.get("overall_tone", "") or "",
        emotional_arc=emotional_arc,
        narrative_pattern=narrative,
        symbol_relations=symbol_relations,
        reflection_prompts=data.get("reflection_prompts", []) or [],
        cautions=data.get("cautions", []) or [],
        detected_keywords=detected,
    )


# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form.get("dream_title", "").strip()
        dream_text = request.form.get("dream_text", "").strip()
        felt_during = request.form.get("felt_during", "")
        felt_after = request.form.get("felt_after", "")
        life_context = request.form.get("life_context", "").strip()

        analysis = analyze_dream(
            dream_text=dream_text,
            title=title,
            felt_during=felt_during,
            felt_after=felt_after,
            life_context=life_context,
        )

        analysis_dict = {
            "summary": analysis.summary,
            "micronarrative": analysis.micronarrative,
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

        input_payload = {
            "title": title,
            "dream_text": dream_text,
            "felt_during": felt_during,
            "felt_after": felt_after,
            "life_context": life_context,
        }
        log_dream(input_payload, analysis_dict)

        visual = build_visual_context_for_analysis(analysis_dict)
        motif_stats = compute_motif_frequencies_from_logs()

        return render_template(
            "result.html",
            title=title,
            dream_text=dream_text,
            analysis=analysis_dict,
            visual=visual,
            motif_stats=motif_stats,
            from_history=False,
        )

    return render_template("index.html")


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
                "title": inp.get("title", "(untitled)"),
                "felt_during": inp.get("felt_during", ""),
                "felt_after": inp.get("felt_after", ""),
            }
        )

    return render_template("history.html", records=indexed)


@app.route("/history/<int:idx>")
def history_detail(idx: int):
    """
    Show a single logged dream by index (from newest-first list).
    """
    records = read_logs()
    records = list(reversed(records))

    if idx < 0 or idx >= len(records):
        abort(404)

    rec = records[idx]
    inp = rec.get("input", {})
    analysis = rec.get("analysis", {}) or {}

    title = inp.get("title", "(untitled)")
    dream_text = inp.get("dream_text", "")

    visual = build_visual_context_for_analysis(analysis)
    motif_stats = compute_motif_frequencies_from_logs()

    return render_template(
        "result.html",
        title=title,
        dream_text=dream_text,
        analysis=analysis,
        visual=visual,
        motif_stats=motif_stats,
        from_history=True,
    )


@app.route("/search")
def search():
    """
    Search over dream history.
    """
    query = request.args.get("q", "").strip()
    results = search_records(query) if query else []
    return render_template("search.html", query=query, results=results)


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    app.run(debug=False)
