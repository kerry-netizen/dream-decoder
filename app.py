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
    """
    Normalize a symbol for lexicon lookup:
    - lowercase
    - strip whitespace
    """
    return (symbol or "").strip().lower()


def lookup_symbol_in_lexicon(symbol: str) -> Dict[str, Any]:
    """
    Try to find symbol in lexicon, with fallbacks:
    - Exact phrase
    - If phrase has adjectives (like 'black cat'), also try last word ('cat')
    - Simple singular form for plurals
    """
    key = normalize_symbol_key(symbol)
    if not key:
        return {}
    if key in LEXICON:
        return LEXICON[key]

    parts = key.split()
    # Try last word for multi-word phrases
    if len(parts) > 1:
        last = parts[-1]
        if last in LEXICON:
            return LEXICON[last]
        # singularize last word
        if last.endswith("s"):
            singular = last[:-1]
            if singular in LEXICON:
                return LEXICON[singular]

    # Single-word plural
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
            # Multi-word phrase: exact phrase with non-word boundaries
            pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
        else:
            # Single word: match the word with optional plural 's'
            # e.g., 'rat' should match 'rat' and 'rats' but not 'congratulations'
            pattern = r'\b' + re.escape(kw) + r's?\b'

        if re.search(pattern, lowered):
            found.append(kw)

    # Deduplicate while preserving order
    return list(dict.fromkeys(found))


# ---------------------------
# spaCy-lite candidate symbol extraction
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
    Very basic noun-ish phrase extractor:
    - Captures capitalized words (names) as symbols.
    - Captures color + noun bigrams (e.g., "black cat").
    - Adds dream keyword phrases with counts.
    Returns a list of {"phrase": str, "count": int}, sorted by count desc.
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
        # Only count full phrase occurrences to avoid junk
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
    Slightly smarter occurrence count:
    - Try exact phrase.
    - If that fails and it's multi-word, try reversed order.
    - As a last resort, count content-word hits.
    """
    if not phrase:
        return 0
    t = text.lower()
    p = phrase.lower().strip()
    if not p:
        return 0

    # Exact phrase
    exact = t.count(p)
    if exact > 0:
        return exact

    words = p.split()
    if len(words) > 1:
        rev = " ".join(reversed(words))
        rev_count = t.count(rev)
        if rev_count > 0:
            return rev_count

    # Fallback: sum counts of non-stopword tokens
    content_words = [w for w in words if w not in STOPWORDS]
    if not content_words:
        return 0
    return sum(t.count(w) for w in content_words)


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
class NarrativePattern:
    pattern_name: str
    description: str
    related_themes: List[str]


@dataclass
class DreamAnalysis:
    summary: str
    interpretive_narrative: str
    key_symbols: List[SymbolMeaning]
    emotional_profile_primary: List[Emotion]
    emotional_profile_tone: str
    narrative_pattern: NarrativePattern
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

Important rules:

- Treat combined phrases ("black cat", "red car") as single symbols.
- Pay attention to animals, colors, family members, and contact actions.
- Integrate the provided life_context into symbol interpretation.
- You will receive:
  - "detected_keywords": common dream motifs automatically found in the text.
  - "candidate_symbols": a list of phrases with counts that appear important.
  - "lexicon_entries": structured notes from a dream symbol lexicon, including
    themes and notes for some symbols.
  You MUST consider these when deciding which key symbols and narrative patterns
  to emphasize. Either include them as symbols or clearly take them into account
  in the narrative (internally).

- When symbols are culturally loaded (e.g. black cat, snake, storm, wedding),
  give multiple possible meanings and note that meanings vary by culture and
  personal experience.

- When people appear in the dream, focus the symbol labels on their ROLE,
  ACTIONS, or RELATIONSHIP to the dreamer (e.g. "warning figure", "authority
  figure", "child asking for help") rather than on demographic traits such as
  race, ethnicity, body size, or age. You may still mention those traits in the
  description if they are clearly central to the dreamer's emotional reaction,
  but avoid making them the core symbolic label.

You must produce TWO different layers:

1) A SHORT SUMMARY (field: "summary")
   - 3–6 sentences.
   - Neutral recap of what happens in the dream, in simple language.

2) An INTERPRETIVE NARRATIVE (field: "interpretive_narrative")
   - 1–3 paragraphs.
   - Weave together the symbols, emotions, narrative pattern, detected_keywords,
     candidate_symbols, lexicon_entries, and life_context into a coherent
     psychological story.
   - Use plain, accessible language.
   - Speak in the second person ("you") and use soft, tentative phrasing
     ("this may suggest...", "it could be that...", "one way to see this is...").
   - Do NOT sound mystical or prophetic. Stay grounded and possibility-based.
   - Do NOT diagnose or provide therapy. This is reflective, not clinical.

Output ONLY valid JSON with this structure:

{
  "summary": "...",
  "interpretive_narrative": "...",
  "key_symbols": [
    {
      "symbol": "...",
      "description": "...",
      "possible_meanings": ["..."],
      "confidence": 0.0
    }
  ],
  "emotional_profile": {
    "primary_emotions": [
      {"name": "...", "intensity": 0.0}
    ],
    "overall_tone": "..."
  },
  "narrative_pattern": {
    "pattern_name": "...",
    "description": "...",
    "related_themes": ["..."]
  },
  "reflection_prompts": ["..."],
  "cautions": ["..."]
}

Never output explanations outside the JSON object.
"""


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

    # Build lexicon context for candidate symbols
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
    # Keep it small to avoid bloating the prompt
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
            "summary": "Parsing error.",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "narrative_pattern": {
                "pattern_name": "Unknown",
                "description": "",
                "related_themes": [],
            },
            "reflection_prompts": [],
            "cautions": ["Model output could not be parsed."],
        }

    # Map JSON → dataclasses
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

    pattern_raw = data.get("narrative_pattern", {})
    narrative = NarrativePattern(
        pattern_name=pattern_raw.get("pattern_name", "") or "",
        description=pattern_raw.get("description", "") or "",
        related_themes=pattern_raw.get("related_themes", []) or [],
    )

    return DreamAnalysis(
        summary=data.get("summary", "") or "",
        interpretive_narrative=data.get("interpretive_narrative", "") or "",
        key_symbols=key_symbols,
        emotional_profile_primary=emotions,
        emotional_profile_tone=emo_block.get("overall_tone", "") or "",
        narrative_pattern=narrative,
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
            "narrative_pattern": {
                "pattern_name": analysis.narrative_pattern.pattern_name,
                "description": analysis.narrative_pattern.description,
                "related_themes": analysis.narrative_pattern.related_themes,
            },
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

        return render_template(
            "result.html",
            title=title,
            dream_text=dream_text,
            analysis=analysis_dict,
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
    analysis = rec.get("analysis", {})

    title = inp.get("title", "(untitled)")
    dream_text = inp.get("dream_text", "")

    return render_template(
        "result.html",
        title=title,
        dream_text=dream_text,
        analysis=analysis,
    )


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    app.run(debug=False)
