"""
Dream Decoder - Multi-User Version
Complete rewrite with user authentication and cross-dream thread analysis.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from openai import OpenAI

# Import our new modules
import database as db
import thread_analyzer
from admin import admin_bp

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Register admin blueprint
app.register_blueprint(admin_bp)


# ----------------------------------------------------
# Jinja Filters
# ----------------------------------------------------

@app.template_filter('friendly_date')
def friendly_date_filter(iso_string):
    """Convert ISO timestamp to friendly format like 'Feb 1, 2026'"""
    if not iso_string:
        return ""
    try:
        # Parse ISO format (handles both with and without microseconds)
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y")
    except (ValueError, AttributeError):
        return iso_string  # Return original if parsing fails


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = None  # Disable default flash - subtitle already says "Log in"


@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'assets'),
        'Dream-Ferret-Logo1.png',
        mimetype='image/png'
    )

client = OpenAI()  # Uses OPENAI_API_KEY


# ----------------------------------------------------
# User class for Flask-Login
# ----------------------------------------------------

class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data["id"]
        self.username = user_data["username"]
        self.created_at = user_data["created_at"]


@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(int(user_id))
    if user_data:
        return User(user_data)
    return None


# ----------------------------------------------------
# Built-in Symbol Lexicon (self-contained)
# ----------------------------------------------------

DREAM_SYMBOL_LEXICON: Dict[str, Dict[str, Any]] = {
    "cat": {"themes": ["independence", "intuition"], "notes": "May point toward emotional independence or boundaries."},
    "dog": {"themes": ["loyalty", "support"], "notes": "Can reflect trust, attachment, or desire for companionship."},
    "snake": {"themes": ["transformation", "fear"], "notes": "May represent change, fear, or something hidden rising up."},
    "spider": {"themes": ["entrapment"], "notes": "Often relates to intricate plans or feeling caught."},
    "rat": {"themes": ["disgust", "hidden problems"], "notes": "Can reflect anxiety about contamination or betrayal."},
    "wolf": {"themes": ["instinct", "threat"], "notes": "Might signal primal drives or feeling targeted."},
    "bear": {"themes": ["power", "rest"], "notes": "May represent strong emotion or need for retreat."},
    "lion": {"themes": ["pride", "authority"], "notes": "Often symbolizes leadership or confidence struggles."},
    "tiger": {"themes": ["anger"], "notes": "Can reflect intense emotion or threat."},
    "bird": {"themes": ["freedom"], "notes": "Desire to rise above circumstances or gain perspective."},
    "fish": {"themes": ["emotion"], "notes": "Emotions beneath the surface."},
    "shark": {"themes": ["threat"], "notes": "Perceived danger or competition."},
    "horse": {"themes": ["drive"], "notes": "Momentum, vitality, or energetic movement forward."},
    "unicorn": {"themes": ["idealism"], "notes": "Longing for purity or magic amid complexity."},
    "baby": {"themes": ["new beginnings"], "notes": "A fragile new feeling, project, or version of self."},
    "child": {"themes": ["innocence", "past self"], "notes": "Your younger self, vulnerability or early coping patterns."},
    "mother": {"themes": ["care"], "notes": "Support, comfort, or emotional smothering."},
    "father": {"themes": ["authority"], "notes": "Judgment, rules, or internal criticism."},
    "stranger": {"themes": ["unknown"], "notes": "Emerging part of self not yet recognized."},
    "guide": {"themes": ["intuition"], "notes": "Inner direction nudging toward growth."},
    "faceless guide": {"themes": ["mystery"], "notes": "Guidance without clarity — a sense of being led."},
    "house": {"themes": ["self"], "notes": "Your internal structure, emotional rooms, or identity."},
    "forest": {"themes": ["mystery"], "notes": "Exploring subconscious territory."},
    "ocean": {"themes": ["depth"], "notes": "Overwhelming emotion or vast intuitive material."},
    "river": {"themes": ["flow"], "notes": "Life direction or momentum."},
    "lake": {"themes": ["containment"], "notes": "Still reflection or emotional quiet."},
    "mountain": {"themes": ["challenge"], "notes": "Big aspirations or obstacles."},
    "car": {"themes": ["control"], "notes": "Agency, autonomy, ability to steer life."},
    "train": {"themes": ["path"], "notes": "Long-term momentum or established route."},
    "plane": {"themes": ["transition"], "notes": "Major shifts or ambitions taking off."},
    "stairs": {"themes": ["progress"], "notes": "Moving between layers of emotion or insight."},
    "door": {"themes": ["opportunity"], "notes": "Thresholds, decisions, new phases."},
    "window": {"themes": ["perspective"], "notes": "Seeing differently."},
    "bridge": {"themes": ["transition"], "notes": "Crossing between internal states."},
    "storm": {"themes": ["conflict"], "notes": "Inner turmoil."},
    "fire": {"themes": ["purification"], "notes": "Burning away the old."},
    "flood": {"themes": ["overwhelm"], "notes": "Overflow of emotion."},
    "mirror": {"themes": ["identity"], "notes": "Self-reflection or distorted self-image."},
    "lost": {"themes": ["confusion"], "notes": "Searching for orientation."},
    "trapped": {"themes": ["pressure"], "notes": "Feeling stuck."},
    "museum": {"themes": ["memory"], "notes": "Curation of past experiences."},
    "letter": {"themes": ["unfinished business"], "notes": "Message or unresolved communication."},
    "starlight": {"themes": ["guidance"], "notes": "Small insights in darkness."},
    "constellation": {"themes": ["connection"], "notes": "Seeing patterns in life events."},
    "glass": {"themes": ["clarity"], "notes": "Transparency or emotional fragility."},
    "cracked glass": {"themes": ["instability"], "notes": "A fragile situation under stress."},
    "alarm": {"themes": ["urgency"], "notes": "Internal alarm or boundary crossed."},
    "gravity shift": {"themes": ["imbalance"], "notes": "Perspective changing, ground shifting."},
    "galaxy face": {"themes": ["mystery"], "notes": "Identity in transformation."},
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

    found, seen = [], set()

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

def simple_candidate_symbols(text: str, max_items=12):
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    stop = {"the", "and", "this", "that", "from", "just", "like", "then", "with", "your"}

    counts = {}
    for tok in tokens:
        if len(tok) < 4 or tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1

    items = [{"phrase": k, "count": v} for k, v in counts.items()]
    return sorted(items, key=lambda x: (-x["count"], -len(x["phrase"])))[:max_items]


def build_priority_symbols(candidates):
    scored = []
    for c in candidates:
        phrase = c["phrase"].lower()
        norm = normalize(phrase)
        lex = DREAM_SYMBOL_LEXICON.get(norm)
        score = c["count"] * 2 + (4 if lex else 0)

        description = None
        if lex:
            description = lex.get("notes")

        scored.append({
            "symbol": norm,
            "original": phrase,
            "description": description,
            "in_lexicon": bool(lex),
            "score": score,
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:10]


# ----------------------------------------------------
# SYSTEM PROMPT
# ----------------------------------------------------

SYSTEM_PROMPT = """
You are Dream Ferret, an evidence-informed, non-mystical interpreter of dreams.

Return VALID JSON.

Formatting:
- "micronarrative": 1–2 sentences only.
- "summary": 3–5 sentences, rich but concise. This is the primary interpretive section.
- "interpretive_narrative": 3–6 paragraphs (deeper dive).
- "key_symbols": for each symbol include:
    { "symbol": "...", "description": "a short interpretive sentence" }
- "emotional_arc": list of:
    { "stage": "beginning/middle/end", "emotion": "string", "intensity": 0.0 }

Be reflective and grounded, not mystical.
Do NOT mention confidence.
Do NOT generate empty placeholder fields.
"""


# ----------------------------------------------------
# LLM CALL
# ----------------------------------------------------

def call_model(payload, user_id=None, endpoint="dream_analysis"):
    """Call OpenAI API and track usage."""
    import time
    start_time = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
        timeout=30,
    )

    # Track API usage
    duration_ms = int((time.time() - start_time) * 1000)
    usage = response.usage
    if usage and user_id:
        db.log_api_usage(
            user_id=user_id,
            endpoint=endpoint,
            tokens_prompt=usage.prompt_tokens,
            tokens_completion=usage.completion_tokens,
            model="gpt-4o-mini",
            duration_ms=duration_ms
        )

    return json.loads(response.choices[0].message.content)


# ----------------------------------------------------
# Normalize emotional fields for template safety
# ----------------------------------------------------

def normalize_emotional_profile(raw):
    profile = raw or {}
    if not isinstance(profile, dict):
        profile = {}
    primary_raw = profile.get("primary_emotions", []) or []
    tone = profile.get("overall_tone", "unknown") or "unknown"
    out = []
    for item in primary_raw:
        if isinstance(item, dict):
            name = item.get("name") or item.get("emotion") or ""
            inten = item.get("intensity", 0.0)
        else:
            name = str(item)
            inten = 0.0
        if not isinstance(inten, (int, float)):
            inten = 0.0
        out.append({"name": name, "intensity": float(inten)})
    return out, tone


def normalize_emotional_arc(raw):
    arc = raw or []
    if not isinstance(arc, list):
        return []
    out = []
    for st in arc:
        if isinstance(st, dict):
            stage = st.get("stage", "")
            emo = st.get("emotion") or st.get("name") or ""
            inten = st.get("intensity", 0.0)
        else:
            stage = ""
            emo = str(st)
            inten = 0.0
        if not isinstance(inten, (int, float)):
            inten = 0.0
        out.append({"stage": stage, "emotion": emo, "intensity": float(inten)})
    return out


# ----------------------------------------------------
# Main analysis
# ----------------------------------------------------

def analyze_dream(dream_text, title="", life_context="", user_id=None):
    detected = detect_keywords(dream_text)
    candidates = simple_candidate_symbols(dream_text)
    priority = build_priority_symbols(candidates)

    payload = {
        "dream_title": title,
        "dream_text": dream_text,
        "life_context": life_context,
        "detected_keywords": detected,
        "candidate_symbols": candidates,
        "priority_symbols": priority,
    }

    try:
        data = call_model(payload, user_id=user_id, endpoint="dream_analysis")
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        print("MODEL ERROR:", err, flush=True)
        # Log the error
        db.log_error(
            error_type=type(e).__name__,
            message=str(e),
            traceback_str=traceback.format_exc(),
            route="/analyze",
            user_id=user_id
        )
        data = {
            "micronarrative": "",
            "summary": f"There was an error contacting the model: {err}",
            "interpretive_narrative": "",
            "key_symbols": [],
            "emotional_profile": {"primary_emotions": [], "overall_tone": "unknown"},
            "emotional_arc": [],
            "narrative_pattern": {},
            "symbol_relations": [],
            "reflection_prompts": [],
            "cautions": [err]
        }

    primary, tone = normalize_emotional_profile(data.get("emotional_profile"))
    arc = normalize_emotional_arc(data.get("emotional_arc"))

    return {
        "micronarrative": data.get("micronarrative", ""),
        "summary": data.get("summary", ""),
        "interpretive_narrative": data.get("interpretive_narrative", ""),
        "key_symbols": data.get("key_symbols", []),
        "emotional_profile_primary": primary,
        "emotional_profile_tone": tone,
        "emotional_arc": arc,
        "narrative_pattern": data.get("narrative_pattern", {}),
        "symbol_relations": data.get("symbol_relations", []),
        "reflection_prompts": data.get("reflection_prompts", []),
        "cautions": data.get("cautions", []),
        "detected_keywords": detected,
    }


# ----------------------------------------------------
# Routes - Authentication
# ----------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        password_confirm = request.form.get("password_confirm", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        if password != password_confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        user_id = db.create_user(username, password)
        if user_id is None:
            flash("Username already exists. Please choose another.", "error")
            return render_template("register.html")

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/disclaimer")
def disclaimer():
    """Display disclaimer and beta notes page."""
    beta_notes = db.get_beta_notes()
    return render_template("disclaimer.html", beta_notes=beta_notes)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user_data = db.authenticate_user(username, password)
        if user_data:
            user = User(user_data)
            login_user(user, remember=True)
            flash(f"Welcome back, {username}!", "success")

            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ----------------------------------------------------
# Routes - Dream Management
# ----------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        return handle_decode()
    dream_count = db.get_user_dream_count(current_user.id)
    return render_template("index.html", dream_count=dream_count)


@app.route("/decode", methods=["GET", "POST"])
@login_required
def decode():
    if request.method == "POST":
        return handle_decode()
    return redirect(url_for("index"))


def handle_decode():
    dream_text = request.form.get("dream_text", "").strip()
    dream_title = request.form.get("dream_title", "").strip()
    life_context = request.form.get("life_context", "").strip()

    if not dream_text:
        flash("Please enter a dream to decode.", "error")
        return render_template("index.html")

    # Analyze dream
    analysis = analyze_dream(
        dream_text=dream_text,
        title=dream_title,
        life_context=life_context,
        user_id=current_user.id
    )

    # Save to database
    dream_id = db.save_dream(
        user_id=current_user.id,
        title=dream_title,
        dream_text=dream_text,
        felt_during="",
        felt_after="",
        life_context=life_context,
        analysis=analysis
    )

    # Check if we should trigger thread analysis
    dream_count = db.get_user_dream_count(current_user.id)
    if dream_count >= 5 and dream_count % 5 == 0:
        # Trigger thread analysis in background
        try:
            dreams = db.get_user_dreams(current_user.id)
            threads, meta_analysis = thread_analyzer.analyze_dream_threads(dreams)

            # Save threads
            for thread in threads:
                db.save_dream_thread(
                    user_id=current_user.id,
                    thread_name=thread.get("thread_name", "Unnamed Thread"),
                    description=thread.get("description", ""),
                    recurring_symbols=thread.get("recurring_symbols", []),
                    emotional_pattern=thread.get("emotional_pattern", ""),
                    narrative_arc=thread.get("narrative_arc", ""),
                    dream_ids=thread.get("dream_ids", [])
                )

            # Save meta-analysis
            if meta_analysis:
                db.save_meta_analysis(
                    user_id=current_user.id,
                    total_dreams=dream_count,
                    top_symbols=[s["symbol"] for s in meta_analysis.get("recurring_patterns", {}).get("recurring_symbols", [])[:10]],
                    emotional_trends=meta_analysis.get("recurring_patterns", {}).get("emotional_patterns", {}),
                    narrative_patterns=meta_analysis.get("psychological_themes", []),
                    insights=meta_analysis.get("overall_theme", ""),
                    full_analysis=meta_analysis
                )

            session["new_threads_available"] = True

        except Exception as e:
            print(f"Thread analysis error: {e}")

    # Add metadata to analysis for display
    analysis["dream_id"] = dream_id
    analysis["dream_title"] = dream_title
    analysis["dream_text"] = dream_text
    analysis["life_context"] = life_context

    # Find similar past dreams based on shared symbols
    new_symbols = analysis.get("detected_keywords", [])
    for sym in analysis.get("key_symbols", []):
        if isinstance(sym, dict) and "symbol" in sym:
            new_symbols.append(sym["symbol"])
    similar_dreams = db.find_similar_dreams(current_user.id, new_symbols, exclude_dream_id=dream_id)

    return render_template("result.html", analysis=analysis, similar_dreams=similar_dreams)


@app.route("/history")
@login_required
def history():
    """Display all logged dreams for the current user."""
    dreams = db.get_user_dreams(current_user.id)

    # Format dreams for display
    formatted_dreams = []
    for dream in dreams:
        title = dream.get("title", "").strip()
        if not title:
            # Generate title from first line
            dream_text = dream.get("dream_text", "")
            first_line = dream_text.split("\n")[0][:50]
            title = first_line + "..." if len(first_line) == 50 else first_line
            if not title:
                title = "Untitled Dream"

        formatted_dreams.append({
            "id": dream["id"],
            "title": title,
            "timestamp": dream.get("timestamp", ""),
        })

    return render_template("history.html", records=formatted_dreams)


@app.route("/history/<int:dream_id>")
@login_required
def history_detail(dream_id):
    """Display a specific dream's full analysis."""
    dream = db.get_dream_by_id(dream_id, current_user.id)

    if not dream:
        flash("Dream not found.", "error")
        return redirect(url_for("history"))

    analysis = dream.get("analysis", {})
    analysis["dream_id"] = dream["id"]
    analysis["dream_title"] = dream.get("title", "")
    analysis["dream_text"] = dream.get("dream_text", "")
    analysis["life_context"] = dream.get("life_context", "")
    analysis["timestamp"] = dream.get("timestamp", "")

    return render_template("result.html", analysis=analysis)


@app.route("/history/<int:dream_id>/delete", methods=["POST"])
@login_required
def delete_dream(dream_id):
    """Delete a dream and clear cached threads/meta for regeneration."""
    deleted = db.delete_dream(dream_id, current_user.id)

    if deleted:
        # Clear threads and meta-analysis so they regenerate with updated data
        db.clear_user_threads(current_user.id)
        db.clear_user_meta_analysis(current_user.id)
        flash("Dream deleted successfully.", "success")
    else:
        flash("Dream not found or you don't have permission to delete it.", "error")

    return redirect(url_for("history"))


@app.route("/history/<int:dream_id>/update-title", methods=["POST"])
@login_required
def update_dream_title(dream_id):
    """Update a dream's title."""
    new_title = request.form.get("title", "").strip()

    if not new_title:
        flash("Title cannot be empty.", "error")
        return redirect(url_for("history_detail", dream_id=dream_id))

    updated = db.update_dream_title(dream_id, current_user.id, new_title)

    if updated:
        flash("Title updated.", "success")
    else:
        flash("Could not update title.", "error")

    return redirect(url_for("history_detail", dream_id=dream_id))


@app.route("/export/dream/<int:dream_id>")
@login_required
def export_dream(dream_id):
    """Export a single dream as text file."""
    dream = db.get_dream_by_id(dream_id, current_user.id)

    if not dream:
        flash("Dream not found.", "error")
        return redirect(url_for("history"))

    analysis = dream.get("analysis", {})
    title = dream.get("title", "Untitled Dream")
    timestamp = dream.get("timestamp", "")

    # Build text content
    lines = [
        "=" * 60,
        f"DREAM: {title}",
        f"Date: {timestamp[:10] if timestamp else 'Unknown'}",
        "=" * 60,
        "",
        "DREAM TEXT:",
        "-" * 40,
        dream.get("dream_text", ""),
        "",
    ]

    if dream.get("life_context"):
        lines.extend([
            "LIFE CONTEXT:",
            "-" * 40,
            dream.get("life_context", ""),
            "",
        ])

    lines.extend([
        "SUMMARY:",
        "-" * 40,
        analysis.get("summary", "No summary available."),
        "",
    ])

    if analysis.get("detected_keywords"):
        lines.extend([
            "DETECTED MOTIFS:",
            "-" * 40,
            ", ".join(analysis.get("detected_keywords", [])),
            "",
        ])

    if analysis.get("key_symbols"):
        lines.append("KEY SYMBOLS:")
        lines.append("-" * 40)
        for sym in analysis.get("key_symbols", []):
            if isinstance(sym, dict):
                lines.append(f"• {sym.get('symbol', 'Unknown')}: {sym.get('description', '')}")
        lines.append("")

    if analysis.get("interpretive_narrative"):
        lines.extend([
            "FULL INTERPRETATION:",
            "-" * 40,
            analysis.get("interpretive_narrative", ""),
            "",
        ])

    if analysis.get("reflection_prompts"):
        lines.append("REFLECTION PROMPTS:")
        lines.append("-" * 40)
        for prompt in analysis.get("reflection_prompts", []):
            lines.append(f"• {prompt}")
        lines.append("")

    lines.extend([
        "=" * 60,
        "Exported from Dream Ferret",
        "=" * 60,
    ])

    content = "\n".join(lines)
    filename = f"dream_{dream_id}_{title[:20].replace(' ', '_')}.txt"

    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/export/journal")
@login_required
def export_journal():
    """Export all dreams as a single text file."""
    dreams = db.get_user_dreams(current_user.id)

    if not dreams:
        flash("No dreams to export.", "error")
        return redirect(url_for("history"))

    lines = [
        "=" * 60,
        "DREAM JOURNAL EXPORT",
        f"Total Dreams: {len(dreams)}",
        f"Exported: {datetime.now().strftime('%B %d, %Y')}",
        "=" * 60,
        "",
    ]

    for i, dream in enumerate(dreams, 1):
        analysis = dream.get("analysis", {})
        title = dream.get("title", "Untitled Dream")
        timestamp = dream.get("timestamp", "")

        lines.extend([
            f"{'='*60}",
            f"DREAM #{i}: {title}",
            f"Date: {timestamp[:10] if timestamp else 'Unknown'}",
            "-" * 40,
            "",
            dream.get("dream_text", ""),
            "",
        ])

        if analysis.get("summary"):
            lines.extend([
                "Summary:",
                analysis.get("summary", ""),
                "",
            ])

        if analysis.get("detected_keywords"):
            lines.append(f"Motifs: {', '.join(analysis.get('detected_keywords', []))}")
            lines.append("")

        lines.append("")

    lines.extend([
        "=" * 60,
        "Exported from Dream Ferret",
        "=" * 60,
    ])

    content = "\n".join(lines)
    filename = f"dream_journal_{datetime.now().strftime('%Y%m%d')}.txt"

    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/search")
@login_required
def search():
    """Search through logged dreams."""
    query = request.args.get("q", "").strip()

    if not query:
        return render_template("search.html", query=None, results=[])

    dreams = db.search_user_dreams(current_user.id, query)

    # Format results for display
    formatted_results = []
    for dream in dreams:
        title = dream.get("title", "").strip()
        if not title:
            dream_text = dream.get("dream_text", "")
            first_line = dream_text.split("\n")[0][:50]
            title = first_line + "..." if len(first_line) == 50 else first_line
            if not title:
                title = "Untitled Dream"

        # Create snippet
        dream_text = dream.get("dream_text", "")
        if query.lower() in dream_text.lower():
            pos = dream_text.lower().find(query.lower())
            start = max(0, pos - 40)
            end = min(len(dream_text), pos + len(query) + 40)
            snippet = "..." + dream_text[start:end] + "..."
        else:
            snippet = dream_text[:100] + "..." if len(dream_text) > 100 else dream_text

        formatted_results.append({
            "id": dream["id"],
            "title": title,
            "timestamp": dream.get("timestamp", ""),
            "snippet": snippet,
        })

    return render_template("search.html", query=query, results=formatted_results)


# ----------------------------------------------------
# Routes - Thread Analysis
# ----------------------------------------------------

@app.route("/threads")
@login_required
def threads():
    """Display detected dream threads."""
    user_threads = db.get_user_threads(current_user.id)
    dream_count = db.get_user_dream_count(current_user.id)

    # Generate threads if none exist but user has 5+ dreams
    if not user_threads and dream_count >= 5:
        try:
            dreams = db.get_user_dreams(current_user.id)
            detected_threads, meta_analysis = thread_analyzer.analyze_dream_threads(dreams)

            # Save threads
            for thread in detected_threads:
                db.save_dream_thread(
                    user_id=current_user.id,
                    thread_name=thread.get("thread_name", "Unnamed Thread"),
                    description=thread.get("description", ""),
                    recurring_symbols=thread.get("recurring_symbols", []),
                    emotional_pattern=thread.get("emotional_pattern", ""),
                    narrative_arc=thread.get("narrative_arc", ""),
                    dream_ids=thread.get("dream_ids", [])
                )

            # Also save meta-analysis if we generated it
            if meta_analysis and not db.get_latest_meta_analysis(current_user.id):
                db.save_meta_analysis(
                    user_id=current_user.id,
                    total_dreams=dream_count,
                    top_symbols=[s["symbol"] for s in meta_analysis.get("recurring_patterns", {}).get("recurring_symbols", [])[:10]],
                    emotional_trends=meta_analysis.get("recurring_patterns", {}).get("emotional_patterns", {}),
                    narrative_patterns=meta_analysis.get("psychological_themes", []),
                    insights=meta_analysis.get("overall_theme", ""),
                    full_analysis=meta_analysis
                )

            # Refresh threads from database
            user_threads = db.get_user_threads(current_user.id)

        except Exception as e:
            print(f"Thread generation error: {e}")
            import traceback
            traceback.print_exc()
            flash("Unable to generate threads at this time.", "error")

    # Clear notification flag
    session.pop("new_threads_available", None)

    # Build dream_id -> title mapping for display
    dreams = db.get_user_dreams(current_user.id)
    dream_map = {d["id"]: d.get("title") or f"Dream #{d['id']}" for d in dreams}

    return render_template("threads.html", threads=user_threads, dream_count=dream_count, dream_map=dream_map)


@app.route("/meta-analysis")
@login_required
def meta_analysis_view():
    """Display meta-analysis of all dreams."""
    meta = db.get_latest_meta_analysis(current_user.id)
    dream_count = db.get_user_dream_count(current_user.id)

    if not meta and dream_count >= 5:
        # Generate meta-analysis if it doesn't exist
        try:
            dreams = db.get_user_dreams(current_user.id)
            threads, meta_analysis = thread_analyzer.analyze_dream_threads(dreams)

            # Save meta-analysis
            db.save_meta_analysis(
                user_id=current_user.id,
                total_dreams=dream_count,
                top_symbols=[s["symbol"] for s in meta_analysis.get("recurring_patterns", {}).get("recurring_symbols", [])[:10]],
                emotional_trends=meta_analysis.get("recurring_patterns", {}).get("emotional_patterns", {}),
                narrative_patterns=meta_analysis.get("psychological_themes", []),
                insights=meta_analysis.get("overall_theme", ""),
                full_analysis=meta_analysis
            )

            meta = db.get_latest_meta_analysis(current_user.id)

        except Exception as e:
            print(f"Meta-analysis generation error: {e}")
            flash("Unable to generate meta-analysis at this time.", "error")

    return render_template("meta_analysis.html", meta=meta, dream_count=dream_count)


@app.route("/refresh-analysis", methods=["POST"])
@login_required
def refresh_analysis():
    """Force regeneration of threads and meta-analysis."""
    dream_count = db.get_user_dream_count(current_user.id)

    if dream_count < 5:
        flash("You need at least 5 dreams to generate analysis.", "error")
        return redirect(request.referrer or url_for("index"))

    # Check cooldown (5 minutes)
    COOLDOWN_MINUTES = 5
    last_refresh = db.get_last_refresh_time(current_user.id)
    if last_refresh:
        try:
            last_refresh_dt = datetime.fromisoformat(last_refresh)
            now = datetime.utcnow()  # Compare UTC to UTC
            elapsed = (now - last_refresh_dt).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                remaining = int(COOLDOWN_MINUTES - elapsed) + 1  # Round up
                flash(f"Please wait {remaining} more minute{'s' if remaining != 1 else ''} before refreshing again.", "error")
                return redirect(request.referrer or url_for("threads"))
        except (ValueError, AttributeError):
            pass  # If parsing fails, allow the refresh

    try:
        # Clear existing cached analysis
        db.clear_user_threads(current_user.id)
        db.clear_user_meta_analysis(current_user.id)

        # Regenerate
        dreams = db.get_user_dreams(current_user.id)
        detected_threads, meta_analysis = thread_analyzer.analyze_dream_threads(dreams)

        # Save threads
        for thread in detected_threads:
            db.save_dream_thread(
                user_id=current_user.id,
                thread_name=thread.get("thread_name", "Unnamed Thread"),
                description=thread.get("description", ""),
                recurring_symbols=thread.get("recurring_symbols", []),
                emotional_pattern=thread.get("emotional_pattern", ""),
                narrative_arc=thread.get("narrative_arc", ""),
                dream_ids=thread.get("dream_ids", [])
            )

        # Save meta-analysis
        if meta_analysis:
            db.save_meta_analysis(
                user_id=current_user.id,
                total_dreams=dream_count,
                top_symbols=[s["symbol"] for s in meta_analysis.get("recurring_patterns", {}).get("recurring_symbols", [])[:10]],
                emotional_trends=meta_analysis.get("recurring_patterns", {}).get("emotional_patterns", {}),
                narrative_patterns=meta_analysis.get("psychological_themes", []),
                insights=meta_analysis.get("overall_theme", ""),
                full_analysis=meta_analysis
            )

        flash("Analysis refreshed successfully!", "success")

    except Exception as e:
        print(f"Refresh analysis error: {e}")
        flash("Unable to refresh analysis at this time.", "error")

    return redirect(request.referrer or url_for("threads"))


# ----------------------------------------------------
# Initialize database on startup
# ----------------------------------------------------

@app.before_request
def initialize_database():
    """Initialize database if it doesn't exist."""
    if not hasattr(app, "db_initialized"):
        db.init_db()
        app.db_initialized = True


@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "dream-decoder"}, 200


# ----------------------------------------------------
# Global Error Handler
# ----------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    """Log unhandled exceptions."""
    import traceback

    # Get user_id if logged in
    user_id = None
    try:
        if current_user and current_user.is_authenticated:
            user_id = current_user.id
    except:
        pass

    # Log the error
    db.log_error(
        error_type=type(e).__name__,
        message=str(e),
        traceback_str=traceback.format_exc(),
        route=request.path if request else "unknown",
        user_id=user_id
    )

    # Re-raise for Flask to handle
    raise e


if __name__ == "__main__":
    app.run(debug=True)
