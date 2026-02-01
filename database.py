"""
Database management for Dream Decoder multi-user system.
Uses SQLite for storing users, dreams, and thread analysis.
"""

import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from encryption import encrypt, decrypt


# Use persistent disk on Render if available, otherwise local directory
if os.path.exists("/var/data"):
    DB_PATH = "/var/data/dreams.db"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "dreams.db")


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)

    # Dreams table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT,
            dream_text TEXT NOT NULL,
            felt_during TEXT,
            felt_after TEXT,
            life_context TEXT,
            timestamp TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Dream threads table (cross-dream patterns)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dream_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            thread_name TEXT NOT NULL,
            description TEXT,
            recurring_symbols TEXT,
            emotional_pattern TEXT,
            narrative_arc TEXT,
            dream_ids TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Meta-analysis table (overall patterns for a user)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            analysis_date TEXT NOT NULL,
            total_dreams INTEGER NOT NULL,
            top_symbols TEXT,
            emotional_trends TEXT,
            narrative_patterns TEXT,
            insights TEXT,
            analysis_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Admin: Error logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            error_type TEXT,
            message TEXT,
            traceback TEXT,
            route TEXT,
            user_id INTEGER,
            reviewed INTEGER DEFAULT 0
        )
    """)

    # Admin: API usage tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER,
            endpoint TEXT,
            tokens_prompt INTEGER DEFAULT 0,
            tokens_completion INTEGER DEFAULT 0,
            model TEXT,
            duration_ms INTEGER DEFAULT 0
        )
    """)

    # Admin: Action log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT,
            details TEXT,
            admin_user TEXT
        )
    """)

    # Admin: Login attempts (brute force protection)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ip_address TEXT,
            username TEXT,
            success INTEGER DEFAULT 0
        )
    """)

    # Site settings (beta notes, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS site_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# ----------------------------------------------------
# User Management
# ----------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a hash."""
    try:
        salt, pwd_hash = password_hash.split("$")
        test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return test_hash == pwd_hash
    except ValueError:
        return False


def create_user(username: str, password: str) -> Optional[int]:
    """Create a new user. Returns user ID or None if username exists."""
    conn = get_db()
    cursor = conn.cursor()

    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user and return user dict or None."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user and verify_password(password, user["password_hash"]):
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), user["id"])
        )
        conn.commit()
        conn.close()

        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
        }

    conn.close()
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
        }
    return None


# ----------------------------------------------------
# Dream Management
# ----------------------------------------------------

def save_dream(
    user_id: int,
    title: str,
    dream_text: str,
    felt_during: str,
    felt_after: str,
    life_context: str,
    analysis: Dict[str, Any]
) -> int:
    """Save a dream and its analysis. Returns dream ID.

    User-entered content (dream_text, life_context) is encrypted at rest.
    Title and analysis are stored unencrypted for search/display.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Encrypt user-entered content
    encrypted_dream_text = encrypt(dream_text)
    encrypted_life_context = encrypt(life_context) if life_context else life_context

    cursor.execute(
        """INSERT INTO dreams
        (user_id, title, dream_text, felt_during, felt_after, life_context, timestamp, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            title,
            encrypted_dream_text,
            felt_during,
            felt_after,
            encrypted_life_context,
            datetime.utcnow().isoformat(),
            json.dumps(analysis, ensure_ascii=False)
        )
    )

    conn.commit()
    dream_id = cursor.lastrowid
    conn.close()

    return dream_id


def get_user_dreams(user_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all dreams for a user, ordered by most recent first.

    Decrypts user-entered content (dream_text, life_context) on read.
    """
    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM dreams WHERE user_id = ? ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, (user_id,))
    dreams = cursor.fetchall()
    conn.close()

    result = []
    for dream in dreams:
        result.append({
            "id": dream["id"],
            "user_id": dream["user_id"],
            "title": dream["title"],
            "dream_text": decrypt(dream["dream_text"]),
            "felt_during": dream["felt_during"],
            "felt_after": dream["felt_after"],
            "life_context": decrypt(dream["life_context"]) if dream["life_context"] else dream["life_context"],
            "timestamp": dream["timestamp"],
            "analysis": json.loads(dream["analysis_json"]),
        })

    return result


def get_dream_by_id(dream_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific dream. Verifies user owns the dream.

    Decrypts user-entered content on read.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM dreams WHERE id = ? AND user_id = ?",
        (dream_id, user_id)
    )
    dream = cursor.fetchone()
    conn.close()

    if dream:
        return {
            "id": dream["id"],
            "user_id": dream["user_id"],
            "title": dream["title"],
            "dream_text": decrypt(dream["dream_text"]),
            "felt_during": dream["felt_during"],
            "felt_after": dream["felt_after"],
            "life_context": decrypt(dream["life_context"]) if dream["life_context"] else dream["life_context"],
            "timestamp": dream["timestamp"],
            "analysis": json.loads(dream["analysis_json"]),
        }

    return None


def search_user_dreams(user_id: int, query: str) -> List[Dict[str, Any]]:
    """Search through user's dreams.

    Searches title and analysis_json in SQL, then decrypts and searches
    dream_text and life_context in Python for encrypted content.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get all user dreams - we need to decrypt to search encrypted content
    cursor.execute(
        "SELECT * FROM dreams WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    )

    dreams = cursor.fetchall()
    conn.close()

    query_lower = query.lower()
    result = []

    for dream in dreams:
        # Decrypt content for searching
        dream_text = decrypt(dream["dream_text"])
        life_context = decrypt(dream["life_context"]) if dream["life_context"] else ""

        # Check if query matches any searchable field
        matches = (
            (dream["title"] and query_lower in dream["title"].lower()) or
            (dream_text and query_lower in dream_text.lower()) or
            (life_context and query_lower in life_context.lower()) or
            (query_lower in dream["analysis_json"].lower())
        )

        if matches:
            result.append({
                "id": dream["id"],
                "user_id": dream["user_id"],
                "title": dream["title"],
                "dream_text": dream_text,
                "felt_during": dream["felt_during"],
                "felt_after": dream["felt_after"],
                "life_context": life_context,
                "timestamp": dream["timestamp"],
                "analysis": json.loads(dream["analysis_json"]),
            })

    return result


def get_user_dream_count(user_id: int) -> int:
    """Get total number of dreams for a user."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM dreams WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result["count"] if result else 0


def delete_dream(dream_id: int, user_id: int) -> bool:
    """Delete a dream. Returns True if deleted, False if not found or not owned."""
    conn = get_db()
    cursor = conn.cursor()

    # Verify ownership and delete
    cursor.execute(
        "DELETE FROM dreams WHERE id = ? AND user_id = ?",
        (dream_id, user_id)
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return deleted


def update_dream_title(dream_id: int, user_id: int, new_title: str) -> bool:
    """Update a dream's title. Returns True if updated, False if not found or not owned."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE dreams SET title = ? WHERE id = ? AND user_id = ?",
        (new_title, dream_id, user_id)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return updated


def clear_user_threads(user_id: int) -> None:
    """Clear all threads for a user (for regeneration after dream deletion)."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM dream_threads WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def clear_user_meta_analysis(user_id: int) -> None:
    """Clear meta-analysis for a user (for regeneration after dream deletion)."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM meta_analysis WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ----------------------------------------------------
# Thread Management
# ----------------------------------------------------

def save_dream_thread(
    user_id: int,
    thread_name: str,
    description: str,
    recurring_symbols: List[str],
    emotional_pattern: str,
    narrative_arc: str,
    dream_ids: List[int]
) -> int:
    """Save a detected dream thread. Returns thread ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO dream_threads
        (user_id, thread_name, description, recurring_symbols, emotional_pattern, narrative_arc, dream_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            thread_name,
            description,
            json.dumps(recurring_symbols),
            emotional_pattern,
            narrative_arc,
            json.dumps(dream_ids),
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    thread_id = cursor.lastrowid
    conn.close()

    return thread_id


def get_user_threads(user_id: int) -> List[Dict[str, Any]]:
    """Get all dream threads for a user."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM dream_threads WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    threads = cursor.fetchall()
    conn.close()

    result = []
    for thread in threads:
        result.append({
            "id": thread["id"],
            "user_id": thread["user_id"],
            "thread_name": thread["thread_name"],
            "description": thread["description"],
            "recurring_symbols": json.loads(thread["recurring_symbols"]),
            "emotional_pattern": thread["emotional_pattern"],
            "narrative_arc": thread["narrative_arc"],
            "dream_ids": json.loads(thread["dream_ids"]),
            "created_at": thread["created_at"],
        })

    return result


# ----------------------------------------------------
# Meta-Analysis Management
# ----------------------------------------------------

def save_meta_analysis(
    user_id: int,
    total_dreams: int,
    top_symbols: List[str],
    emotional_trends: Dict[str, Any],
    narrative_patterns: List[str],
    insights: str,
    full_analysis: Dict[str, Any]
) -> int:
    """Save a meta-analysis. Returns analysis ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO meta_analysis
        (user_id, analysis_date, total_dreams, top_symbols, emotional_trends, narrative_patterns, insights, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            datetime.utcnow().isoformat(),
            total_dreams,
            json.dumps(top_symbols),
            json.dumps(emotional_trends),
            json.dumps(narrative_patterns),
            insights,
            json.dumps(full_analysis, ensure_ascii=False)
        )
    )

    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()

    return analysis_id


def get_latest_meta_analysis(user_id: int) -> Optional[Dict[str, Any]]:
    """Get the most recent meta-analysis for a user."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM meta_analysis WHERE user_id = ? ORDER BY analysis_date DESC LIMIT 1",
        (user_id,)
    )
    analysis = cursor.fetchone()
    conn.close()

    if analysis:
        return {
            "id": analysis["id"],
            "user_id": analysis["user_id"],
            "analysis_date": analysis["analysis_date"],
            "total_dreams": analysis["total_dreams"],
            "top_symbols": json.loads(analysis["top_symbols"]),
            "emotional_trends": json.loads(analysis["emotional_trends"]),
            "narrative_patterns": json.loads(analysis["narrative_patterns"]),
            "insights": analysis["insights"],
            "full_analysis": json.loads(analysis["analysis_json"]),
        }

    return None


def get_last_refresh_time(user_id: int) -> Optional[str]:
    """Get the last time analysis was refreshed for a user."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT analysis_date FROM meta_analysis WHERE user_id = ? ORDER BY analysis_date DESC LIMIT 1",
        (user_id,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return result["analysis_date"]
    return None


def find_similar_dreams(user_id: int, new_symbols: List[str], exclude_dream_id: int = None, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Find past dreams with similar symbols/motifs.
    Returns list of {id, title, shared_symbols, match_count}.
    """
    if not new_symbols:
        return []

    new_symbols_lower = set(s.lower() for s in new_symbols)
    dreams = get_user_dreams(user_id)
    similar = []

    for dream in dreams:
        if exclude_dream_id and dream["id"] == exclude_dream_id:
            continue

        analysis = dream.get("analysis", {})
        if not analysis:
            continue

        # Get symbols from past dream
        past_symbols = []
        past_symbols.extend(analysis.get("detected_keywords", []))
        for sym in analysis.get("key_symbols", []):
            if isinstance(sym, dict) and "symbol" in sym:
                past_symbols.append(sym["symbol"])
            elif isinstance(sym, str):
                past_symbols.append(sym)

        past_symbols_lower = set(s.lower() for s in past_symbols)

        # Find overlap
        shared = new_symbols_lower & past_symbols_lower
        if shared:
            title = dream.get("title", "").strip()
            if not title:
                text = dream.get("dream_text", "")
                title = text[:40] + "..." if len(text) > 40 else text
                if not title:
                    title = "Untitled Dream"

            similar.append({
                "id": dream["id"],
                "title": title,
                "shared_symbols": list(shared),
                "match_count": len(shared)
            })

    # Sort by match count descending, take top N
    similar.sort(key=lambda x: x["match_count"], reverse=True)
    return similar[:limit]


# ----------------------------------------------------
# Admin Functions
# ----------------------------------------------------

def log_error(error_type: str, message: str, traceback_str: str, route: str, user_id: int = None) -> None:
    """Log an error to the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO error_logs (timestamp, error_type, message, traceback, route, user_id) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), error_type, message, traceback_str, route, user_id)
    )
    conn.commit()
    conn.close()


def get_error_logs(limit: int = 100, include_reviewed: bool = False) -> List[Dict[str, Any]]:
    """Get recent error logs."""
    conn = get_db()
    cursor = conn.cursor()
    if include_reviewed:
        cursor.execute("SELECT * FROM error_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    else:
        cursor.execute("SELECT * FROM error_logs WHERE reviewed = 0 ORDER BY timestamp DESC LIMIT ?", (limit,))
    errors = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return errors


def mark_error_reviewed(error_id: int) -> None:
    """Mark an error as reviewed."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE error_logs SET reviewed = 1 WHERE id = ?", (error_id,))
    conn.commit()
    conn.close()


def mark_all_errors_reviewed() -> int:
    """Mark all unreviewed errors as reviewed. Returns count of updated rows."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE error_logs SET reviewed = 1 WHERE reviewed = 0")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def log_api_usage(user_id: int, endpoint: str, tokens_prompt: int, tokens_completion: int, model: str, duration_ms: int) -> None:
    """Log an API call."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO api_usage (timestamp, user_id, endpoint, tokens_prompt, tokens_completion, model, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), user_id, endpoint, tokens_prompt, tokens_completion, model, duration_ms)
    )
    conn.commit()
    conn.close()


def get_api_usage_stats() -> Dict[str, Any]:
    """Get API usage statistics."""
    conn = get_db()
    cursor = conn.cursor()

    # Total calls and tokens
    cursor.execute("SELECT COUNT(*) as calls, SUM(tokens_prompt) as prompt, SUM(tokens_completion) as completion FROM api_usage")
    totals = dict(cursor.fetchone())

    # Today
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT COUNT(*) as calls, SUM(tokens_prompt) as prompt, SUM(tokens_completion) as completion FROM api_usage WHERE timestamp LIKE ?",
        (f"{today}%",)
    )
    today_stats = dict(cursor.fetchone())

    # This week (last 7 days)
    cursor.execute(
        "SELECT COUNT(*) as calls, SUM(tokens_prompt) as prompt, SUM(tokens_completion) as completion FROM api_usage WHERE timestamp >= date('now', '-7 days')"
    )
    week_stats = dict(cursor.fetchone())

    # Slowest 10 calls
    cursor.execute("SELECT * FROM api_usage ORDER BY duration_ms DESC LIMIT 10")
    slowest = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "totals": totals,
        "today": today_stats,
        "week": week_stats,
        "slowest": slowest
    }


def log_login_attempt(ip_address: str, username: str, success: bool) -> None:
    """Log a login attempt."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO login_attempts (timestamp, ip_address, username, success) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), ip_address, username, 1 if success else 0)
    )
    conn.commit()
    conn.close()


def get_failed_login_count(ip_address: str, minutes: int = 15) -> int:
    """Get count of failed login attempts from an IP in the last N minutes."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT COUNT(*) FROM login_attempts
           WHERE ip_address = ? AND success = 0
           AND timestamp >= datetime('now', ?)""",
        (ip_address, f"-{minutes} minutes")
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def log_admin_action(action: str, details: str, admin_user: str) -> None:
    """Log an admin action."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO admin_logs (timestamp, action, details, admin_user) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), action, details, admin_user)
    )
    conn.commit()
    conn.close()


def get_admin_stats() -> Dict[str, Any]:
    """Get admin dashboard statistics."""
    conn = get_db()
    cursor = conn.cursor()

    # Total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Total dreams
    cursor.execute("SELECT COUNT(*) FROM dreams")
    total_dreams = cursor.fetchone()[0]

    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Dreams today
    cursor.execute("SELECT COUNT(*) FROM dreams WHERE timestamp LIKE ?", (f"{today}%",))
    dreams_today = cursor.fetchone()[0]

    # Dreams this week
    cursor.execute("SELECT COUNT(*) FROM dreams WHERE timestamp >= date('now', '-7 days')")
    dreams_week = cursor.fetchone()[0]

    # Dreams this month
    cursor.execute("SELECT COUNT(*) FROM dreams WHERE timestamp >= date('now', '-30 days')")
    dreams_month = cursor.fetchone()[0]

    # Signups today
    cursor.execute("SELECT COUNT(*) FROM users WHERE created_at LIKE ?", (f"{today}%",))
    signups_today = cursor.fetchone()[0]

    # Signups this week
    cursor.execute("SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-7 days')")
    signups_week = cursor.fetchone()[0]

    # Signups this month
    cursor.execute("SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-30 days')")
    signups_month = cursor.fetchone()[0]

    # Dreams per day (last 14 days)
    cursor.execute("""
        SELECT date(timestamp) as day, COUNT(*) as count
        FROM dreams
        WHERE timestamp >= date('now', '-14 days')
        GROUP BY date(timestamp)
        ORDER BY day
    """)
    dreams_by_day = [dict(row) for row in cursor.fetchall()]

    # Signups per day (last 14 days)
    cursor.execute("""
        SELECT date(created_at) as day, COUNT(*) as count
        FROM users
        WHERE created_at >= date('now', '-14 days')
        GROUP BY date(created_at)
        ORDER BY day
    """)
    signups_by_day = [dict(row) for row in cursor.fetchall()]

    # Unreviewed errors count
    cursor.execute("SELECT COUNT(*) FROM error_logs WHERE reviewed = 0")
    unreviewed_errors = cursor.fetchone()[0]

    conn.close()

    return {
        "total_users": total_users,
        "total_dreams": total_dreams,
        "dreams_today": dreams_today,
        "dreams_week": dreams_week,
        "dreams_month": dreams_month,
        "signups_today": signups_today,
        "signups_week": signups_week,
        "signups_month": signups_month,
        "dreams_by_day": dreams_by_day,
        "signups_by_day": signups_by_day,
        "unreviewed_errors": unreviewed_errors
    }


def get_all_users() -> List[Dict[str, Any]]:
    """Get all users with dream counts."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.username, u.created_at, u.last_login,
               (SELECT COUNT(*) FROM dreams WHERE user_id = u.id) as dream_count
        FROM users u
        ORDER BY u.created_at DESC
    """)
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def delete_user_and_data(user_id: int) -> bool:
    """Delete a user and all their data."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dreams WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM dream_threads WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM meta_analysis WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
    except Exception:
        deleted = False
    conn.close()
    return deleted


# ----------------------------------------------------
# Site Settings
# ----------------------------------------------------

def get_setting(key: str) -> Optional[str]:
    """Get a site setting value."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM site_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    """Set a site setting value."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO site_settings (key, value, updated_at)
        VALUES (?, ?, ?)""",
        (key, value, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_beta_notes() -> Optional[str]:
    """Get beta notes."""
    return get_setting("beta_notes")


def set_beta_notes(notes: str) -> None:
    """Set beta notes."""
    set_setting("beta_notes", notes)


# ----------------------------------------------------
# Encryption Migration
# ----------------------------------------------------

def migrate_encrypt_dreams() -> int:
    """
    Migrate existing unencrypted dreams to encrypted format.
    Only encrypts dreams that aren't already encrypted.
    Returns count of migrated dreams.
    """
    from encryption import encrypt, is_encrypted

    conn = get_db()
    cursor = conn.cursor()

    # Get all dreams
    cursor.execute("SELECT id, dream_text, life_context FROM dreams")
    dreams = cursor.fetchall()

    migrated = 0
    for dream in dreams:
        dream_id = dream["id"]
        dream_text = dream["dream_text"]
        life_context = dream["life_context"]

        needs_update = False
        new_dream_text = dream_text
        new_life_context = life_context

        # Check if dream_text needs encryption
        if dream_text and not is_encrypted(dream_text):
            new_dream_text = encrypt(dream_text)
            needs_update = True

        # Check if life_context needs encryption
        if life_context and not is_encrypted(life_context):
            new_life_context = encrypt(life_context)
            needs_update = True

        if needs_update:
            cursor.execute(
                "UPDATE dreams SET dream_text = ?, life_context = ? WHERE id = ?",
                (new_dream_text, new_life_context, dream_id)
            )
            migrated += 1

    conn.commit()
    conn.close()

    return migrated
