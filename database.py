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
    """Save a dream and its analysis. Returns dream ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO dreams
        (user_id, title, dream_text, felt_during, felt_after, life_context, timestamp, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            title,
            dream_text,
            felt_during,
            felt_after,
            life_context,
            datetime.utcnow().isoformat(),
            json.dumps(analysis, ensure_ascii=False)
        )
    )

    conn.commit()
    dream_id = cursor.lastrowid
    conn.close()

    return dream_id


def get_user_dreams(user_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all dreams for a user, ordered by most recent first."""
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
            "dream_text": dream["dream_text"],
            "felt_during": dream["felt_during"],
            "felt_after": dream["felt_after"],
            "life_context": dream["life_context"],
            "timestamp": dream["timestamp"],
            "analysis": json.loads(dream["analysis_json"]),
        })

    return result


def get_dream_by_id(dream_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific dream. Verifies user owns the dream."""
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
            "dream_text": dream["dream_text"],
            "felt_during": dream["felt_during"],
            "felt_after": dream["felt_after"],
            "life_context": dream["life_context"],
            "timestamp": dream["timestamp"],
            "analysis": json.loads(dream["analysis_json"]),
        }

    return None


def search_user_dreams(user_id: int, query: str) -> List[Dict[str, Any]]:
    """Search through user's dreams."""
    conn = get_db()
    cursor = conn.cursor()

    # SQLite full-text search on title, dream_text, life_context, and analysis
    cursor.execute(
        """SELECT * FROM dreams
        WHERE user_id = ?
        AND (
            title LIKE ? OR
            dream_text LIKE ? OR
            life_context LIKE ? OR
            analysis_json LIKE ?
        )
        ORDER BY timestamp DESC""",
        (user_id, f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%")
    )

    dreams = cursor.fetchall()
    conn.close()

    result = []
    for dream in dreams:
        result.append({
            "id": dream["id"],
            "user_id": dream["user_id"],
            "title": dream["title"],
            "dream_text": dream["dream_text"],
            "felt_during": dream["felt_during"],
            "felt_after": dream["felt_after"],
            "life_context": dream["life_context"],
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
