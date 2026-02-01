"""
Admin routes for Dream Decoder.
Separate authentication from regular users.
"""

import os
from functools import wraps
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, session

import database as db

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Admin credentials from environment (with defaults for development)
ADMIN_USER = os.environ.get("ADMIN_USER", "ddadmin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "DreamAdmin2026!")

# Brute force protection settings
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


def admin_required(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.login"))
        # Check session timeout (30 min)
        last_activity = session.get("admin_last_activity")
        if last_activity:
            elapsed = (datetime.utcnow() - datetime.fromisoformat(last_activity)).total_seconds()
            if elapsed > 1800:  # 30 minutes
                session.pop("admin_logged_in", None)
                session.pop("admin_last_activity", None)
                flash("Admin session expired. Please log in again.", "error")
                return redirect(url_for("admin.login"))
        session["admin_last_activity"] = datetime.utcnow().isoformat()
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    """Admin login page."""
    if session.get("admin_logged_in"):
        return redirect(url_for("admin.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        ip_address = request.remote_addr

        # Check for lockout
        failed_attempts = db.get_failed_login_count(ip_address, LOCKOUT_MINUTES)
        if failed_attempts >= MAX_LOGIN_ATTEMPTS:
            flash(f"Too many failed attempts. Please wait {LOCKOUT_MINUTES} minutes.", "error")
            return render_template("admin/login.html")

        # Verify credentials
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin_logged_in"] = True
            session["admin_user"] = username
            session["admin_last_activity"] = datetime.utcnow().isoformat()
            db.log_login_attempt(ip_address, username, True)
            db.log_admin_action("login", f"Admin logged in from {ip_address}", username)
            return redirect(url_for("admin.dashboard"))
        else:
            db.log_login_attempt(ip_address, username, False)
            remaining = MAX_LOGIN_ATTEMPTS - failed_attempts - 1
            if remaining > 0:
                flash(f"Invalid credentials. {remaining} attempts remaining.", "error")
            else:
                flash(f"Too many failed attempts. Please wait {LOCKOUT_MINUTES} minutes.", "error")

    return render_template("admin/login.html")


@admin_bp.route("/logout")
def logout():
    """Admin logout."""
    admin_user = session.get("admin_user", "unknown")
    db.log_admin_action("logout", "Admin logged out", admin_user)
    session.pop("admin_logged_in", None)
    session.pop("admin_user", None)
    session.pop("admin_last_activity", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("admin.login"))


@admin_bp.route("/")
@admin_required
def dashboard():
    """Admin dashboard with stats."""
    stats = db.get_admin_stats()
    return render_template("admin/dashboard.html", stats=stats)


@admin_bp.route("/users")
@admin_required
def users():
    """User management page."""
    all_users = db.get_all_users()
    search = request.args.get("search", "").strip().lower()
    if search:
        all_users = [u for u in all_users if search in u["username"].lower()]
    return render_template("admin/users.html", users=all_users, search=search)


@admin_bp.route("/users/<int:user_id>")
@admin_required
def user_detail(user_id):
    """View a user's dreams."""
    users = db.get_all_users()
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("admin.users"))

    dreams = db.get_user_dreams(user_id)
    return render_template("admin/user_detail.html", user=user, dreams=dreams)


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id):
    """Delete a user and all their data."""
    users = db.get_all_users()
    user = next((u for u in users if u["id"] == user_id), None)

    if not user:
        flash("User not found.", "error")
        return redirect(url_for("admin.users"))

    admin_user = session.get("admin_user", "unknown")
    db.log_admin_action("delete_user", f"Deleted user {user['username']} (ID: {user_id})", admin_user)

    if db.delete_user_and_data(user_id):
        flash(f"User '{user['username']}' and all their data deleted.", "success")
    else:
        flash("Failed to delete user.", "error")

    return redirect(url_for("admin.users"))


@admin_bp.route("/errors")
@admin_required
def errors():
    """Error log page."""
    include_reviewed = request.args.get("reviewed", "0") == "1"
    error_logs = db.get_error_logs(limit=100, include_reviewed=include_reviewed)
    return render_template("admin/errors.html", errors=error_logs, include_reviewed=include_reviewed)


@admin_bp.route("/errors/<int:error_id>/review", methods=["POST"])
@admin_required
def review_error(error_id):
    """Mark an error as reviewed."""
    db.mark_error_reviewed(error_id)
    admin_user = session.get("admin_user", "unknown")
    db.log_admin_action("review_error", f"Marked error {error_id} as reviewed", admin_user)
    flash("Error marked as reviewed.", "success")
    return redirect(url_for("admin.errors"))


@admin_bp.route("/errors/review-all", methods=["POST"])
@admin_required
def review_all_errors():
    """Mark all unreviewed errors as reviewed."""
    count = db.mark_all_errors_reviewed()
    admin_user = session.get("admin_user", "unknown")
    db.log_admin_action("review_all_errors", f"Marked {count} errors as reviewed", admin_user)
    flash(f"Marked {count} error(s) as reviewed.", "success")
    return redirect(url_for("admin.errors"))


@admin_bp.route("/api-usage")
@admin_required
def api_usage():
    """API usage statistics page."""
    stats = db.get_api_usage_stats()
    return render_template("admin/api_usage.html", stats=stats)


@admin_bp.route("/beta-notes", methods=["GET", "POST"])
@admin_required
def beta_notes():
    """Edit beta notes displayed on disclaimer page."""
    if request.method == "POST":
        notes = request.form.get("beta_notes", "").strip()
        db.set_beta_notes(notes)
        admin_user = session.get("admin_user", "unknown")
        db.log_admin_action("update_beta_notes", "Updated beta notes", admin_user)
        flash("Beta notes updated.", "success")
        return redirect(url_for("admin.beta_notes"))

    current_notes = db.get_beta_notes() or ""
    return render_template("admin/beta_notes.html", beta_notes=current_notes)


@admin_bp.route("/health")
@admin_required
def health():
    """System health page."""
    health_info = {}

    # Database size
    db_path = db.DB_PATH
    if os.path.exists(db_path):
        health_info["db_size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    else:
        health_info["db_size_mb"] = 0

    # Disk usage (if on Render with /var/data)
    if os.path.exists("/var/data"):
        import shutil
        total, used, free = shutil.disk_usage("/var/data")
        health_info["disk_total_gb"] = round(total / (1024**3), 2)
        health_info["disk_used_gb"] = round(used / (1024**3), 2)
        health_info["disk_free_gb"] = round(free / (1024**3), 2)
        health_info["disk_percent"] = round((used / total) * 100, 1)
    else:
        health_info["disk_total_gb"] = None

    health_info["db_path"] = db_path

    return render_template("admin/health.html", health=health_info)


@admin_bp.route("/migrate-encryption", methods=["POST"])
@admin_required
def migrate_encryption():
    """Migrate existing dreams to encrypted format."""
    try:
        count = db.migrate_encrypt_dreams()
        admin_user = session.get("admin_user", "unknown")
        db.log_admin_action("migrate_encryption", f"Encrypted {count} dreams", admin_user)
        flash(f"Successfully encrypted {count} dreams.", "success")
    except Exception as e:
        flash(f"Migration failed: {str(e)}", "error")

    return redirect(url_for("admin.health"))
