# Admin Console Plan

## Authentication
- Separate admin login at `/admin/login`
- Single admin account via environment variables:
  - `ADMIN_USER` (default: `ddadmin`)
  - `ADMIN_PASS` (default: `DreamAdmin2026!`)
- Session-based with 30 min inactivity timeout
- Brute force protection: 5 failed attempts = 15 min lockout

## Routes

### Dashboard (`/admin`)
- Total users
- Total dreams analyzed
- Dreams today / this week / this month
- New signups today / this week / this month
- Dreams per day (last 14 days)
- Signups per day (last 14 days)

### User Management (`/admin/users`)
- List all users: username, signup date, dream count, last active
- Search/filter users
- View user's dreams (read-only)
- Delete user (and all their data)

### Error Log (`/admin/errors`)
- Stored Python exceptions
- Fields: timestamp, error type, message, route, user_id
- Mark as "reviewed"
- Last 100 errors by default

### API Usage (`/admin/api-usage`)
- Track OpenAI calls: timestamp, user_id, tokens, endpoint
- Daily/weekly/monthly totals
- Estimated cost
- **10 slowest API calls** for responsiveness tracking

### System Health (`/admin/health`)
- Database size
- Disk usage on /var/data
- Uptime info

## Database Tables

```sql
-- Error logging
CREATE TABLE error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    error_type TEXT,
    message TEXT,
    traceback TEXT,
    route TEXT,
    user_id INTEGER,
    reviewed INTEGER DEFAULT 0
);

-- API usage tracking
CREATE TABLE api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id INTEGER,
    endpoint TEXT,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    model TEXT,
    duration_ms INTEGER
);

-- Admin action log
CREATE TABLE admin_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT,
    details TEXT,
    admin_user TEXT
);

-- Login attempts (for brute force protection)
CREATE TABLE login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ip_address TEXT,
    success INTEGER
);
```

## Security
1. All admin routes behind `/admin/*` prefix
2. `@admin_required` decorator
3. Rate limiting on login
4. Log all admin actions
5. Separate session from user auth

## Implementation Order
1. Database tables + admin auth
2. Dashboard with stats
3. Error logging hooks
4. User management
5. API usage tracking
6. System health
