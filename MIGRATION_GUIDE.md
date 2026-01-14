# Migration Guide

## Migrating from Single-User to Multi-User System

If you have existing dreams in the old `dream_logs.jsonl` file, here's how to migrate them to the new multi-user system.

### Quick Migration Script

Create a file called `migrate_data.py` in the DreamDecoder directory:

```python
"""
Migrate dreams from dream_logs.jsonl to the new multi-user database.
Run this once after creating your first user account.
"""

import json
import os
from database import get_db, save_dream

# Configuration
JSONL_FILE = "dream_logs.jsonl"
TARGET_USER_ID = 1  # Change this to your user ID

def migrate():
    if not os.path.exists(JSONL_FILE):
        print("No dream_logs.jsonl file found. Nothing to migrate.")
        return

    print(f"Migrating dreams from {JSONL_FILE} to user ID {TARGET_USER_ID}...")

    count = 0
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                inp = record.get("input", {})
                analysis = record.get("analysis", {})

                save_dream(
                    user_id=TARGET_USER_ID,
                    title=inp.get("title", ""),
                    dream_text=inp.get("dream_text", ""),
                    felt_during=inp.get("felt_during", ""),
                    felt_after=inp.get("felt_after", ""),
                    life_context=inp.get("life_context", ""),
                    analysis=analysis
                )
                count += 1
                print(f"  Migrated dream {count}")

            except Exception as e:
                print(f"  Error migrating record: {e}")
                continue

    print(f"\nMigration complete! Migrated {count} dreams.")
    print(f"You can now delete or archive {JSONL_FILE}")

if __name__ == "__main__":
    migrate()
```

### Migration Steps

1. **Start the new system**:
   ```bash
   python app.py
   ```

2. **Create your user account**:
   - Navigate to http://localhost:5000
   - Click "Sign up"
   - Create your account

3. **Find your user ID** (it's probably 1 if you're the first user):
   - You can check by logging in and creating a test dream
   - Or query the database: `sqlite3 dreams.db "SELECT id, username FROM users;"`

4. **Run the migration**:
   ```bash
   python migrate_data.py
   ```

5. **Verify the migration**:
   - Log in to your account
   - Check the History page - all your old dreams should be there

6. **Archive the old file**:
   ```bash
   mv dream_logs.jsonl dream_logs_backup.jsonl
   ```

### What Gets Migrated

The migration script transfers:
- ✅ Dream title
- ✅ Dream text
- ✅ Emotional states (felt_during, felt_after)
- ✅ Life context
- ✅ Full analysis (symbols, interpretations, emotional arc, etc.)
- ✅ Timestamps (preserved from original)

### Thread Detection After Migration

After migrating, threads won't be automatically generated for your old dreams. To trigger thread detection:

1. Submit 5 new dreams (threads auto-generate at 5, 10, 15, etc.)
2. Or manually trigger by visiting `/meta-analysis` which will generate analysis on-demand

### Troubleshooting

**"No such table: users"**
- Make sure you started the app at least once so the database is initialized

**"UNIQUE constraint failed"**
- Dreams are being imported correctly, this is just a warning you can ignore

**Migration script can't find database**
- Make sure you're running it from the DreamDecoder directory
- Check that `dreams.db` exists

## Starting Fresh

If you want to start completely fresh:

1. Stop the server
2. Delete `dreams.db` (this deletes ALL user data)
3. Delete `dream_logs.jsonl` (if it exists)
4. Start the server again
5. Create a new account

## Backing Up Your Data

### Backup Database
```bash
# Create a backup
cp dreams.db dreams_backup_2026-01-13.db

# Restore from backup
cp dreams_backup_2026-01-13.db dreams.db
```

### Export Individual Dreams
You can export dreams by querying the database:

```bash
sqlite3 dreams.db "SELECT dream_text, analysis_json FROM dreams WHERE user_id = 1;" > my_dreams.txt
```

### Full Database Export
```bash
sqlite3 dreams.db .dump > dreams_backup.sql
```

## Security Note

The new system uses:
- **Password hashing**: SHA-256 with random salt
- **Session management**: Secure cookies via Flask-Login
- **User isolation**: Each user can only see their own dreams

Never share your `dreams.db` file as it contains all user passwords (though they are hashed).
