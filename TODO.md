# Dream Ferret - Planned Improvements

## Completed

- [x] Export to text - single dream and full journal (Feb 1, 2026)
- [x] "Similar to past dreams" on new analysis (Feb 1, 2026)
- [x] Refresh cooldown (5 min) + move dream count to title row (Feb 1, 2026)
- [x] Dream count on home page (Feb 1, 2026)
- [x] Refresh Analysis button on threads/meta pages (Feb 1, 2026)
- [x] Edit dream title - inline edit on result page (Feb 1, 2026)
- [x] Delete dream functionality (Feb 1, 2026)
- [x] Loading spinner on dream submit (Feb 1, 2026)
- [x] Date formatting - friendly dates across all pages (Feb 1, 2026)
- [x] Full interpretation placement - now expands below button (Feb 1, 2026)
- [x] Clickable dream links in threads (Feb 1, 2026)
- [x] Navigation icons on result page (Feb 1, 2026)
- [x] Remove "felt during/after" fields (Feb 1, 2026)
- [x] Fix cross-dream analysis - threads/meta now generate on demand (Feb 1, 2026)
- [x] Setup persistent disk on Render (Feb 1, 2026)

---

## Anonymous "Try It" Landing (READY FOR TESTING)

**Status:** Implemented - ready for internal testing (Feb 2, 2026)

**Goal:** Allow single dream exploration without account creation to reduce friction for curious visitors.

### Design Principles
- Session-only storage - data exists only during processing, gone when session ends
- No persistence - nothing to come back to, nothing saved server-side
- Gentle prompt to create account after seeing results (with option to save that analysis)
- User can export data at any time (mention in micro-copy)

### Implementation Tasks (Priority Order)

**Phase 1: Admin Metrics Foundation (do first - needed for both features)**
1. [x] Add aggregate counters table to database (total_users, total_dreams, anon_tries)
2. [x] Create `/admin/metrics` route - protected, not linked in UI
3. [x] Display basic counts: registered users, total dreams, dreams today

**Phase 2: Anonymous Try-It Core**
4. [x] Create `/try` route - feature-flagged, NOT linked anywhere
5. [x] Build minimal dream input form (dream text only, no life context)
6. [x] Process dream via existing analysis logic, store result in Flask session only
7. [x] Create try_result.html template to display analysis
8. [x] Ensure session data is never written to database

**Phase 3: Rate Limiting & Metrics**
9. [x] Implement rate limiting - 2 minute cooldown (session-based)
10. [x] Rate limit message: "We rate limit to prevent abuse. Try again in X seconds."
11. [x] Increment anonymous try counter on each submission

**Phase 4: Account Conversion Flow (trickiest - do last)**
12. [x] Add micro-copy: "Want to track patterns over time? Create a free account."
13. [x] Add "Save this dream" button that leads to account creation
14. [x] On account creation, move session analysis to new user's saved dreams
15. [x] Clear session after successful save
16. [x] Handle edge cases: session expired, user abandons mid-flow

**Unchanged:**
- Keep existing landing page as-is - `/try` is for internal testing only

### Testing Access
- **Test URL:** `https://dream-ferret.onrender.com/try`
- Not linked from any public page
- Share manually for internal testing only
- Public landing remains unchanged until feature is validated
- Feature flag: `TRY_FEATURE_ENABLED = True` in app.py

### Explicit Non-Goals
- No session persistence across browser close
- No "come back later" functionality
- No tracking of anonymous users beyond aggregate counter

---

## Admin Observability & Beta Metrics (IMPLEMENTED)

**Status:** Implemented (Feb 2, 2026) - existing admin dashboard already covers most needs

**Goal:** Basic visibility into whether the app is being used during beta. Instrumentation, not analysis.

### Design Principles
- Know IF the app is used, not WATCH users
- No content browsing - admins cannot read dream text
- No psychological profiling
- Nothing we'd be uncomfortable explaining publicly
- Logs contain IDs, timestamps, counts only - never raw dream text

### Metrics to Track (Priority Order)
1. [x] Total registered users (count) - in admin dashboard
2. [x] Total dreams entered by logged-in users (count) - in admin dashboard
3. [x] Total anonymous "try-it" submissions (count) - added to admin dashboard
4. [x] Dreams per day (aggregate count, not per-user) - in admin dashboard
5. [ ] Repeat usage indicator - deferred (aggregate only, not "which user")

### What Is Intentionally NOT Visible
- Individual dream content
- Per-user dream counts with user identification
- Free-form "read all dreams" page
- Raw IP addresses in admin UI
- Any voyeuristic dashboard

### Implementation Tasks
1. [x] Admin dashboard already exists at `/admin`
2. [x] Protected with admin auth + session timeout
3. [x] Not linked in any public UI
4. [x] Aggregate counters derived from existing relations + new anon_try_count
5. [x] Display as text/tables with simple charts
6. [x] Logs contain only: request IDs, timestamps, counts (never dream content)

### Abuse Detection (Optional - Lower Priority)
- [ ] Consider: Is there a clean way to detect repeat abusers without storing PII?
- [ ] If IP needed: hash or truncate to /24, store only at request level
- [ ] Make this optional and defensible - can drop if no clean solution
- [ ] Note: Beta phase allows more flexibility; production will have stricter privacy

### Data Retention Policy
- Aggregate counters: kept indefinitely (no PII)
- Request logs: IDs and timestamps only, rotate after 30 days
- Anonymous try-it sessions: purged immediately on session end
- No dream content stored for analytics purposes, ever

### Future Considerations
- [ ] Role-based admin auth (currently env flag is fine for beta)
- [ ] Privacy policy updates when exiting beta
- [ ] User-facing transparency about what metrics we collect

---

## Notes

- Test account: kbryson / Sally2026!
- Deployed on Render with persistent disk at /var/data
