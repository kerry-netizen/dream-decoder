# Dream Decoder - Planned Improvements

## High Priority (Quick Wins)

### 1. Clickable dream links in threads
- The threads page says "Appears in 4 dream(s)" but doesn't link to them
- Show dream titles and make them clickable to view full analysis

### 2. Better date formatting
- Currently shows ugly ISO format: "2026-02-01T02:06:13.720834"
- Should show friendly format: "Feb 1, 2026" or "2 hours ago"
- Apply across: history page, threads page, meta-analysis, result page

### 3. Hide empty optional fields
- "During: — | After: relieved" looks awkward
- Just don't display fields that are empty
- Applies to: history list, dream detail view

### 4. Remove "felt during/after" fields - DONE
- Decision: Remove entirely (can add back later if needed)
- Removed from: input form, database saves, display, AI payload

---

## Medium Priority (Usability)

### 5. Delete a dream
- Add ability to remove a dream from history
- Confirmation dialog required
- Should trigger thread re-analysis if dream count drops

### 6. Edit dream title
- Allow renaming a dream after submission
- Simple inline edit or modal

### 7. "Refresh Analysis" button on threads/meta page
- Let users manually trigger re-analysis
- Instead of waiting for 5th/10th dream milestone
- Useful after adding several dreams

### 8. Loading indicator
- When submitting a dream, there's a pause while AI analyzes
- Add a spinner or "Analyzing your dream..." message
- Prevents user confusion / double-submits

---

## Lower Priority (Engagement & Polish)

### 9. Dream count/streak on home page
- Show "You've logged 5 dreams"
- Optional: "3-day streak!" for consecutive days
- Encourages continued use

### 10. "Similar to past dreams" on new analysis
- After analyzing a new dream, note connections to previous dreams
- "This dream shares themes with 'The Flooded Library'"

### 11. Export to PDF/text
- Download entire dream journal
- Or export individual dream analysis

---

## Notes

- Started: Feb 1, 2026
- Last session: Fixed cross-dream analysis (threads + meta-analysis now generate on demand)
- Test account: kbryson / Sally2026!
- Deployed on Render with persistent disk at /var/data
