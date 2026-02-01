# Dream Decoder - Planned Improvements

## Next Up

### 1. Date formatting + hide empty fields (combined)
- Change "2026-02-01T02:06:13.720834" to "Feb 1, 2026"
- Apply to: history page, threads page, search results
- Also hide any empty optional fields while we're in the templates

### 2. Full interpretation placement fix
- Move the expanded "full interpretation" content to display directly below the button
- Currently it appears after key symbols, which is confusing

### 3. Clickable dream links in threads
- "Appears in 4 dream(s)" should show dream titles as clickable links
- Requires passing dream data to threads template

### 4. Add icon menu to result page
- Result page is missing the top navigation icons (history, threads, meta, logout)
- Should match the header on other pages

### 5. Loading indicator on dream submit
- Add spinner or "Analyzing your dream..." when form submits
- Prevents confusion during AI processing delay

---

## Medium Priority

### 5. Delete a dream
- Add delete button to dream detail view
- Confirmation dialog required
- May need to regenerate threads after deletion

### 6. Edit dream title
- Allow renaming after submission
- Simple inline edit or modal

### 7. "Refresh Analysis" button on threads/meta page
- Manual trigger for re-analysis instead of waiting for milestones

---

## Lower Priority (Polish)

### 8. Dream count on home page
- Show "You've logged 6 dreams" on input form
- Encourages continued use

### 9. "Similar to past dreams" on new analysis
- Note connections to previous dreams after analysis

### 10. Export to PDF/text
- Download dream journal or individual analyses

---

## Completed

- [x] Remove "felt during/after" fields (Feb 1, 2026)
- [x] Fix cross-dream analysis - threads/meta now generate on demand (Feb 1, 2026)
- [x] Setup persistent disk on Render (Feb 1, 2026)

---

## Notes

- Test account: kbryson / Sally2026!
- Deployed on Render with persistent disk at /var/data
