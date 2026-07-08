# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Earlier history predates this file — see the git log.

## [Unreleased] — Rubric-judge redesign (branch `redesign/rubric-judge`)

A ground-up replacement of the regex/heuristic triage with LLM rubric judgment.

### Added
- **`rubric.md`** — a plain-English rubric that fully defines Action / Worth-knowing / noise. Behavior is
  tuned by editing this file (no code change); it is a hard precondition — a missing/empty rubric fails
  the run loudly (error email, no digest).
- **Cross-digest de-duplication** — each digest embeds a `[digest-ids]` marker (plain-text line + HTML
  comment mirror); the next run reads the last few Sent digests and suppresses repeats. Unreadable
  markers **fail open**. `DEDUP_SUPPRESS_ACT` toggles whether Action items are suppressed too (default:
  only Worth-knowing).
- **OTP/2FA masking** narrowed to digit-bearing 4–8 char tokens in a keyword window; the same masked
  text feeds both the judge prompt and the rendered preview (single source — no unmasked leak path).
- **Heartbeat** one-liner on quiet days (empty judgment) so silence is never ambiguous.
- **`dry_run` workflow input / `DRY_RUN` env** — renders the digest and prints per-item decisions to the
  log, skipping SMTP send and watermark write.
- **`tests/`** — pytest suite (mocked Gmail + LLM, plus an API-gated live golden-set test) covering
  masking, parsing, dedup round-trip + fail-open, rendering, provider fallback, and hygiene.

### Changed
- Output is now two sections — **Action needed** / **Worth knowing** — with a budget of 7 items total
  (Action first), replacing the flat "Top 5 with Type/Category/Urgency" list.
- Provider contract clarified: an empty-but-valid judgment is a success and does **not** trigger the
  OpenAI fallback (only genuine exhaustion does).

### Removed
- All regex classifiers and heuristic scoring: snippet extraction, transaction/amount scoring,
  social/promo/booking detectors, and the `SNIPPET_LEN` / `TXN_ALERT_MIN` knobs.
