> **Disclaimer**
> - This repository’s code **and** this README were **AI-generated** and human-reviewed.
> - You are responsible for your own API keys, OAuth credentials, and any data the workflow processes.

# AI Email Digest — Gmail + GPT triage assistant

A daily (or weekly) email that ranks your most important new emails and summarizes only the top ones.  
Runs locally or on **GitHub Actions** so it’s laptop-agnostic.

---

## ✨ Features

- Considers all emails since the last digest (≈ last 24h by default).
- **LLM rubric judgment, not regex.** Every email is judged by an LLM against a plain-English rubric you own (`rubric.md`) and sorted into two buckets: **Action needed** (things you must do) and **Worth knowing** (things worth reading). No hand-tuned keyword classifiers.
- **OTP/2FA masking** before anything reaches the model or the rendered card — one-time codes are redacted (`‹code›`), so secrets never leave in the prompt or the digest.
- **Cross-digest de-duplication.** Each digest embeds a `[digest-ids]` marker; the next run reads your last few Sent digests and suppresses repeats. "Worth knowing" items are shown once; "Action needed" items resurface until handled (toggle with `DEDUP_SUPPRESS_ACT`). Unreadable markers **fail open** — a missed suppression, never a crash.
- **Budgeted output:** up to 7 items total, Action-needed first, then up to `min(3, 7 − #action)` Worth-knowing; a `+N more` line when action items overflow.
- **Heartbeat on quiet days:** when nothing qualifies, you still get a one-line "all clear" so a silent failure is never mistaken for a calm inbox.
- Sends a clean **HTML + plain-text** digest to your inbox.
- Provider fallback (Gemini → OpenAI) with retries to keep costs low and availability high.

---

## Quick Start (Local)

### Prerequisites
- Python **3.11+**
- A Gmail account you control
- An OpenAI API key (optional if using Gemini)

### 1) Create a virtual environment & install

**Windows (PowerShell):**
~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

**macOS/Linux (bash/zsh):**
~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

### 2) Create `.env` from the example

**Windows:**
~~~powershell
Copy-Item .env.example .env
~~~

**macOS/Linux:**
~~~bash
cp .env.example .env
~~~

Then edit `.env` and set:
- LLM_PROVIDER=gemini
- GEMINI_API_KEY=...
- GEMINI_MODEL=gemini-3.1-flash-lite
- EMAIL_USERNAME=your_gmail_address
- EMAIL_APP_PASSWORD=your_gmail_app_password

(If you prefer OpenAI, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` instead.)
> Optional (see the Configuration table for the full list):
> DEDUP_SUPPRESS_ACT=
> DEBUG=
> DISABLE_DIGEST=


### 3) Create Google OAuth credentials (Desktop)

- Google Cloud Console → **APIs & Services** → **Credentials** → **Create Credentials** → **OAuth client ID** → **Desktop app**  
- Download `credentials.json` into the repo folder (**do not commit** it).

### 4) First-run auth (generates `token.json`)
~~~bash
python main.py
~~~
A browser will request consent; on success, `token.json` is created for future runs.

> `credentials.json` and `token.json` are already in `.gitignore`.

---

## Deploy on GitHub Actions (Automated Runs)

### Add repository secrets
**Settings → Secrets and variables → Actions**:
- `OPENAI_API_KEY`
- `EMAIL_USERNAME` (your Gmail)
- `EMAIL_APP_PASSWORD` (Gmail App Password)
- `GMAIL_CREDENTIALS_JSON_B64` – base64 of `credentials.json`
- `GMAIL_TOKEN_JSON_B64` – base64 of your locally-generated `token.json`
- *(optional)* `DISABLE_DIGEST=1` to pause; `DEBUG=1` for extra logs (avoid in public)

**Base64 commands**

macOS/Linux:
~~~bash
base64 -w0 credentials.json
base64 -w0 token.json
~~~

Windows PowerShell:
~~~powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("credentials.json"))
[Convert]::ToBase64String([IO.File]::ReadAllBytes("token.json"))
~~~

### Schedule (UTC)

Edit `.github/workflows/digest.yml`. Example (08:17 SGT = 00:17 UTC):
~~~yaml
on:
  schedule:
    - cron: "17 0 * * *"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Dry run: render digest + print decisions, skip send"
        type: boolean
        default: false
~~~

### Run it

- **Actions** → **Email Digest (Daily)** → **Run workflow** (manual test).
- You should receive the digest email shortly.

### Dry run (safe preview, no email)

Tick **`dry_run`** when running the workflow manually (or set `DRY_RUN=1` locally). The pipeline reads
your inbox and calls the LLM exactly as normal, then **prints** the rendered digest plus the raw
per-item decisions to the log and **skips both the SMTP send and the watermark write**. Use it to
preview judgment quality or verify a `rubric.md` change without touching your inbox or advancing the
"since" watermark.

---

## Configuration

| Variable             | Default        | Purpose                                                                                  |
|----------------------|----------------|------------------------------------------------------------------------------------------|
| `RUBRIC_PATH`        | `rubric.md`    | Path to the judging rubric (a hard precondition — a missing/empty rubric fails the run).  |
| `DEDUP_SUPPRESS_ACT` | (blank / `0`)  | `0` = suppress only "Worth knowing" repeats (Action items resurface). `1` = suppress both.|
| `SINCE_BUFFER_HOURS` | `6`            | Overlap window subtracted from the watermark so nothing slips through a boundary.         |
| `LLM_BODY_LEN`       | `2000`         | Max masked characters of each email body sent to the judge.                              |
| `PREVIEW_LEN`        | `200`          | Max masked characters shown in each digest card preview (same masked source as the judge).|
| `LLM_MAX_OUTPUT_TOKENS` | `2000`      | Output-token cap for the judgment call.                                                  |
| `DRY_RUN`            | (blank / `0`)  | `1` = render + print decisions, skip SMTP send and watermark write (see Dry run above).   |
| `DISABLE_DIGEST`     | (blank)        | Set `1` to pause runs cleanly.                                                            |
| `DEBUG`              | (blank)        | Set `1` for verbose logs (avoid in public repos).                                        |
| `STRICT_INBOX`       | (blank / `0`)  | `1` = restrict the fetch to the Inbox only.                                              |
| `DISPLAY_TZ`         | `Asia/Singapore` | Timezone used for the display window and timestamps.                                   |

**Gemini default (free-tier friendly):** `gemini-3.1-flash-lite` (override via `GEMINI_MODEL`)

**OpenAI fallback** (with 4 retries + backoff):  
`gpt-5-mini → gpt-5-nano → gpt-4.1-mini → gpt-4o-mini → gpt-3.5-turbo`

### Tuning the digest — edit `rubric.md`, not the code

What counts as **Action needed**, **Worth knowing**, or noise is defined entirely in `rubric.md` in
plain English (which senders/newsletters matter, the transaction threshold, security-alert handling,
etc.). To change behavior, **edit `rubric.md` and commit** — no code change and no redeploy logic is
needed; the next run picks it up. Preview the effect first with a **dry run**. The digest footer also
invites quick corrections (`noise: X` / `missed: X`) that you fold back into the rubric over time.

---

## How It Works (High Level)

1. **Watermark** – Finds the last sent digest in **Sent** and uses its internal date as “since”; falls back to `last_sent_ts.txt` (local runs) and then ~24h. A `SINCE_BUFFER_HOURS` overlap guards the boundary.
2. **Pool build** – Fetches all emails since the watermark and normalizes each into a compact record (sender, subject, body).
3. **Masking** – Redacts OTP/2FA codes in the body **once**; the same masked text feeds both the judge prompt and the rendered preview, so no unmasked secret can leak by either path.
4. **Cross-digest dedup** – Reads the `[digest-ids]` marker from your last few Sent digests to build a suppression set (`DEDUP_SUPPRESS_ACT` controls whether Action items are suppressed too). Unreadable markers fail open.
5. **Judgment** – One structured LLM call scores each email against `rubric.md` as **Action needed** / **Worth knowing** / excluded (IDs must match the pool — no hallucinated items). Empty-but-valid results are honored (they do **not** trigger the fallback provider).
6. **Selection & render** – Applies the budget (7 total, Action first) and suppression set, then renders **HTML + plain text**. On a quiet day it renders a one-line heartbeat instead.
7. **Send** – Delivers via Gmail SMTP and writes the new watermark. On total provider failure it emails you an error and writes **no** digest and **no** watermark (no silent failure). `DRY_RUN=1` stops before send.

---

## Privacy & Safety

- No personal data is committed to the repo. Secrets are injected via **GitHub Actions Secrets**.  
- After making the repo public, enable **Secret scanning** and **Push protection** in **Settings → Code security**.  
- Rotate keys/tokens immediately if you suspect exposure.

---

## Troubleshooting

- **OAuth expired/revoked** → Re-run locally to regenerate `token.json` and update `GMAIL_TOKEN_JSON_B64`.  
- **Model errors/quota** → The script auto-falls back across models with retries; check the error email it sends you.  
- **Timing drift** → Cron is UTC; Actions logs print **UTC & SGT** timestamps—adjust cron accordingly.

---

## Roadmap

- Weekly digest summary & trend notes  
- Labels-aware triage (Gmail labels)  
- Links back to original Gmail threads  
- Zero-shot classifier hints from your label history (privacy-preserving)

---

## Contributing

PRs and issues welcome. Please avoid including any personal data or secrets in examples or logs.

---

## License

MIT (see `LICENSE`). If you need a different license for your fork, adjust accordingly.
