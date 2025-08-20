> **Disclaimer**
> - This repository’s code **and** this README were **AI-generated** and human-reviewed.
> - You are responsible for your own API keys, OAuth credentials, and any data the workflow processes.

# AI Email Digest — Gmail + GPT triage assistant

A daily (or weekly) email that ranks your most important new emails and summarizes only the top ones.  
Runs locally or on **GitHub Actions** so it’s laptop-agnostic.

---

## ✨ Features

- Considers all emails since the last digest (≈ last 24h by default).
- Drops OTP/2FA noise; de-duplicates threads (keeps newest; retains older only if they include deadlines).
- Smart snippet extraction (money, dates, actions) with **decimal fix** (`25.\n76 → 25.76`) and robust amount parsing (e.g., `SGD 25.76` or `25.76 SGD`).
- Ranking rules: **Action > FYI**, **Legal/Gov/Billing first**, small card alerts down-ranked unless risk cues (declined/fraud).
- Always returns the **Top 5** (or fewer if fewer exist), with **Type, Category, Urgency, Why, Action**.
- Sends a clean **HTML + plain-text** digest to your inbox.
- Model fallback with retries to keep costs low and availability high.

---

## Quick Start (Local)

### Prerequisites
- Python **3.11+**
- A Gmail account you control
- An OpenAI API key

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
- OPENAI_API_KEY=...
- EMAIL_USERNAME=your_gmail_address
- EMAIL_APP_PASSWORD=your_gmail_app_password
> Optional:
> SNIPPET_LEN=350
> TXN_ALERT_MIN=100
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
  workflow_dispatch: {}
~~~

### Run it

- **Actions** → **Email Digest (Daily)** → **Run workflow** (manual test).  
- You should receive the digest email shortly.

---

## Configuration

| Variable           | Default | Purpose                                                                              |
|-------------------|---------|--------------------------------------------------------------------------------------|
| `SNIPPET_LEN`     | 350     | Max characters per email snippet                                                     |
| `TXN_ALERT_MIN`   | 100     | Card alerts below this amount are down-ranked unless risk terms appear               |
| `DISABLE_DIGEST`  | (blank) | Set `1` to pause runs cleanly                                                        |
| `DEBUG`           | (blank) | Set `1` for verbose logs (avoid in public repos)                                    |

**Model fallback** (with 4 retries + backoff):  
`gpt-5-mini → gpt-5-nano → gpt-4.1-mini → gpt-4o-mini → gpt-3.5-turbo`

---

## How It Works (High Level)

1. **Watermark** – Finds the last sent digest in **Sent** and uses its internal date as “since”; falls back to ~24h.  
2. **Pool build** – Fetches all emails since watermark; drops OTPs; thread de-dupes with deadline exceptions.  
3. **Parsing & snippet** – Reads full body, fixes split decimals, parses amounts with context, builds a windowed snippet around money/date/action cues.  
4. **Selection** – Prompts an LLM to return structured JSON with `Type/Category/Urgency/Why/Action` for the top K (no hallucinations—IDs must match).  
5. **Render & send** – HTML + plain text via Gmail SMTP.

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
