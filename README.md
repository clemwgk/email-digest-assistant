> **Disclaimer**
> - This repository’s code **and** this README were **AI-generated** with human review.
> - You are responsible for your own API keys, OAuth credentials, and any data the workflow processes.

# AI Email Digest — Gmail + GPT triage assistant

A daily (or weekly) email that ranks your most important new emails and summarizes only the top ones.  
Runs locally or on **GitHub Actions** so it’s laptop-agnostic.

## ✨ What it does
- Fetches all emails since the last digest (approx. last 24h by default).
- Drops OTP/2FA and de-duplicates threads (keeps the newest; keeps older only if deadlines appear).
- Extracts a **smart snippet** from each email (money, dates, actions) and parses amounts robustly (e.g., `SGD 25.76`).
- Scores priority (Action > FYI; Legal/Gov/Payments first; small bank alerts down-ranked unless risk).
- Chooses **Top 5** (or fewer if fewer exist), with **Type, Category, Urgency, Why, Action**.
- Emails you a clean **HTML + plain-text** digest.

## 🧭 Why this project
Portfolio-friendly demo of practical LLM automation:
- OAuth to Gmail (read-only)
- Sensible triage heuristics + LLM ranking
- Cost-aware design and model fallback
- CI-style automation via GitHub Actions

---

## Quick Start (Local)

**Prereqs**
- Python **3.11+**
- A Gmail account you control
- An OpenAI API key

**1) Clone & install**
```bash
pip install -r requirements.txt
