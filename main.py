import os
import datetime
import base64
import re
import html
import smtplib
import traceback
import time
from typing import List, Tuple

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError

# =========================================================
# Kill switch (pause runs without errors)
# =========================================================
if os.getenv("DISABLE_DIGEST", "").strip() == "1":
    print("Digest disabled via DISABLE_DIGEST=1")
    raise SystemExit(0)

# =========================================================
# Env & OpenAI client
# =========================================================
load_dotenv()  # no-op in Actions unless you create .env; useful locally
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model fallback + retries
MODEL_CANDIDATES = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]
MAX_RETRIES = 2
BASE_BACKOFF = 2.0  # seconds

# Pool and snippet tuning
MAX_SINGLE_PASS = 60      # single-pass if pool <= this; else rare fallback
SNIPPET_LEN     = 220     # characters per email snippet
DEADLINE_WORDS  = ["invoice", "bill", "payment", "due", "pay by", "action required", "iras", "mom", "hdb", "bank"]

# Gmail scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# =========================================================
# Gmail auth
# =========================================================
def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                raise RuntimeError(
                    "Gmail refresh token invalid/expired. "
                    "Re-auth locally to regenerate token.json, then update GMAIL_TOKEN_JSON in GitHub Secrets."
                ) from e
        else:
            # Local-only OAuth flow (won't run in Actions)
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# =========================================================
# Email sending
# =========================================================
def send_digest_email(subject, body_text):
    sender = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_APP_PASSWORD")
    if not sender or not password:
        raise RuntimeError("Missing EMAIL_USERNAME or EMAIL_APP_PASSWORD in environment.")

    recipient = sender  # send to self
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Plain
    msg.attach(MIMEText(body_text, "plain"))

    # Simple HTML wrapper that preserves Markdown-ish formatting
    html_content = f"""
    <html><body>
      <pre style="font-family: monospace; font-size: 14px; white-space: pre-wrap;">{body_text}</pre>
    </body></html>
    """
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

def send_error_email(subject, error_text):
    try:
        send_digest_email(subject, f"Email digest failed.\n\nDetails:\n{error_text}")
    except Exception as e:
        print("Failed to send error email:", e)

# =========================================================
# Body extraction helpers
# =========================================================
TAG_RE = re.compile(r"<[^>]+>")

def _clean_html_to_text(html_str: str) -> str:
    return TAG_RE.sub("", html.unescape(html_str))

def _extract_body_snippet(msg_data, max_len: int) -> str:
    """
    Extract a text snippet from Gmail message payload:
    - Try top-level body
    - Walk nested parts, prefer text/plain then text/html (stripped)
    """
    try:
        payload = msg_data.get("payload", {})

        # 1) Direct body (top-level)
        body_data = payload.get("body", {}).get("data")
        if body_data:
            text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
            return text[:max_len].strip().replace("\n", " ")

        # 2) Walk nested parts (stack DFS)
        stack = [payload]
        while stack:
            part = stack.pop()
            if not isinstance(part, dict):
                continue
            mime = part.get("mimeType", "")
            part_body = part.get("body", {}).get("data")
            parts = part.get("parts", [])
            if parts:
                stack.extend(parts)

            if part_body and mime == "text/plain":
                text = base64.urlsafe_b64decode(part_body).decode("utf-8", errors="ignore")
                return text[:max_len].strip().replace("\n", " ")
            if part_body and mime == "text/html":
                html_txt = base64.urlsafe_b64decode(part_body).decode("utf-8", errors="ignore")
                text = _clean_html_to_text(html_txt)
                return text[:max_len].strip().replace("\n", " ")
    except Exception:
        pass
    return "[Could not extract body]"

# =========================================================
# Watermark = last sent digest time
# =========================================================
def get_last_run_timestamp(service) -> int:
    """
    Look in Sent for the most recent digest (subject starts with 'AI Email Digest —')
    and use its internalDate (ms) as the watermark. If none found, fallback to now-24h.
    """
    query = 'in:sent subject:"AI Email Digest —"'
    now = datetime.datetime.now(datetime.UTC)
    fallback = int((now - datetime.timedelta(days=1)).timestamp())
    try:
        res = service.users().messages().list(userId='me', q=query, maxResults=1).execute()
        msgs = res.get('messages', [])
        if not msgs:
            return fallback
        msg = service.users().messages().get(userId='me', id=msgs[0]['id'], format='metadata').execute()
        return int(int(msg.get('internalDate', '0')) / 1000) or fallback
    except Exception:
        return fallback

# =========================================================
# Fetch all emails since watermark (OTP drop + thread dedupe)
# =========================================================
def fetch_all_since(service, since_unix: int, snippet_len: int = SNIPPET_LEN):
    """
    Build the full pool since the last run.
    - Drop obvious OTP/verification emails.
    - Keep the latest per thread; keep older items only if they contain deadline words.
    """
    query = f'after:{since_unix}'
    pool = []
    page_token = None
    latest_by_thread = {}

    while True:
        res = service.users().messages().list(
            userId='me', q=query, maxResults=100, pageToken=page_token
        ).execute()
        msgs = res.get('messages', [])
        page_token = res.get('nextPageToken')

        for m in msgs:
            msg = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            headers = msg['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'N/A')
            ts = int(msg.get('internalDate', '0')) // 1000
            thread_id = msg.get('threadId')
            snippet = _extract_body_snippet(msg, max_len=snippet_len)

            combined = (subject + ' ' + snippet).lower()
            if any(k in combined for k in ['otp', 'verification code', '2fa', 'login code']):
                continue

            from_domain = sender
            if '@' in sender:
                try:
                    from_domain = sender.split('@')[-1].split('>')[0].strip().lower()
                except Exception:
                    pass

            item = {
                'id': m['id'],
                'threadId': thread_id,
                'timestamp': ts,
                'from_domain': from_domain,
                'subject': subject,
                'snippet': snippet,
                'has_deadline_word': any(w in combined for w in DEADLINE_WORDS),
            }

            prev = latest_by_thread.get(thread_id)
            if not prev or ts >= prev['timestamp']:
                latest_by_thread[thread_id] = item
            elif item['has_deadline_word']:
                pool.append(item)

        if not page_token:
            break

    # Include the latest message per thread
    pool.extend(latest_by_thread.values())
    # Newest first for readability (model still re-ranks)
    pool.sort(key=lambda e: e['timestamp'], reverse=True)
    return pool

# =========================================================
# GPT fallback + prompts
# =========================================================
def call_gpt_with_fallback(messages: List[dict]) -> Tuple[str, str]:
    """
    Try each model with light retry/backoff. Return (content, model_used).
    """
    errors = []
    for m in MODEL_CANDIDATES:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=0.3
                )
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content, m
            except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
                errors.append((f"{m} attempt {attempt}", f"{e.__class__.__name__}: {e}"))
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF * attempt)  # 2s, then 4s
                    continue
                break
            except Exception as e:
                errors.append((m, f"{e.__class__.__name__}: {e}"))
                break
    combined = "\n".join([f"[{model}] {err}" for model, err in errors])
    raise RuntimeError(f"All GPT model calls failed.\n{combined}")

def build_single_pass_prompt(emails):
    base = (
        "You are a personal email triage assistant.\n\n"
        "Context:\n"
        "- These are ALL emails since the last digest run.\n"
        "- Always return exactly **5 emails**, ranked most→least important.\n"
        "- If none are critical, still choose the best 5.\n\n"
        "Ranking rules (strict):\n"
        "1) Action > Notification.\n"
        "2) Legal/Government/Billing/Payment matters first—esp. deadlines or required responses.\n"
        "3) Bills/payment due always matter.\n"
        "4) Social media last; include only if nothing better exists.\n"
        "5) Avoid duplicates (similar thread/topic).\n\n"
        "For each selected email provide:\n"
        "- **Type**: \"Call to Action\" OR \"For Information Only\"\n"
        "- **Category**: Legal | Government | Billing/Payment | Other\n"
        "- **Why (1–2 lines)**\n\n"
        "Output format (Markdown):\n"
        "# Today’s Top 5 Emails\n"
        "1. **<Subject>** — _<From domain>_\n"
        "   - **Type:** ...\n"
        "   - **Category:** ...\n"
        "   - **Why:** ...\n\n"
        "Now, here are the emails:\n"
    )
    lines = []
    for i, e in enumerate(emails, 1):
        lines.append(
            f"{i}. From domain: {e['from_domain']}\n"
            f"   Subject: {e['subject']}\n"
            f"   Snippet: {e['snippet']}\n"
            f"   Time: {datetime.datetime.fromtimestamp(e['timestamp']).isoformat()}\n"
        )
    return base + "\n".join(lines)

# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    try:
        service = authenticate_gmail()

        # Determine window
        since_ts = get_last_run_timestamp(service)
        pool = fetch_all_since(service, since_ts, snippet_len=SNIPPET_LEN)

        from datetime import datetime as _dt
        window_start = _dt.fromtimestamp(since_ts).strftime('%Y-%m-%d %H:%M')
        window_end = _dt.now().strftime('%Y-%m-%d %H:%M')

        # If nothing to consider, send a short note
        if not pool:
            subject = f"AI Email Digest — {_dt.now().strftime('%Y-%m-%d')} (empty)"
            body = (
                f"# AI Email Digest ({_dt.now().strftime('%Y-%m-%d')})\n"
                f"Considered: 0 emails (from {window_start} to {window_end}).\n\n"
                f"No emails met the criteria."
            )
            send_digest_email(subject, body)
            print("Digest sent (empty).")
            raise SystemExit(0)

        # Single-pass for your normal volume; rare fallback if pool is unusually large
        if len(pool) <= MAX_SINGLE_PASS:
            prompt = build_single_pass_prompt(pool)
            messages = [{"role": "user", "content": prompt}]
            summary, model_used = call_gpt_with_fallback(messages)
        else:
            # Simple safety valve: chunk and ask for top candidates, then final top-5.
            # You likely won't hit this, but it keeps costs sane on spike days.
            batch_size = 30
            per_batch_pick = 8
            picks = []
            for i in range(0, len(pool), batch_size):
                chunk = pool[i:i+batch_size]
                brief_prompt = (
                    "Select the TOP {k} emails from the list below that most warrant attention "
                    "(CTA and critical matters first). Return ONLY a JSON array of the selected indexes."
                ).format(k=per_batch_pick)
                numbered = "\n".join([f"{j+1}. {e['subject']} — {e['from_domain']}" for j, e in enumerate(chunk)])
                messages = [{"role": "user", "content": brief_prompt + "\n\n" + numbered}]
                content, _ = call_gpt_with_fallback(messages)
                import re as _re, json as _json
                idxs = []
                try:
                    m = _re.search(r'\[.*\]', content, _re.S)
                    if m:
                        idxs = _json.loads(m.group(0))
                except Exception:
                    idxs = list(range(1, min(per_batch_pick, len(chunk)) + 1))
                for idx in idxs:
                    if isinstance(idx, int) and 1 <= idx <= len(chunk):
                        picks.append(chunk[idx-1])
            # Dedup by subject+domain
            dedup = {(p['subject'], p['from_domain']): p for p in picks}
            final_pool = list(dedup.values())
            prompt = build_single_pass_prompt(final_pool)
            messages = [{"role": "user", "content": prompt}]
            summary, model_used = call_gpt_with_fallback(messages)

        # Compose + send
        header = (
            f"# AI Email Digest ({_dt.now().strftime('%Y-%m-%d')})\n"
            f"_Model: {model_used}_\n"
            f"Considered: {len(pool)} emails (from {window_start} to {window_end}).\n\n"
        )
        body = header + (summary or "No content returned.")
        subject = f"AI Email Digest — {_dt.now().strftime('%Y-%m-%d')} (model: {model_used}; considered {len(pool)})"

        send_digest_email(subject, body)
        print("Digest sent.")

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
