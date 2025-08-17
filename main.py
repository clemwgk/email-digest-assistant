import os
import sys
import datetime
import base64
import re
import html
import json
import smtplib
import traceback
import time
from typing import List, Tuple, Optional

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
load_dotenv()  # no-op in Actions unless .env exists; useful locally
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model fallback order — preserved from your repo, with retries bumped to 4
MODEL_CANDIDATES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]
MAX_RETRIES = 4           # ⬅️ increased from 2 → 4 (your request)
BASE_BACKOFF = 2.0        # seconds (2s, 4s, 6s, 8s)

# Tunables (preserved defaults)
MAX_SINGLE_PASS = 60
SNIPPET_LEN = int(os.getenv("SNIPPET_LEN", "350"))
TXN_ALERT_MIN = float(os.getenv("TXN_ALERT_MIN", "100"))

# Deadline cues (preserved from your published version)
DEADLINE_WORDS = ["invoice", "bill", "payment", "due", "pay by", "action required"]

# Gmail scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Debug toggle
DEBUG = os.getenv("DEBUG", "").strip() == "1"
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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
                    "Re-auth locally to regenerate token.json, then update GMAIL_TOKEN_JSON_B64 in GitHub Secrets."
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
def send_digest_email(subject: str, body_text: str, html_body: str | None = None):
    sender = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_APP_PASSWORD")
    if not sender or not password:
        raise RuntimeError("Missing EMAIL_USERNAME or EMAIL_APP_PASSWORD in environment.")

    recipient = sender  # send to self
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Plain text
    msg.attach(MIMEText(body_text, "plain"))

    # HTML version
    if html_body is None:
        html_body = f"""
        <html><body>
          <pre style="font-family: monospace; font-size: 14px; white-space: pre-wrap;">{html.escape(body_text)}</pre>
        </body></html>
        """
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

def send_error_email(subject, error_text):
    try:
        html_err = f"<html><body><pre style='font-family:monospace;white-space:pre-wrap'>{html.escape(error_text)}</pre></body></html>"
        send_digest_email(subject, f"Email digest failed.\n\nDetails:\n{error_text}", html_err)
    except Exception as e:
        print("Failed to send error email:", e)

# =========================================================
# Body extraction + decimal-fix
# =========================================================
TAG_RE = re.compile(r"<[^>]+>")
DECIMAL_JOIN_RE = re.compile(r'(\d+)\.\s+(\d{2})(\b)')  # join "25. \n76" → "25.76"

def _fix_split_decimals(text: str) -> str:
    return DECIMAL_JOIN_RE.sub(r'\1.\2\3', text)

def _clean_html_to_text(html_str: str) -> str:
    return TAG_RE.sub("", html.unescape(html_str))

def _extract_full_body_text(msg_data) -> str:
    """Extract full body text from Gmail message payload; prefer text/plain but strip HTML; fix split decimals."""
    try:
        payload = msg_data.get("payload", {})

        # 1) Direct body (top-level)
        body_data = payload.get("body", {}).get("data")
        if body_data:
            text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
            return _fix_split_decimals(text.strip())

        # 2) Walk nested parts (stack DFS)
        stack = [payload]
        full_texts = []
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
                full_texts.append(text.strip())
            elif part_body and mime == "text/html":
                html_txt = base64.urlsafe_b64decode(part_body).decode("utf-8", errors="ignore")
                full_texts.append(_clean_html_to_text(html_txt).strip())

        return _fix_split_decimals("\n".join(full_texts))
    except Exception:
        return ""

# =========================================================
# Authoritative amount parsing (currency-first and amount-first)
# =========================================================
CUR_AMOUNT_RE = re.compile(
    r'(?ix)\b(?:SGD|S\$|\$|USD|US\$|AUD|EUR|GBP)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?)'
)
AMT_CURRENCY_RE = re.compile(
    r'(?ix)\b([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?)\s*(?:SGD|S\$|\$|USD|US\$|AUD|EUR|GBP)\b'
)

TXN_POSITIVE_WORDS = [
    'transaction alert','charged','spent','purchase','payment made','transfer',
    'debit','withdrawal','pos','merchant','card ending','authorized','unauthorized',
    'declined','failed'
]
TXN_NEGATIVE_CONTEXT = [
    # statements/advice/non-action contexts
    'statement','estatement','advice ready','available for viewing','monthly statement',
    'account summary','balance summary','available credit','credit limit',
    'interest rate','fx rate','exchange rate'
]

def _nearest_score_to_keywords(text: str, span: tuple[int, int]) -> int:
    """Simple score: presence of txn keywords within ~120 chars window around span."""
    s, e = span
    win_s = max(0, s - 120)
    win_e = min(len(text), e + 120)
    window = text[win_s:win_e].lower()
    score = 0
    for kw in TXN_POSITIVE_WORDS:
        if kw in window:
            score += 2
    for bad in TXN_NEGATIVE_CONTEXT:
        if bad in window:
            score -= 2
    return score

def _parse_amount_from_text(cleaned_text: str) -> Optional[tuple[str, float, tuple[int,int], str]]:
    """
    Return (currency, amount, span, fmt) or None.
    Prefers matches near txn-positive words and away from negative contexts.
    Skips percentages like '5.76%'.
    Defaults '$' to SGD for your use-case unless 'USD' explicitly present.
    """
    candidates = []

    # currency-first
    for m in CUR_AMOUNT_RE.finditer(cleaned_text):
        end = m.end()
        # Guard against immediate percent sign
        if end < len(cleaned_text) and cleaned_text[end:end+1] == '%':
            continue
        raw_num = m.group(1)
        num = float(raw_num.replace(',', ''))
        raw_span = (m.start(), m.end())
        # Currency heuristic
        cur = "SGD"
        prefix = cleaned_text[m.start():m.end()].upper()
        if "USD" in prefix or "US$" in prefix:
            cur = "USD"
        elif "SGD" in prefix or "S$" in prefix or "$" in prefix:
            cur = "SGD"  # default assumption for your inbox
        score = _nearest_score_to_keywords(cleaned_text, raw_span)
        candidates.append((cur, num, raw_span, "cur-first", score))

    # amount-first
    for m in AMT_CURRENCY_RE.finditer(cleaned_text):
        end = m.end()
        if end < len(cleaned_text) and cleaned_text[end:end+1] == '%':
            continue
        raw_num = m.group(1)
        num = float(raw_num.replace(',', ''))
        raw_span = (m.start(), m.end())
        suffix = cleaned_text[m.start():m.end()].upper()
        cur = "SGD"
        if "USD" in suffix or "US$" in suffix:
            cur = "USD"
        elif "SGD" in suffix or "S$" in suffix or "$" in suffix:
            cur = "SGD"
        score = _nearest_score_to_keywords(cleaned_text, raw_span)
        candidates.append((cur, num, raw_span, "amt-first", score))

    if not candidates:
        return None

    # Choose best by score, then by larger amount (fraud/importance bias)
    candidates.sort(key=lambda t: (t[4], t[1]), reverse=True)
    cur, num, span, fmt, _ = candidates[0]
    return (cur, num, span, fmt)

# =========================================================
# Smart snippet (span windows; amount-centered; decimal-safe)
# =========================================================
MONEY_TOKEN_RE = re.compile(r'(?i)(?:sgd|s\$|\$|usd|us\$|aud|eur|gbp)\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?')
DATE_RE  = re.compile(r'(?i)\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b')
ACTION_RE = re.compile(r'(?i)\b(due|deadline|expire|action required|respond|pay|submit|urgent|overdue|payment due|bill)\b')

def _merge_spans(spans: list[tuple[int,int]]) -> list[tuple[int,int]]:
    if not spans: return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s > merged[-1][1] + 10:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [tuple(x) for x in merged]

def _extract_smart_snippet_v2(full_text: str, max_len: int, preferred_span: Optional[tuple[int,int]] = None) -> str:
    """
    Build snippet windows around amounts/dates/actions; if preferred_span provided (authoritative amount),
    center snippet around it. Decimal-friendly; avoids sentence splitting issues.
    """
    if not full_text:
        return "[Could not extract body]"
    text = _fix_split_decimals(full_text)
    text = re.sub(r'\s+', ' ', text).strip()

    def spans_for(regex, pre=80, post=120):
        outs = []
        for m in regex.finditer(text):
            s = max(0, m.start() - pre)
            e = min(len(text), m.end() + post)
            outs.append((s, e))
        return outs

    spans: list[tuple[int,int]] = []

    if preferred_span:
        s, e = preferred_span
        s = max(0, s - 120); e = min(len(text), e + 140)
        spans.append((s, e))

    spans += spans_for(MONEY_TOKEN_RE, pre=80, post=120)
    spans += spans_for(ACTION_RE, pre=80, post=120)
    spans += spans_for(DATE_RE,  pre=60, post=100)

    if not spans:
        return text[:max_len]

    merged = _merge_spans(spans)

    # Stitch windows until max_len
    pieces, total = [], 0
    for s, e in merged:
        chunk = text[s:e]
        if total + len(chunk) + 5 > max_len:
            pieces.append(chunk[: max_len - total])
            break
        pieces.append(chunk)
        total += len(chunk) + 5
    snippet = ' … '.join(pieces)[:max_len]
    return snippet.strip()

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
# Transaction detection (precise; authoritative amount + low-amount tag)
# =========================================================
def enhanced_txn_detection_v2(subject: str, full_text: str,
                              parsed_currency: Optional[str],
                              parsed_amount: Optional[float]):
    """
    Decide if this is a real transaction alert vs statement/info.
    Returns: (is_txn_alert, currency, amount, txn_small_flag)
    """
    text = f"{subject} {full_text}".lower()

    # Exclude statements/advice and non-action contexts
    if any(ind in text for ind in TXN_NEGATIVE_CONTEXT):
        return (False, None, None, False)

    # Positive indicators
    is_txn = any(ind in text for ind in TXN_POSITIVE_WORDS)
    if not is_txn and parsed_amount is None:
        return (False, None, None, False)

    cur = parsed_currency
    amt = parsed_amount
    small = False
    if amt is not None:
        if amt < TXN_ALERT_MIN:
            small = True
        if cur is None:
            cur = "SGD"  # default assumption for your inbox

    return (True, cur, amt, small)

# =========================================================
# Fetch all emails since watermark (OTP drop + thread dedupe) — v2 pipeline
# =========================================================
def fetch_all_since_v2(service, since_unix: int, snippet_len: int = SNIPPET_LEN):
    """
    Build the full pool since the last run.
    Pipeline:
      full body → parse authoritative amount → snippet around amount/action/date → txn detection
    Plus:
      - Drop OTP/2FA emails
      - Keep latest per thread; keep older items only if they contain deadline words
    """
    query = f'after:{since_unix}'
    pool = []
    page_token = None
    latest_by_thread: dict[str, dict] = {}

    while True:
        res = service.users().messages().list(
            userId='me', q=query, maxResults=100, pageToken=page_token
        ).execute()
        msgs = res.get('messages', [])
        page_token = res.get('nextPageToken')

        for m in msgs:
            msg = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'N/A')
            ts = int(msg.get('internalDate', '0')) // 1000
            thread_id = msg.get('threadId')

            full_body = _extract_full_body_text(msg)
            combined_lower = (subject + ' ' + full_body).lower()

            # OTP / verification drop
            if any(k in combined_lower for k in ['otp', 'verification code', '2fa', 'login code']):
                continue

            # Authoritative amount parse
            parsed = _parse_amount_from_text(full_body)
            parsed_cur = parsed[0] if parsed else None
            parsed_amt = parsed[1] if parsed else None
            amount_span = parsed[2] if parsed else None

            # Snippet around amount/action/date (decimal-safe)
            snippet = _extract_smart_snippet_v2(full_body, max_len=snippet_len, preferred_span=amount_span)

            # From domain (best-effort)
            from_domain = sender
            if '@' in sender:
                try:
                    from_domain = sender.split('@')[-1].split('>')[0].strip().lower()
                except Exception:
                    pass

            # Transaction detection using authoritative amount
            is_txn, cur, amt, small = enhanced_txn_detection_v2(subject, full_body, parsed_cur, parsed_amt)

            item = {
                'id': m['id'],
                'threadId': thread_id,
                'timestamp': ts,
                'from_domain': from_domain,
                'subject': subject,
                'snippet': snippet,
                'has_deadline_word': any(w in combined_lower for w in DEADLINE_WORDS),
                'txn_alert': is_txn,
                'txn_currency': cur,
                'txn_amount': amt,
                'txn_small': bool(small),
            }

            # DEBUG observability for txn emails
            if DEBUG and is_txn:
                debug_print(f"[TXN] subj={subject!r} amount={amt} {cur} small={small} snippet={snippet[:120]!r}")

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
# GPT: fallback + enhanced JSON selection
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
                    time.sleep(BASE_BACKOFF * attempt)  # 2s, 4s, 6s, 8s
                    continue
                break
            except Exception as e:
                errors.append((m, f"{e.__class__.__name__}: {e}"))
                break
    combined = "\n".join([f"[{model}] {err}" for model, err in errors])
    raise RuntimeError(f"All GPT model calls failed.\n{combined}")

def build_enhanced_selection_prompt_json(emails):
    """
    Enhanced JSON-only selection with authoritative txn info and richer output (why/action/urgency).
    """
    K = min(5, len(emails))
    header = (
        "You are a personal email triage assistant.\n\n"
        "INPUT:\n"
        f"- EMAIL_COUNT: {len(emails)}\n"
        f"- RETURN_COUNT: {K}\n"
        "- Each item includes: id, from_domain, subject, snippet, time, "
        "  txn_alert (bool), txn_amount (number|null), txn_currency (string|null), txn_small (bool).\n\n"
        "TASK:\n"
        f"Select the top {K} emails that most warrant attention.\n\n"
        "Ranking rules (strict):\n"
        "1) Action > Notification.\n"
        "2) Legal/Government/Billing/Payment first—especially deadlines or required responses.\n"
        "3) Bills/payment due always matter.\n"
        "4) Social media last; include only if nothing better exists.\n"
        "5) Avoid duplicates (same thread/topic).\n"
        "6) **Bank/financial notifications**: treat as lower priority **unless** any of the following:\n"
        f"   - amount ≥ {TXN_ALERT_MIN} (assume SGD if currency missing)\n"
        "   - wording suggests risk: declined, failed, suspicious, unauthorized, dispute, fraud\n"
        "   - unusual merchant hints (use judgment)\n\n"
        "Authoritative data rules:\n"
        "- If `txn_alert` is true and `txn_amount` is present, treat `txn_amount` as authoritative; do NOT infer other amounts from free text.\n"
        "- If `txn_small` is true, rank lower unless there are no higher-priority items or risk words are present.\n\n"
        "For each selection, provide:\n"
        "- **why**: 2–3 sentences: (a) what it is, (b) why it matters, (c) what to do\n"
        "- **action**: specific action if any (e.g., 'Pay by Aug 20', 'Review and respond', 'No action needed')\n"
        "- **urgency**: High/Medium/Low based on deadlines and consequences\n\n"
        "RULES (CRITICAL):\n"
        "- Select only from provided `id` values. DO NOT invent emails.\n"
        "- If EMAIL_COUNT < 5, return exactly EMAIL_COUNT items.\n"
        "- Output ONLY valid JSON with this schema and nothing else:\n"
        '{\n'
        '  "picks": [\n'
        '    {"id": "<id>", "type": "Call to Action" | "For Information Only", '
        '     "category": "Legal" | "Government" | "Billing/Payment" | "Other", '
        '     "why": "<2-3 sentences>", "action": "<specific action or No action needed>", '
        '     "urgency": "High" | "Medium" | "Low"}\n'
        '  ]\n'
        '}\n\n'
        "EMAILS:\n"
    )
    lines = []
    for e in emails:
        lines.append(json.dumps({
            "id": e["id"],
            "from_domain": e["from_domain"],
            "subject": e["subject"],
            "snippet": e["snippet"],
            "time": datetime.datetime.fromtimestamp(e["timestamp"]).isoformat(),
            "txn_alert": bool(e.get("txn_alert")),
            "txn_amount": e.get("txn_amount"),
            "txn_currency": e.get("txn_currency"),
            "txn_small": bool(e.get("txn_small", False)),
        }, ensure_ascii=False))
    return header + "\n".join(lines)

def parse_enhanced_json_picks(raw: str, valid_ids: set, want_count: int):
    """
    Parse the enhanced model's JSON with action and urgency fields.
    """
    picks = []
    try:
        m = re.search(r'\{.*\}', raw, re.S)
        obj = json.loads(m.group(0) if m else raw)
        for item in obj.get("picks", []):
            _id = item.get("id")
            _type = item.get("type")
            _cat = item.get("category")
            _why = item.get("why")
            _action = item.get("action")
            _urgency = item.get("urgency")
            if (_id in valid_ids and 
                _type in ("Call to Action", "For Information Only") and 
                _cat in ("Legal", "Government", "Billing/Payment", "Other") and 
                isinstance(_why, str) and isinstance(_action, str) and 
                _urgency in ("High", "Medium", "Low")):
                picks.append({
                    "id": _id, "type": _type, "category": _cat,
                    "why": _why.strip(), "action": _action.strip(), "urgency": _urgency
                })
            if len(picks) >= want_count:
                break
    except Exception:
        picks = []
    return picks

# =========================================================
# Rendering
# =========================================================
def render_text_digest(picks, emails_by_id, model_used, considered_count, window_start, window_end):
    from datetime import datetime as _dt
    header = (
        f"# AI Email Digest ({_dt.now().strftime('%Y-%m-%d')})\n"
        f"_Model: {model_used}_\n"
        f"Considered: {considered_count} emails (from {window_start} to {window_end}).\n\n"
        "# Today's Top " + str(len(picks)) + " Emails\n"
    )
    lines = []
    for i, p in enumerate(picks, 1):
        e = emails_by_id[p["id"]]
        lines.append(
            f"{i}. **{e['subject']}** — _{e['from_domain']}_\n"
            f"   - **Type:** {p['type']}\n"
            f"   - **Category:** {p['category']}\n"
            f"   - **Urgency:** {p['urgency']}\n"
            f"   - **Why:** {p['why']}\n"
            f"   - **Action:** {p['action']}\n"
            f"   - **Snippet:** {e['snippet']}\n"
        )
    return header + ("\n".join(lines) if lines else "_No eligible emails._")

def _badge(label: str, bg: str, color: str = "#111"):
    return f"<span style='display:inline-block;padding:2px 6px;border-radius:12px;background:{bg};color:{color};font-size:12px;font-weight:600'>{html.escape(label)}</span>"

def render_html_digest(picks, emails_by_id, model_used, considered_count, window_start, window_end):
    items_html = []
    for i, p in enumerate(picks, 1):
        e = emails_by_id[p["id"]]
        subj = html.escape(e["subject"]); dom = html.escape(e["from_domain"])
        snip = html.escape(e["snippet"]); why = html.escape(p["why"]); action = html.escape(p["action"])
        type_bg = "#fde68a" if p["type"] == "Call to Action" else "#dbeafe"
        cat_bg_map = {"Legal": "#fecaca", "Government": "#fde68a", "Billing/Payment": "#bbf7d0", "Other": "#e5e7eb"}
        cat_bg = cat_bg_map.get(p["category"], "#e5e7eb")
        urgency_bg_map = {"High": "#fecaca", "Medium": "#fde68a", "Low": "#d1fae5"}
        urgency_bg = urgency_bg_map.get(p["urgency"], "#e5e7eb")
        txn_chip = ""
        if e.get("txn_alert"):
            amt = e.get("txn_amount"); cur = e.get("txn_currency") or "SGD"
            if amt is not None:
                over = amt >= TXN_ALERT_MIN
                txn_chip = _badge(f"Txn {cur} {amt:.2f}", "#bbf7d0" if over else "#e5e7eb")
            else:
                txn_chip = _badge("Txn alert", "#e5e7eb")
        items_html.append(f"""
          <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:10px 0;background:#ffffff">
            <div style="font-weight:700;font-size:15px;line-height:1.3;margin-bottom:4px">{i}. {subj}</div>
            <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{dom} {txn_chip}</div>
            <div style="margin-bottom:8px">
              {_badge(p["type"], type_bg)}
              <span style="display:inline-block;width:6px"></span>
              {_badge(p["category"], cat_bg)}
              <span style="display:inline-block;width:6px"></span>
              {_badge(f"Urgency: {p['urgency']}", urgency_bg)}
            </div>
            <div style="font-size:13px;margin-bottom:6px"><strong>Why:</strong> {why}</div>
            <div style="font-size:13px;margin-bottom:6px"><strong>Action:</strong> {action}</div>
            <div style="font-size:12px;color:#6b7280"><strong>Snippet:</strong> {snip}</div>
          </div>
        """)
    items_html = "\n".join(items_html) if items_html else "<p><em>No eligible emails.</em></p>"

    title = f"AI Email Digest ({datetime.datetime.now().strftime('%Y-%m-%d')})"
    header_html = f"""
      <div style="font-family:Inter,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:16px 12px;background:#f8fafc">
        <h2 style="margin:0 0 4px 0;font-size:20px">{html.escape(title)}</h2>
        <div style="color:#6b7280;font-size:12px;margin-bottom:12px">
          Model: <strong>{html.escape(model_used)}</strong> · Considered: <strong>{considered_count}</strong> emails · Window: {html.escape(window_start)} → {html.escape(window_end)}
        </div>
        {items_html}
        <div style="color:#9ca3af;font-size:11px;margin-top:12px">— Generated by your Email Digest Assistant</div>
      </div>
    """
    return f"<html><body>{header_html}</body></html>"

# =========================================================
# Selection wrapper
# =========================================================
def select_top_k_via_enhanced_json(pool, window_start, window_end):
    """Enhanced JSON-based selection with deterministic fallback."""
    want = min(5, len(pool))
    prompt = build_enhanced_selection_prompt_json(pool)
    messages = [{"role": "user", "content": prompt}]
    raw, model_used = call_gpt_with_fallback(messages)

    valid_ids = {e["id"] for e in pool}
    picks = parse_enhanced_json_picks(raw, valid_ids, want)

    if len(picks) < want:
        missing = want - len(picks)
        chosen = {p["id"] for p in picks}
        fallback_items = [e for e in pool if e["id"] not in chosen][:missing]
        picks += [{
            "id": e["id"], "type": "For Information Only", "category": "Other",
            "why": "Selected by fallback (newest) due to invalid/empty model output. Please review manually.",
            "action": "Review manually", "urgency": "Low"
        } for e in fallback_items]
    return picks, model_used

# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    try:
        service = authenticate_gmail()

        # Determine window
        since_ts = get_last_run_timestamp(service)

        # Use v2 pipeline (full body → amount → snippet → detection)
        pool = fetch_all_since_v2(service, since_unix=since_ts, snippet_len=SNIPPET_LEN)

        from datetime import datetime as _dt
        window_start = _dt.fromtimestamp(since_ts).strftime('%Y-%m-%d %H:%M')
        window_end = _dt.now().strftime('%Y-%m-%d %H:%M')

        # If nothing to consider, send a short note
        if not pool:
            subject = f"AI Email Digest — {_dt.now().strftime('%Y-%m-%d')} (empty)"
            plain = (
                f"# AI Email Digest ({_dt.now().strftime('%Y-%m-%d')})\n"
                f"Considered: 0 emails (from {window_start} to {window_end}).\n\n"
                f"No emails met the criteria."
            )
            html_body = render_html_digest([], {}, "n/a", 0, window_start, window_end)
            send_digest_email(subject, plain, html_body)
            print("Digest sent (empty).")
            raise SystemExit(0)

        # Normal path (single pass)
        if len(pool) <= MAX_SINGLE_PASS:
            picks, model_used = select_top_k_via_enhanced_json(pool, window_start, window_end)
            emails_by_id = {e["id"]: e for e in pool}
        else:
            # Rare safety valve: batch reduce, then final JSON selection
            batch_size = 30
            per_batch_pick = 8
            prelim = []
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
                        prelim.append(chunk[idx-1])
            dedup = {(p['subject'], p['from_domain']): p for p in prelim}
            final_pool = list(dedup.values())

            picks, model_used = select_top_k_via_enhanced_json(final_pool, window_start, window_end)
            emails_by_id = {e["id"]: e for e in final_pool}

        # Compose and send
        plain = render_text_digest(picks, emails_by_id, model_used, len(pool), window_start, window_end)
        html_body = render_html_digest(picks, emails_by_id, model_used, len(pool), window_start, window_end)
        subject = f"AI Email Digest — {_dt.now().strftime('%Y-%m-%d')} (model: {model_used}; considered {len(pool)})"

        send_digest_email(subject, plain, html_body)
        print("Digest sent.")

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
