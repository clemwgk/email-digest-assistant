import os
import time
import json
import html
import re
import base64
import smtplib
import traceback
import datetime
from typing import List, Tuple, Optional

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
try:
    from openai import NotFoundError  # present in openai>=1.x
except Exception:  # pragma: no cover
    class NotFoundError(Exception):
        pass

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
# Env, constants & OpenAI client
# =========================================================
load_dotenv()  # no-op on Actions unless .env exists; useful locally

DEBUG = os.getenv("DEBUG", "").strip() == "1"
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# -- Model fallback list (per your request; keep as-is) --
MODEL_CANDIDATES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]
MAX_RETRIES = 4
BASE_BACKOFF = 2.0  # seconds

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Triaging tunables
SNIPPET_LEN = int(os.getenv("SNIPPET_LEN", "350"))            # ~how much context we send per email
TXN_ALERT_MIN = float(os.getenv("TXN_ALERT_MIN", "100"))      # demote "small" transactions below this
TXN_HIGH_ALERT_MIN = float(os.getenv("TXN_HIGH_ALERT_MIN", str(TXN_ALERT_MIN)))
ISSUER_DOMAINS = [d.strip().lower() for d in os.getenv("ISSUER_DOMAINS", "").split(",") if d.strip()]

# Include archived mail by default (NOT just Inbox)
STRICT_INBOX = os.getenv("STRICT_INBOX", "0") == "1"          # 0 = include archived, 1 = Inbox only

# Display timezone for the email (selection window remains UTC)
DISPLAY_TZ = ZoneInfo(os.getenv("DISPLAY_TZ", "Asia/Singapore"))

# Watermark buffer in hours (protect against late previous digests)
SINCE_BUFFER_HOURS = int(os.getenv("SINCE_BUFFER_HOURS", "6"))

# Words that hint deadlines/action (keep older thread items if present)
DEADLINE_WORDS = [
    "invoice", "bill", "payment", "due", "pay by",
    "action required", "deadline", "respond", "submit", "overdue",
    "iras", "hdb", "mom", "lta", "gov", "paynow"
]

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
def send_digest_email(subject: str, body_text: str, html_body: Optional[str] = None):
    sender = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_APP_PASSWORD")
    if not sender or not password:
        raise RuntimeError("Missing EMAIL_USERNAME or EMAIL_APP_PASSWORD in environment.")

    recipient = sender  # send to self
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Plain text part
    msg.attach(MIMEText(body_text, "plain"))

    # HTML part
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


def send_error_email(subject: str, error_text: str):
    try:
        html_err = f"<html><body><pre style='font-family:monospace;white-space:pre-wrap'>{html.escape(error_text)}</pre></body></html>"
        send_digest_email(subject, f"Email digest failed.\n\nDetails:\n{error_text}", html_err)
    except Exception as e:
        print("Failed to send error email:", e)


# =========================================================
# Body extraction & normalization
# =========================================================
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Neutralize last-4 patterns so amount parser won't pick them up
_CARD_LAST4_RE = re.compile(r"(?i)(visa|master\s?card|mastercard|amex|american express|discover|diners|jcb|unionpay|card(?:\s*ending)?|••••)\D{0,12}\b(\d{4})\b")
# Join split decimals like "25.\n76" → "25.76"
_DECIMAL_SPLIT_RE = re.compile(r"(\d)\.\s*\n\s*(\d{2})(?!\d)")

def _clean_html_to_text(html_str: str) -> str:
    # Preserve block breaks to avoid concatenating numbers
    text = html_str.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n").replace("</div>", "\n")
    text = _HTML_TAG_RE.sub("", text)
    return html.unescape(text)

def _neutralize_card_last4_text(text: str) -> str:
    return _CARD_LAST4_RE.sub(r"\1 •••• ####", text)

def _fix_split_decimals(text: str) -> str:
    return _DECIMAL_SPLIT_RE.sub(r"\1.\2", text)

def _urlsafe_b64decode(data: str) -> bytes:
    data += '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data)

def _extract_full_body_text(msg_data) -> str:
    """
    Extract full body text from Gmail message payload.
    - Prefer text/plain; fall back to cleaned text/html.
    - Preserve breaks; neutralize last-4; fix split decimals.
    """
    try:
        payload = msg_data.get("payload", {})

        # 1) Direct body (top-level)
        body_data = payload.get("body", {}).get("data")
        if body_data:
            text = _urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
            text = _neutralize_card_last4_text(_fix_split_decimals(text))
            return text.strip()

        # 2) Walk nested parts
        stack = [payload]
        parts_text_plain = []
        parts_text_html = []

        while stack:
            part = stack.pop()
            if not isinstance(part, dict):
                continue
            mime = part.get("mimeType", "")
            part_body = part.get("body", {}).get("data")
            parts = part.get("parts", [])
            if parts:
                stack.extend(parts)

            if part_body:
                raw = _urlsafe_b64decode(part_body).decode("utf-8", errors="ignore")
                if mime == "text/plain":
                    parts_text_plain.append(raw.strip())
                elif mime == "text/html":
                    parts_text_html.append(_clean_html_to_text(raw).strip())

        text = "\n".join(parts_text_plain) if parts_text_plain else "\n".join(parts_text_html)
        text = _neutralize_card_last4_text(_fix_split_decimals(text))
        return text.strip()
    except Exception:
        return ""


# =========================================================
# Amount parsing & transaction detection
# =========================================================
AMOUNT_CURR_RE = re.compile(
    r"(?ix)"
    r"(?:\b(SGD|S\$|USD|US\$|AUD|EUR|GBP|\$)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?))"
    r"|"
    r"((?:[0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(?:\.[0-9]{2})?)\s*(SGD|S\$|USD|US\$|AUD|EUR|GBP|\$)"
)

PCT_NEAR_RE = re.compile(r"(?i)%|\bpercent\b|\b%p\.a\b|p\.a\.")
MARKET_WORDS = {"bitcoin","btc","eth","stock","market","index","price","nav","fund","shares","forex","fx","rates"}
PROMO_WORDS  = {"sale","deal","offer","promo","voucher","discount","save","coupon","bundle"}
NEGATIVE_STATEMENT_WORDS = {
    "statement","estatement","e-statement","advice ready","available for viewing","account summary",
    "monthly statement","balance summary","eadvice"
}

TXN_VERBS = {
    "transaction alert","charged","purchase","payment made","receipt","debit","withdrawal",
    "pos","merchant","transfer","authorized","unauthorized","declined","failed","successful transaction",
    "card ending","spent","charge","autopay",
    # brokerage / finance confirmations
    "trade confirmation","order executed","executed","filled","contract note","trade filled","order filled"
}

def _parse_amounts_from_text(text: str) -> list[Tuple[str, float, Tuple[int,int]]]:
    """
    Return a list of (currency, amount_float, (start_idx, end_idx)) for all plausible amounts.
    Skips percentages.
    """
    out = []
    for m in AMOUNT_CURR_RE.finditer(text):
        if m.group(1):  # currency-first
            cur = m.group(1).upper()
            amt_str = m.group(2)
            span = m.span(0)
        else:           # amount-first
            cur = m.group(4).upper()
            amt_str = m.group(3)
            span = m.span(0)

        # Skip % contexts
        left = max(0, span[0]-3); right = min(len(text), span[1]+3)
        ctx = text[left:right]
        if PCT_NEAR_RE.search(ctx):
            continue

        # Normalize currency
        if cur in {"$", "US$"}:
            cur = "USD" if "US$" in cur else "SGD"
        elif cur == "S$":
            cur = "SGD"

        try:
            amt = float(amt_str.replace(",", ""))
        except Exception:
            continue

        out.append((cur, amt, span))
    return out

_NEAR_TXN_RE = re.compile("|".join(re.escape(w) for w in TXN_VERBS), re.I)
_NEAR_NEG_RE = re.compile(r"(?i)\b(coverage|limit|credit limit|sum insured|policy|premium|price|valued at|apr|interest)\b")

def _best_amount_by_context(text: str, cands: list[Tuple[str,float,Tuple[int,int]]], is_issuer: bool) -> Optional[Tuple[str,float,Tuple[int,int]]]:
    """
    Score amounts by nearby context:
      +3 if within ~80 chars of a txn verb
      -3 if within ~60 chars of negative contexts (coverage/price/limit/etc.)
      +1 if issuer domain
      + small tie-breaker for larger amount (gentle)
    Return the highest-scoring candidate; None if no candidates.
    """
    if not cands:
        return None

    def nearest_dist(regex: re.Pattern, pos: int) -> int:
        best = 10**9
        for m in regex.finditer(text):
            d = abs(m.start() - pos)
            if d < best:
                best = d
        return best

    best_score = -10**9
    best = None
    for (cur, amt, (a,b)) in cands:
        center = (a + b) // 2
        d_txn = nearest_dist(_NEAR_TXN_RE, center)
        d_neg = nearest_dist(_NEAR_NEG_RE, center)

        score = 0.0
        if d_txn <= 80: score += 3
        if d_neg <= 60: score -= 3
        if is_issuer:   score += 1
        score += (min(amt, 10000) / 10000.0) * 0.5  # gentle preference

        if score > best_score:
            best_score = score
            best = (cur, amt, (a,b))

    return best


def _domain_is_issuer(from_domain: str) -> bool:
    d = (from_domain or "").lower()
    return any(d.endswith(x) for x in ISSUER_DOMAINS) if ISSUER_DOMAINS else False


def enhanced_txn_detection_v2(subject: str, body: str,
                              parsed_cur: Optional[str], parsed_amt: Optional[float],
                              from_domain: str) -> Tuple[bool, Optional[str], Optional[float], bool]:
    """
    Decide if an email is a transaction alert (vs. promo/news/statement).
    Rules:
      - 'statement/advice' words override → NOT txn
      - Market/news/promos with amounts → NOT txn
      - If txn verb present + amount → likely txn
      - Else if issuer domain + amount → likely txn
      - Demote 'small' amounts via TXN_ALERT_MIN (returned as small=True)
    """
    text = f"{subject}\n{body}".lower()

    if any(w in text for w in NEGATIVE_STATEMENT_WORDS):
        return (False, None, None, False)

    if any(w in text for w in MARKET_WORDS) or any(w in text for w in PROMO_WORDS):
        return (False, None, None, False)

    has_txn_verb = any(w in text for w in TXN_VERBS)
    is_issuer = _domain_is_issuer(from_domain)

    if has_txn_verb and parsed_amt is not None:
        small = parsed_amt < TXN_ALERT_MIN
        return (True, parsed_cur, parsed_amt, small)

    if is_issuer and parsed_amt is not None:
        small = parsed_amt < TXN_ALERT_MIN
        return (True, parsed_cur, parsed_amt, small)

    if parsed_amt is not None:
        if any(w in text for w in ["coverage", "limit", "sum insured", "policy", "price", "valued at"]):
            return (False, None, None, False)

    return (False, None, None, False)


# =========================================================
# Snippet extraction (amount/action/date-focused, decimal-safe)
# =========================================================
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
ACTION_HINTS = re.compile(r'(?i)\b(due|deadline|expire|action required|respond|pay|submit|urgent|invoice|bill|payment|receipt)\b')
DATE_HINTS = re.compile(r'(?i)\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})\b')

def _extract_window(text: str, center: int, radius: int = 220) -> str:
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    seg = text[start:end]
    # Try to align to sentence boundaries
    rel = center - start
    prev = seg.rfind('.', 0, rel)
    if prev != -1:
        seg = seg[prev+1:]
        rel -= (prev + 1)
    nxt = seg.find('.', rel)
    if nxt != -1:
        seg = seg[:nxt+1]
    return seg.strip()

def _extract_smart_snippet_v2(full_text: str, max_len: int, preferred_span: Optional[Tuple[int,int]]) -> str:
    if not full_text:
        return "[Could not extract body]"

    if preferred_span:
        center = (preferred_span[0] + preferred_span[1]) // 2
        win = _extract_window(full_text, center)
        out = re.sub(r'\s+', ' ', win).strip()
        return out[:max_len]

    sentences = SENT_SPLIT.split(full_text)
    scored = []
    for s in sentences:
        score = 0
        if ACTION_HINTS.search(s):
            score += 2
        if DATE_HINTS.search(s):
            score += 1
        if AMOUNT_CURR_RE.search(s):
            score += 1
        if score:
            scored.append((score, s.strip()))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        out = " ".join(s for _, s in scored[:3])
    else:
        out = full_text[:max_len]

    out = re.sub(r'\s+', ' ', out).strip()
    return out[:max_len]


# =========================================================
# Watermark = last sent digest time
# =========================================================
def get_last_run_timestamp(service) -> int:
    """
    Look in Sent for the most recent digest (subject starts with 'AI Email Digest —')
    and use its internalDate (ms) as the watermark. If none found, fallback to now-24h.
    """
    query = 'in:sent subject:"AI Email Digest —"'
    now = datetime.datetime.now(datetime.timezone.utc)
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
# Fetch all emails since watermark (archived mail included)
# =========================================================
def fetch_all_since_v2(service, since_unix: int, snippet_len: int = SNIPPET_LEN):
    """
    Build the full pool since the last run.
      - Include archived mail by default (STRICT_INBOX=0)
      - Drop OTP/verification emails
      - Keep latest per thread; keep older items only if they contain deadline words
      - Exclude past digests (subject:"AI Email Digest —")
      - Parse amounts (multi-candidate) → choose best by context → extract snippet → detect txn
    """
    base_q = f'after:{since_unix} -subject:"AI Email Digest —"'
    query = f'in:inbox {base_q}' if STRICT_INBOX else base_q

    if DEBUG:
        print("[Query]", query)

    pool: List[dict] = []
    latest_by_thread: dict[str, dict] = {}
    page_token = None

    while True:
        res = service.users().messages().list(
            userId='me', q=query, maxResults=100, pageToken=page_token
        ).execute()
        msgs = res.get('messages', [])
        page_token = res.get('nextPageToken')

        # Fallback if totally empty first page: broaden to last 2 days without in:inbox
        if not msgs and page_token is None:
            if DEBUG:
                print("[Query] 0 results. Trying fallback: newer_than:2d -subject:\"AI Email Digest —\"")
            fallback_q = 'newer_than:2d -subject:"AI Email Digest —"'
            res = service.users().messages().list(userId='me', q=fallback_q, maxResults=100).execute()
            msgs = res.get('messages', [])
            page_token = res.get('nextPageToken')

        for m in msgs:
            msg = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            if subject.startswith("AI Email Digest —"):
                continue  # never include our own digests

            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'N/A')
            ts = int(msg.get('internalDate', '0')) // 1000
            thread_id = msg.get('threadId')

            full_body = _extract_full_body_text(msg)
            combined_lower = (subject + ' ' + full_body).lower()

            # OTP / verification drop
            if any(k in combined_lower for k in ['otp', 'verification code', '2fa', 'login code', 'one-time password']):
                continue

            # From domain (best-effort) + issuer flag
            from_domain = sender
            if '@' in sender:
                try:
                    from_domain = sender.split('@')[-1].split('>')[0].strip().lower()
                except Exception:
                    pass
            is_issuer = _domain_is_issuer(from_domain)

            # Consider ALL amounts; pick the best by context (txn verbs, negative words, issuer)
            cands = _parse_amounts_from_text(full_body)
            best = _best_amount_by_context(combined_lower, cands, is_issuer)
            parsed_cur = best[0] if best else None
            parsed_amt = best[1] if best else None
            amount_span = best[2] if best else None

            # Snippet around the chosen amount (or action/date hints)
            snippet = _extract_smart_snippet_v2(full_body, max_len=snippet_len, preferred_span=amount_span)

            # Txn detection
            is_txn, cur, amt, small = enhanced_txn_detection_v2(
                subject, full_body, parsed_cur, parsed_amt, from_domain
            )

            item = {
                'id': m['id'],
                'threadId': thread_id,
                'timestamp': ts,
                'from_domain': from_domain,
                'from_raw': sender,
                'subject': subject,
                'snippet': snippet,
                'has_deadline_word': any(w in combined_lower for w in DEADLINE_WORDS),
                'txn_alert': is_txn,
                'txn_currency': cur,
                'txn_amount': amt,
                'txn_small': bool(small),
            }

            if DEBUG and is_txn:
                debug_print(f"[TXN] subj={subject!r} amount={amt} {cur} small={small} issuer={from_domain} snippet={snippet[:160]!r}")

            prev = latest_by_thread.get(thread_id)
            if not prev or ts >= prev['timestamp']:
                latest_by_thread[thread_id] = item
            elif item['has_deadline_word']:
                pool.append(item)

        if not page_token:
            break

    pool.extend(latest_by_thread.values())
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
            except (NotFoundError, RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
                errors.append((f"{m} attempt {attempt}", f"{e.__class__.__name__}: {e}"))
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF * attempt)  # 2s, 4s, 6s, ...
                    continue
                break
            except Exception as e:
                errors.append((m, f"{e.__class__.__name__}: {e}"))
                break
    combined = "\n".join([f"[{model}] {err}" for model, err in errors])
    raise RuntimeError(f"All GPT model calls failed.\n{combined}")

def build_enhanced_selection_prompt_json(emails: list[dict]) -> str:
    K = min(5, len(emails))
    header = (
        "You are a personal email triage assistant.\n\n"
        "INPUT:\n"
        f"- EMAIL_COUNT: {len(emails)}\n"
        f"- RETURN_COUNT: {K}\n"
        "- Each item includes: id, from_domain, subject, snippet, time, issuer (bool), "
        "txn_alert (bool), txn_amount (number|null), txn_currency, txn_small.\n\n"
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
        "   - contains terms like: declined, failed, suspicious, unauthorized, dispute, fraud\n"
        "   - unusual merchant hints (use judgment)\n"
        "7) Market/news/promos (e.g., Bitcoin price, coverage limits, discounts) are generally NOT transactions and should be deprioritized.\n"
        "8) `issuer=true` is a weak prior; still deprioritize amounts without clear action/txn context.\n"
        # >>> YOUR NEW SUB-RULES (deterministic preference within Billing/Payment) <<<
        "9) Within **Billing/Payment** emails:\n"
        f"   - Any transaction with amount ≥ {TXN_ALERT_MIN} must outrank any transaction < {TXN_ALERT_MIN}, "
        "unless the smaller one contains: declined, failed, unauthorized, suspicious, dispute, fraud.\n"
        "   - If still tied, prefer explicit Call to Action, then newer.\n\n"
        "For each selection, provide details:\n"
        "- **why**: 2-3 sentences explaining: (1) what the email is about, (2) why it matters, (3) what might need to be done\n"
        "- **action**: specific action needed if any (e.g., 'Pay by Aug 20', 'Review and respond', 'No action needed')\n"
        "- **urgency**: High/Medium/Low based on deadlines and consequences\n\n"
        "RULES (CRITICAL):\n"
        "- Select only from provided `id` values. DO NOT invent emails.\n"
        "- If EMAIL_COUNT < 5, return exactly EMAIL_COUNT items.\n"
        "- Output ONLY valid JSON in this schema and nothing else:\n"
        '{\n'
        '  "picks": [\n'
        '    {"id": "<id>", "type": "Call to Action" | "For Information Only", '
        '     "category": "Legal" | "Government" | "Billing/Payment" | "Other", '
        '     "why": "<2-3 sentences>", '
        '     "action": "<specific action or No action needed>", '
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
            "time": datetime.datetime.fromtimestamp(e["timestamp"], tz=datetime.timezone.utc).isoformat(),
            "issuer": bool(_domain_is_issuer(e["from_domain"])),
            "txn_alert": bool(e.get("txn_alert")),
            "txn_amount": e.get("txn_amount"),
            "txn_currency": e.get("txn_currency"),
            "txn_small": bool(e.get("txn_small", False)),
        }, ensure_ascii=False))
    return header + "\n".join(lines)

def parse_enhanced_json_picks(raw: str, valid_ids: set, want_count: int):
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
                    "id": _id,
                    "type": _type,
                    "category": _cat,
                    "why": _why.strip(),
                    "action": _action.strip(),
                    "urgency": _urgency
                })
            if len(picks) >= want_count:
                break
    except Exception:
        picks = []
    return picks


# =========================================================
# Renderers (Markdown + HTML + Appendix)
# =========================================================
def render_text_digest(picks, emails_by_id, model_used, considered_count, window_start, window_end):
    from datetime import datetime as _dt
    header = (
        f"# AI Email Digest ({_dt.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d')})\n"
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

def render_html_digest(picks, emails_by_id, model_used, considered_count, window_start, window_end, extra_html: str = ""):
    # Gmail-friendly inline CSS
    items_html = []
    for i, p in enumerate(picks, 1):
        e = emails_by_id[p["id"]]
        subj = html.escape(e["subject"])
        dom = html.escape(e["from_domain"])
        snip = html.escape(e["snippet"])
        why = html.escape(p["why"])
        action = html.escape(p["action"])

        # badges
        type_bg = "#fde68a" if p["type"] == "Call to Action" else "#dbeafe"  # amber vs blue
        cat_bg_map = {"Legal": "#fecaca", "Government": "#fde68a", "Billing/Payment": "#bbf7d0", "Other": "#e5e7eb"}
        cat_bg = cat_bg_map.get(p["category"], "#e5e7eb")

        urgency_bg_map = {"High": "#fecaca", "Medium": "#fde68a", "Low": "#d1fae5"}
        urgency_bg = urgency_bg_map.get(p["urgency"], "#e5e7eb")

        txn_chip = ""
        if e.get("txn_alert"):
            amt = e.get("txn_amount")
            cur = e.get("txn_currency") or "SGD"
            if amt is not None:
                over = amt >= TXN_HIGH_ALERT_MIN
                txn_chip = _badge(f"Txn {cur} {amt:.2f}", "#bbf7d0" if over else "#e5e7eb")
            else:
                txn_chip = _badge("Txn alert", "#e5e7eb")

        items_html.append(f"""
          <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:10px 0;background:#ffffff">
            <div style="font-weight:700;font-size:15px;line-height:1.3;margin-bottom:4px">{i}. {subj}</div>
            <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{dom}</div>
            <div style="margin-bottom:8px">
              {_badge(p["type"], type_bg)}
              <span style="display:inline-block;width:6px"></span>
              {_badge(p["category"], cat_bg)}
              <span style="display:inline-block;width:6px"></span>
              {_badge(f"Urgency: {p['urgency']}", urgency_bg)}
              {'<span style="display:inline-block;width:6px"></span>' + txn_chip if txn_chip else ''}
            </div>
            <div style="font-size:13px;margin-bottom:6px"><strong>Why:</strong> {why}</div>
            <div style="font-size:13px;margin-bottom:6px"><strong>Action:</strong> {action}</div>
            <div style="font-size:12px;color:#6b7280"><strong>Snippet:</strong> {snip}</div>
          </div>
        """)
    items_html = "\n".join(items_html) if items_html else "<p><em>No eligible emails.</em></p>"

    title = f"AI Email Digest ({datetime.datetime.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d')})"
    header_html = f"""
      <div style="font-family:Inter,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:16px 12px;background:#f8fafc">
        <h2 style="margin:0 0 4px 0;font-size:20px">{html.escape(title)}</h2>
        <div style="color:#6b7280;font-size:12px;margin-bottom:12px">
          Model: <strong>{html.escape(model_used)}</strong> · Considered: <strong>{considered_count}</strong> emails · Window: {html.escape(window_start)} → {html.escape(window_end)}
        </div>
        {items_html}
        {extra_html}
        <div style="color:#9ca3af;font-size:11px;margin-top:12px">— Generated by your Email Digest Assistant</div>
      </div>
    """
    return f"<html><body>{header_html}</body></html>"

def render_text_appendix(pool, cap=100):
    lines = ["\n\n---\n## Appendix — All considered (" + str(len(pool)) + ")\n"]
    for e in pool[:cap]:
        t = datetime.datetime.fromtimestamp(e['timestamp'], tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M')
        lines.append(f"- [{t}] {e.get('from_raw','?')} — {e['subject']}")
    if len(pool) > cap:
        lines.append(f"... and {len(pool)-cap} more")
    return "\n".join(lines)

def render_html_appendix(pool, cap=100):
    rows = []
    for e in pool[:cap]:
        t = html.escape(datetime.datetime.fromtimestamp(e['timestamp'], tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M'))
        f = html.escape(e.get('from_raw','?'))
        s = html.escape(e['subject'])
        rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;color:#6b7280;font-size:12px'>{t}</td>"
            f"<td style='padding:6px 8px;font-size:12px'>{f}</td>"
            f"<td style='padding:6px 8px;font-size:12px'>{s}</td>"
            "</tr>"
        )
    more = ""
    if len(pool) > cap:
        more = f"<div style='color:#6b7280;font-size:12px;margin-top:6px'>… and {len(pool)-cap} more</div>"
    table = (
        "<table style='border-collapse:collapse;width:100%;margin-top:8px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Time</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>From</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Subject</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return (
        "<div style='margin-top:14px'>"
        "<h3 style='margin:0 0 6px 0;font-size:14px'>Appendix — All considered (" + str(len(pool)) + ")</h3>"
        + table + more + "</div>"
    )


# =========================================================
# Selection wrapper
# =========================================================
def select_top_k_via_enhanced_json(pool, window_start, window_end):
    want = min(5, len(pool))
    prompt = build_enhanced_selection_prompt_json(pool)
    messages = [{"role": "user", "content": prompt}]
    raw, model_used = call_gpt_with_fallback(messages)

    valid_ids = {e["id"] for e in pool}
    picks = parse_enhanced_json_picks(raw, valid_ids, want)

    # Deterministic fallback if model output is empty/invalid
    if len(picks) < want:
        missing = want - len(picks)
        chosen = {p["id"] for p in picks}
        fallback_items = [e for e in pool if e["id"] not in chosen][:missing]
        picks += [{
            "id": e["id"],
            "type": "For Information Only",
            "category": "Other",
            "why": "Selected by fallback (newest) due to invalid/empty model output. Please review manually.",
            "action": "Review manually",
            "urgency": "Low"
        } for e in fallback_items]
    return picks, model_used


# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    try:
        service = authenticate_gmail()

        # Optional: confirm which Gmail account we're reading (DEBUG only)
        if DEBUG:
            try:
                profile = service.users().getProfile(userId='me').execute()
                print("Gmail API profile:", profile.get("emailAddress"))
            except Exception as _e:
                print("Profile check failed:", _e)

        # Determine window with buffer (selection uses UTC)
        since_ts = get_last_run_timestamp(service)
        since_ts = max(0, since_ts - SINCE_BUFFER_HOURS * 3600)

        # Build pool (includes archived mail by default)
        pool = fetch_all_since_v2(service, since_ts, snippet_len=SNIPPET_LEN)

        # Display window in DISPLAY_TZ
        window_start = datetime.datetime.fromtimestamp(since_ts, tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M')
        window_end = datetime.datetime.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M')

        # If nothing to consider, send a short note (still include empty appendix)
        if not pool:
            subject = f"AI Email Digest — {datetime.datetime.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d')} (empty)"
            plain = (
                f"# AI Email Digest ({datetime.datetime.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d')})\n"
                f"Considered: 0 emails (from {window_start} to {window_end}).\n\n"
                f"No emails met the criteria."
            )
            html_body = render_html_digest([], {}, "n/a", 0, window_start, window_end, extra_html=render_html_appendix(pool, cap=100))
            send_digest_email(subject, plain, html_body)
            print("Digest sent (empty).")
            raise SystemExit(0)

        # Selection (single pass; your daily volume ~20–40)
        picks, model_used = select_top_k_via_enhanced_json(pool, window_start, window_end)
        emails_by_id = {e["id"]: e for e in pool}

        # Compose bodies (+ appendix)
        plain = render_text_digest(picks, emails_by_id, model_used, len(pool), window_start, window_end)
        plain += render_text_appendix(pool, cap=100)

        appendix_html = render_html_appendix(pool, cap=100)
        html_body = render_html_digest(picks, emails_by_id, model_used, len(pool), window_start, window_end, extra_html=appendix_html)

        subject = f"AI Email Digest — {datetime.datetime.now(tz=datetime.timezone.utc).astimezone(DISPLAY_TZ).strftime('%Y-%m-%d')} (model: {model_used}; considered {len(pool)})"
        send_digest_email(subject, plain, html_body)
        print("Digest sent.")

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
