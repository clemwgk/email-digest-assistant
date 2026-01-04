# main.py — Gmail AI Email Digest

import os
import time
import json
import html
import re
import base64
import smtplib
import traceback
import datetime
from typing import List, Tuple, Optional, Dict

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from openai import OpenAI

# -----------------------------
# Env & constants
# -----------------------------
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Accept either GOOGLE_* or GMAIL_*; else fall back to on-disk files
OAUTH_B64 = os.getenv("GOOGLE_OAUTH_JSON_B64") or os.getenv("GMAIL_CREDENTIALS_JSON_B64") or ""
TOKEN_B64 = os.getenv("GOOGLE_TOKEN_JSON_B64") or os.getenv("GMAIL_TOKEN_JSON_B64") or ""

ISSUER_DOMAINS = [d.strip().lower() for d in os.getenv("ISSUER_DOMAINS", "").split(",") if d.strip()]
STRICT_INBOX = os.getenv("STRICT_INBOX", "0").strip() == "1"
DISPLAY_TZ_NAME = os.getenv("DISPLAY_TZ", "Asia/Singapore")
DISPLAY_TZ = ZoneInfo(DISPLAY_TZ_NAME)
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
DISABLE_DIGEST = os.getenv("DISABLE_DIGEST", "0").strip() == "1"

# SMTP config (auto-detect 465 vs 587)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = [x.strip() for x in os.getenv("MAIL_TO", "").split(",") if x.strip()]

# Model fallback order (keep exact order)
MODEL_CANDIDATES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]
BASE_BACKOFF = 2.0  # seconds

# OpenAI client (lazily initialized to avoid import-time crashes when the key is missing)
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else None

# Tunables
SNIPPET_LEN = int(os.getenv("SNIPPET_LEN", "500"))
TXN_ALERT_MIN = float(os.getenv("TXN_ALERT_MIN", "100"))
TXN_HIGH_ALERT_MIN = float(os.getenv("TXN_HIGH_ALERT_MIN", "100"))
SINCE_BUFFER_HOURS = int(os.getenv("SINCE_BUFFER_HOURS", "6"))

# -----------------------------
# Utils
# -----------------------------
def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def sg_time(ts: int) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")

def _extract_headers(msg) -> dict:
    headers = {}
    for h in msg.get('payload', {}).get('headers', []):
        headers[h.get('name', '').lower()] = h.get('value', '')
    return headers

def _get_header(headers: dict, name: str, default: str = "") -> str:
    return headers.get(name.lower(), default)

# Always return a safe text string (prevents html.escape crashes)
def _as_text(v) -> str:
    return v if isinstance(v, str) else ""

# -----------------------------
# Gmail auth (accept env b64 or existing files)
# -----------------------------
def _gmail_service():
    creds = None
    oauth_path = "credentials.json"
    token_path = "token.json"

    # Write files if provided via secrets
    if OAUTH_B64:
        with open(oauth_path, "wb") as f:
            f.write(base64.b64decode(OAUTH_B64))
    if TOKEN_B64:
        with open(token_path, "wb") as f:
            f.write(base64.b64decode(TOKEN_B64))

    if not os.path.exists(oauth_path):
        raise RuntimeError("Missing credentials.json (provide GOOGLE_OAUTH_JSON_B64 or GMAIL_CREDENTIALS_JSON_B64).")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # For local/manual runs only
            flow = InstalledAppFlow.from_client_secrets_file(oauth_path, SCOPES)
            creds = flow.run_local_server(port=0)

    with open(token_path, "w") as f:
        f.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

# -----------------------------
# HTML/text extraction + normalization
# -----------------------------
def _html_to_text(s: str) -> str:
    s = re.sub(r'(?is)<script.*?>.*?</script>', ' ', s)
    s = re.sub(r'(?is)<style.*?>.*?</style>', ' ', s)
    s = re.sub(r'(?is)<br[^>]*>', '\n', s)
    s = re.sub(r'(?is)</p>', '\n', s)
    s = re.sub(r'(?is)<.*?>', ' ', s)
    s = html.unescape(s)
    # remove zero-width chars like &zwnj;
    s = s.replace("\u200c", "").replace("\u200b", "")
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _post_process_text(s: str) -> str:
    # Fix broken decimals across line breaks: "25.\n76" -> "25.76"
    s = re.sub(r'(?m)(\d+)\.\s*\n\s*(\d{2})', r'\1.\2', s)
    # Mask card last-4 so they aren't parsed as money
    s = re.sub(r'(?i)\b(ending(?: in)?|last(?:\s*4|\s*four)\s*(?:digits?)?)\s*(\d{4})\b', r'\1 ‹last4›', s)
    s = re.sub(r'(?i)(?:\*{4}|x{4})[\s\-]?(\d{4})\b', r'**** ‹last4›', s)
    return s

def _extract_full_body_text(msg) -> str:
    def decode_data(data_b64: Optional[str]) -> str:
        if not data_b64:
            return ""
        try:
            return base64.urlsafe_b64decode(data_b64.encode('UTF-8')).decode('utf-8', errors='replace')
        except Exception:
            return ""
    payload = msg.get('payload', {})
    mimeType = payload.get("mimeType", "")
    if mimeType == "text/plain":
        return _post_process_text(decode_data(payload.get("body", {}).get("data")))
    if mimeType == "text/html":
        return _post_process_text(_html_to_text(decode_data(payload.get("body", {}).get("data"))))
    if mimeType.startswith("multipart/"):
        txt = []
        parts = payload.get("parts", []) or []
        def walk(parts):
            for p in parts:
                mt = p.get("mimeType", "")
                if mt == "text/plain":
                    txt.append(decode_data(p.get('body', {}).get('data')))
                elif mt == "text/html":
                    txt.append(_html_to_text(decode_data(p.get('body', {}).get('data'))))
                elif mt.startswith("multipart/"):
                    walk(p.get("parts", []) or [])
        walk(parts)
        return _post_process_text("\n".join([t for t in txt if t]))
    return _post_process_text("")

# -----------------------------
# OTP/2FA detection — mask (do not drop)
# -----------------------------
OTP_KEYWORDS_RE = re.compile(r'(?i)\b(otp|one[- ]?time password|verification code|2fa|login code|security code)\b')
OTP_CODE_RE = re.compile(r'(?i)(?<![A-Z0-9])([A-Z0-9]{4,8})(?![A-Z0-9])')
OTP_NEG_CUE_RE = re.compile(r'(?i)(do not share|never (?:ask|request)|we will never (?:ask|request)|stay secure|for your security|phishing|scam)')

def classify_and_mask_otp(subject: str, body: str, window: int = 80) -> Tuple[bool, str, str]:
    text = (subject or "") + "\n" + (body or "")
    otp_like = False
    for m in OTP_KEYWORDS_RE.finditer(text):
        left = max(0, m.start() - window)
        right = min(len(text), m.end() + window)
        window_text = text[left:right]
        if OTP_NEG_CUE_RE.search(window_text):
            continue
        if OTP_CODE_RE.search(window_text):
            otp_like = True
            break
    if not otp_like:
        return False, subject, body

    def _mask(s: str) -> str:
        return OTP_CODE_RE.sub("‹code›", s or "")

    return True, _mask(subject), _mask(body)

# -----------------------------
# Amount parsing & context (with refined heuristics)
# -----------------------------
AMOUNT_CURR_RE = re.compile(r'(?i)\b(SGD|USD|EUR|GBP|\$)\s?([0-9]{1,3}(?:[,\s][0-9]{3})*|[0-9]+)(\.[0-9]{2})?\b')
AMOUNT_SIMPLE_RE = re.compile(r'(?<![0-9])([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(\.[0-9]{2})\b')

# Tightened: no bare "transaction" token
_NEAR_TXN_RE = re.compile(r'(?i)\b(transaction alert|charged|debited|purchase(?:d)?|withdrawal|transfer(?:red)?|authorized|unauthorized|declined|failed|credited|you(?:’|\'|)ve received|you received)\b')
_NEUTRAL_TXN_RE = re.compile(r'(?i)\b(transaction alert|transfer status|funds transfer|fund transfer)\b')
_BILLING_CUE_RE = re.compile(r'(?i)\b(invoice|payment due|bill(?:ing)? statement|pay by|due on|outstanding balance|amount due)\b')
_BOOKING_CUE_RE = re.compile(r'(?i)\b(reservation|booking|check[- ]?in|check[- ]?out|confirmation|hotel|guest|stay|inquiry|question|request)\b')
# Escape literal # (not a comment) and avoid verbose mode to prevent parse errors
_RESERVATION_REF_RE = re.compile(
    r"(?i)\b("
    r"reservation|booking|confirmation|itinerary|record\s+locator|pnr|ticket"
    r")\s*(?:no\.?|\#|num(?:ber)?|code|id|reference|ref|:)?\s*([A-Z0-9]{5,12})\b"
)
_MEMBERSHIP_WORD_RE = re.compile(r'(?i)\b(membership|member|loyalty|rewards?|points|club|tier)\b')

# Hard negatives near amount (coverage/insurance/limits)
_NEAR_NEG_RE = re.compile(
    r"(?i)\b("
    r"coverage|cover|sum insured|insured|insurance|deposit insurance|scheme|by law|"
    r"up to|maximum|max\.?|cap|capped|limit(?:ed)? to|per depositor|per account|"
    r"policy|premium"
    r")\b"
)

# Soft negatives: promo/T&C — scoring only
_SOFT_NEG_SCORE_RE = re.compile(
    r"(?i)\b(terms and conditions|tnc|for full terms|promotion|promo|offer|voucher|promo code|unsubscribe|manage preferences)\b"
)

CAP_WORDS = re.compile(
    r"(?i)\b(up to|maximum|max\.?|cap|capped|limit(?:ed)? to|coverage|insured|insurance|deposit insurance|per depositor|per account|scheme|by law)\b"
)

def _money_from_groups(cur: Optional[str], num: Optional[str], cents: Optional[str]) -> Optional[Tuple[str, float]]:
    if num is None:
        return None
    n = num.replace(",", "").replace(" ", "")
    try:
        val = float(n + (cents or ""))
    except Exception:
        return None
    if cur is None or cur == "$":
        cur = "SGD"
    return (cur, val)

def _parse_amounts_from_text(text: str) -> List[Tuple[str,float,Tuple[int,int]]]:
    out = []
    for m in AMOUNT_CURR_RE.finditer(text):
        cur, amt = _money_from_groups(m.group(1), m.group(2), m.group(3))
        if cur and amt is not None:
            out.append((cur, amt, (m.start(), m.end())))
    for m in AMOUNT_SIMPLE_RE.finditer(text):
        try:
            amt = float(m.group(1).replace(",", "") + m.group(2))
            out.append(("SGD", amt, (m.start(), m.end())))
        except Exception:
            pass
    return out

def _window_has(regex: re.Pattern, text: str, span: Tuple[int,int], win: int) -> bool:
    a, b = span
    left = max(0, a - win); right = min(len(text), b + win)
    return bool(regex.search(text[left:right]))

def _best_amount_by_context(text: str, cands: List[Tuple[str,float,Tuple[int,int]]], is_issuer: bool) -> Optional[Tuple[str,float,Tuple[int,int]]]:
    """
    Score amounts by nearby context:
      +3 if txn verb within ~80 chars
      -3 if hard negative within ~60 chars (coverage/insurance/limits)
      -1.5 if soft-negative within ~80 chars (T&C, promo, etc.)
      +1 if issuer domain
      + tiny tie-break for larger amount
    """
    if not cands:
        return None

    text_lower = text.lower()

    def nearest_dist(regex: re.Pattern, pos: int) -> int:
        best = 10**9
        for m in regex.finditer(text_lower):
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
        d_soft = nearest_dist(_SOFT_NEG_SCORE_RE, center)
        score = 0.0
        if d_txn <= 80: score += 3
        if d_neg <= 60: score -= 3
        if d_soft <= 80: score -= 1.5
        billing_near = _window_has(_BILLING_CUE_RE, text, (a,b), win=120)
        neutral_txn_near = _window_has(_NEUTRAL_TXN_RE, text, (a,b), win=140)
        if neutral_txn_near and not billing_near:
            score -= 2.5
        if is_issuer: score += 1
        score += min(amt, 1000) / 100000.0
        if score > best_score:
            best_score = score
            best = (cur, amt, (a,b))
    return best

# -----------------------------
# Snippet extraction (amount-/action-aware)
# -----------------------------
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _extract_window(text: str, center: int, radius: int = 220) -> str:
    a = max(0, center - radius)
    b = min(len(text), center + radius)
    seg = text[a:b]
    seg = re.sub(r'\s+', ' ', seg).strip()
    return seg

def _extract_smart_snippet_v2(full_text: str, max_len: int, preferred_span: Optional[Tuple[int,int]]) -> str:
    if not full_text:
        return "[Could not extract body]"
    if preferred_span:
        center = (preferred_span[0] + preferred_span[1]) // 2
        win = _extract_window(full_text, center)
        return win[:max_len]
    sentences = SENT_SPLIT.split(full_text)
    scored = []
    for s in sentences:
        score = 0
        if _NEAR_TXN_RE.search(s): score += 2
        if _BILLING_CUE_RE.search(s) and not _NEUTRAL_TXN_RE.search(s): score += 1.5
        if re.search(r'(?i)\b(due|deadline|expire|expiry|renew|invoice|pay by|payment due|statement)\b', s): score += 1
        if _BOOKING_CUE_RE.search(s): score += 1
        if _has_reservation_reference(s): score += 1.5
        if AMOUNT_CURR_RE.search(s): score += 1
        if score: scored.append((score, s.strip()))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return re.sub(r'\s+', ' ', scored[0][1])[:max_len]
    return re.sub(r'\s+', ' ', full_text)[:max_len]

# -----------------------------
# Promo tagging for ranking fixes (A)
# -----------------------------
PROMO_NEAR_AMOUNT = re.compile(
    r"(?i)\b(spend (?:a )?minimum|stand a chance to win|giveaway|voucher|flash deal|rebate|bonus miles|rsvp|t&cs apply|terms and conditions apply)\b"
)

def looks_adv(subject: str) -> bool:
    return (subject or "").strip().upper().startswith("<ADV>")

def promo_like_near_amount(text: str, span: Optional[Tuple[int,int]], win: int = 120) -> bool:
    if not span:
        return False
    a, b = span
    left = max(0, a - win); right = min(len(text), b + win)
    return bool(PROMO_NEAR_AMOUNT.search(text[left:right]))

# -----------------------------
# Social / billing / booking cues
# -----------------------------
SOCIAL_DOMAINS = {
    "linkedin.com",
    "notifications.linkedin.com",
    "facebookmail.com",
    "messenger.com",
}
SOCIAL_KEYWORDS_RE = re.compile(r'(?i)\b(connection|invitation|profile|endorsed|people (?:viewed|are viewing)|add you|join my network)\b')

def _is_social_notification(from_domain: str, subject: str, snippet: str) -> bool:
    domain = (from_domain or "").lower()
    if any(domain.endswith(d) for d in SOCIAL_DOMAINS):
        return True
    text = f"{subject}\n{snippet}"
    return bool(SOCIAL_KEYWORDS_RE.search(text))

def _sanitize_snippet_for_social(snippet: str) -> str:
    if not snippet:
        return snippet
    # Collapse bio-like separators early
    cleaned = re.split(r"\s*[•·|]\s*", snippet, maxsplit=1)[0]
    cleaned = re.sub(r"(?i)\b(viewed your profile|see who|people you may know).*", "", cleaned).strip()
    return cleaned or snippet

def _has_billing_cue(text: str, span: Optional[Tuple[int,int]]) -> bool:
    if span:
        billing_near = _window_has(_BILLING_CUE_RE, text, span, win=140)
        neutral_txn_near = _window_has(_NEUTRAL_TXN_RE, text, span, win=140)
        return bool(billing_near and not neutral_txn_near)
    return bool(_BILLING_CUE_RE.search(text))

def _is_reply_email(headers: dict, subject: str) -> bool:
    subj = (subject or "").lstrip()
    if subj.lower().startswith("re:"):
        return True
    return bool(headers.get("in-reply-to") or headers.get("references"))

def _has_booking_cue(text: str) -> bool:
    return bool(_BOOKING_CUE_RE.search(text))

def _has_reservation_reference(text: str) -> bool:
    for m in _RESERVATION_REF_RE.finditer(text):
        # Ignore matches that are clearly tied to memberships/loyalty programs (marketing heavy)
        left = max(0, m.start() - 48)
        right = min(len(text), m.end() + 48)
        ctx = text[left:right]
        if _MEMBERSHIP_WORD_RE.search(ctx):
            continue
        return True
    return False

# -----------------------------
# Gmail superset fetch + local filter
# -----------------------------
def _with_retries(fn, what: str, tries: int = 5, base_sleep: float = 0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            sl = base_sleep * (2 ** i)
            debug_print(f"{what} attempt {i+1}/{tries} failed: {e} (sleep {sl:.2f}s)")
            time.sleep(sl)
    raise last

def _build_superset_query(days: int, strict_inbox: bool) -> str:
    scope = "in:inbox" if strict_inbox else "in:anywhere"
    # Exclude our own sent mail
    return (
        f'newer_than:{days}d {scope} '
        f'-in:spam -in:trash -in:sent -from:me '
        f'-subject:"AI Email Digest —"'
    )

def _list_message_ids(service, q: str, max_pages: int = 50, page_size: int = 500) -> List[Dict[str,str]]:
    user_id = "me"
    out: List[Dict[str,str]] = []
    page_token = None
    for p in range(max_pages):
        def _call():
            return service.users().messages().list(
                userId=user_id, q=q, pageToken=page_token,
                maxResults=page_size, includeSpamTrash=False
            ).execute()
        resp = _with_retries(_call, what=f"messages.list[{p}]")
        msgs = resp.get("messages", []) or []
        out.extend(msgs)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out

def _get_messages_minimal(service, ids: List[str]) -> List[Dict]:
    user_id = "me"
    res: List[Dict] = []
    for mid in ids:
        def _call():
            return service.users().messages().get(userId=user_id, id=mid, format="minimal").execute()
        msg = _with_retries(_call, what=f"messages.get[min:{mid}]")
        if "internalDate" in msg:
            res.append(msg)
    return res

def _get_messages_full(service, ids: List[str]) -> List[Dict]:
    user_id = "me"
    res: List[Dict] = []
    for mid in ids:
        def _call():
            return service.users().messages().get(userId=user_id, id=mid, format="full").execute()
        msg = _with_retries(_call, what=f"messages.get[full:{mid}]")
        res.append(msg)
    return res

def fetch_all_since_v2(service, since_unix: int, snippet_len: int = SNIPPET_LEN):
    """
    Build pool since watermark:
      - Superset Gmail query (3d; 7d fallback) + local filter internalDate >= since_unix
      - Include archived mail by default; exclude spam/trash; exclude past digests; exclude Sent/from:me
      - OTP/code masking (do not drop)
      - Thread de-dup (keep latest; keep older w/ deadline words)
      - Amount parsing + context → amount-centered snippet → txn flag
      - Tagging for ranking fixes: is_adv, promo_like
    """
    q3 = _build_superset_query(3, STRICT_INBOX)
    debug_print(f"gmail.superset.query(primary) = {q3}")
    raw_ids_3 = _list_message_ids(service, q3)
    debug_print(f"gmail.superset.raw_ids(primary) = {len(raw_ids_3)}")

    since_ms = int(since_unix) * 1000
    mini_3 = _get_messages_minimal(service, [m["id"] for m in raw_ids_3])
    filtered = [m for m in mini_3 if int(m.get("internalDate", 0)) >= since_ms]

    SMALL_THRESHOLD = 5
    if len(filtered) < SMALL_THRESHOLD:
        q7 = _build_superset_query(7, STRICT_INBOX)
        debug_print(f"gmail.superset.pool_small={len(filtered)}<={SMALL_THRESHOLD} → fallback query = {q7}")
        raw_ids_7 = _list_message_ids(service, q7)
        mini_7 = _get_messages_minimal(service, [m["id"] for m in raw_ids_7])
        idmap = {m["id"]: m for m in mini_3}
        idmap.update({m["id"]: m for m in mini_7})
        filtered = [m for m in idmap.values() if int(m.get("internalDate", 0)) >= since_ms]

    filtered.sort(key=lambda m: int(m.get("internalDate", 0)), reverse=True)
    full_msgs = _get_messages_full(service, [m["id"] for m in filtered])
    debug_print(f"gmail.superset.final_pool_size (pre-OTP/thread-dedup) = {len(full_msgs)}; since_unix={since_unix}")

    items = []
    for msg in full_msgs:
        headers = _extract_headers(msg)
        sender_hdr = _get_header(headers, "from")
        display_name, addr = parseaddr(sender_hdr)
        friendly_from = sender_hdr if display_name else (display_name or addr)

        subject = _get_header(headers, "subject")
        try:
            ts = int(msg.get('internalDate', '0')) // 1000
        except Exception:
            ts = 0
        thread_id = msg.get('threadId')

        full_body = _extract_full_body_text(msg)

        # OTP masking
        otp_like, subject, full_body = classify_and_mask_otp(subject, full_body)
        if otp_like:
            # Avoid leaking subjects (PII) into logs; note only the message id.
            debug_print(f"[OTP] masked codes for id={msg.get('id')}")

        # From domain
        from_domain = (addr.split('@')[-1].lower() if addr else (sender_hdr or "").lower()).strip()
        is_issuer = any(from_domain.endswith(x) for x in ISSUER_DOMAINS)

        # Amounts + best by context
        cands = _parse_amounts_from_text(full_body)
        best = _best_amount_by_context(full_body, cands, is_issuer)
        parsed_cur = best[0] if best else None
        parsed_amt = best[1] if best else None
        amount_span = best[2] if best else None

        # Refined is_txn logic
        is_txn = False
        if best:
            center_near_txn = _window_has(_NEAR_TXN_RE, full_body, amount_span, win=120)
            center_near_neg  = _window_has(_NEAR_NEG_RE, full_body, amount_span, win=80)
            cap_near = _window_has(CAP_WORDS, full_body, amount_span, win=60)
            is_txn = bool(center_near_txn and not center_near_neg and not cap_near)

        billing_cue = _has_billing_cue(full_body, amount_span)

        # Tag for ranking fixes (A)
        is_adv = looks_adv(subject)
        promo_near = promo_like_near_amount(full_body, amount_span)

        small = None
        if parsed_cur and parsed_amt is not None:
            small = (parsed_cur == "SGD" and parsed_amt < TXN_ALERT_MIN)

        snippet = _extract_smart_snippet_v2(f"{subject}\n{full_body}", snippet_len, amount_span)
        is_social = _is_social_notification(from_domain, subject, snippet)
        if is_social:
            snippet = _sanitize_snippet_for_social(snippet)

        is_reply = _is_reply_email(headers, subject)
        booking_text = f"{subject}\n{full_body}"
        booking_cue = _has_booking_cue(booking_text)
        reservation_ref = _has_reservation_reference(booking_text)

        item = {
            'id': msg.get('id'),
            'threadId': thread_id,
            'timestamp': ts,
            'from_domain': from_domain,
            'from_raw': sender_hdr,            # "Name <email>"
            'from_friendly': friendly_from,    # readable
            'subject': subject,
            'snippet': snippet,
            'has_deadline_word': bool(re.search(r'(?i)\b(due|deadline|expire|expiry|renew|invoice|pay by|payment due|statement)\b', (subject + ' ' + full_body))),
            'txn_alert': bool(is_txn),
            'txn_currency': parsed_cur,
            'txn_amount': parsed_amt,
            'txn_small': bool(small),
            'otp_like': bool(otp_like),
            'billing_cue': bool(billing_cue),
            'is_social': bool(is_social),
            'is_reply': bool(is_reply),
            'booking_cue': bool(booking_cue),
            'reservation_ref': bool(reservation_ref),
            # New tags for ranking
            'is_adv': bool(is_adv),
            'promo_like': bool(promo_near),
        }
        items.append(item)

    # Thread de-dup (keep latest; keep older with deadline words)
    items.sort(key=lambda x: x['timestamp'], reverse=True)
    by_thread = {}
    for it in items:
        tid = it['threadId'] or it['id']
        if tid not in by_thread:
            by_thread[tid] = it
        else:
            if it['has_deadline_word'] and not by_thread[tid]['has_deadline_word']:
                by_thread[tid] = it
    deduped = list(by_thread.values())

    debug_print(f"[Pool] after thread de-dup: {len(deduped)}")
    return deduped

# -----------------------------
# Ranking (LLM) with JSON-only selection
# -----------------------------
RANK_SYSTEM = """You rank candidate emails into Top-5 JSON selections with strict ID-only.

Preferences:
- Action > FYI.
- Legal/Gov/Billing outrank others.
- Bills / "payment due" are important.
- Social notifications last.
- When nothing critical, prefer newsletters over promos/marketing.
- Direct replies to your inquiries or reservation/booking questions are important.
- Do not treat plain transaction alerts/transfers as billing unless there is a billing cue or a deadline.

Rules to reduce noise:
- Do NOT treat large amounts as important if txn_alert=false AND (is_adv=true OR promo_like=true).
- If is_adv=true or promo_like=true, down-rank unless there is a clear action or deadline.
- Do not mark category=Billing unless has_deadline_word=true OR billing_cue=true; neutral transaction alerts without billing words should be lower urgency.
- Social/profile notifications (is_social=true) should be last unless they include security/account risk.
- Prefer replies (is_reply=true) and booking_cue=true over newsletters and promos; when reservation_ref=true treat as more actionable than marketing/membership notices.
- reservation_ref=true is only for reservation/confirmation codes (not membership IDs); use it to boost genuine booking threads.
- In your "why", avoid citing "large amount mentioned" unless txn_alert=true.

Return ONLY a compact JSON array of objects:
  {id, type, category, urgency, why, action}
Only choose from provided IDs; if fewer candidates exist, return fewer.
"""

RANK_USER_TMPL = """You are given a list of emails with fields:
id, subject, from_domain, snippet, txn_alert, txn_currency, txn_amount, txn_small, has_deadline_word, billing_cue, is_social, is_reply, booking_cue, reservation_ref, is_adv, promo_like.
Choose Top-5 strictly from these IDs. Prefer actionability and legal/gov/billing. Down-rank promos/marketing (is_adv/promo_like) when non-transactional.
"""

def llm_rank(items: List[dict]) -> Tuple[List[dict], str]:
    if not items:
        return [], MODEL_CANDIDATES[0]

    if client is None:
        raise RuntimeError("Missing OPENAI_API_KEY; cannot rank emails without an API key.")

    model_idx = 0
    backoff = BASE_BACKOFF
    last_err = None

    feed = [{
        "id": it["id"],
        "subject": it["subject"],
        "from_domain": it["from_domain"],
        "snippet": it["snippet"],
        "txn_alert": bool(it["txn_alert"]),
        "txn_currency": it["txn_currency"],
        "txn_amount": it["txn_amount"],
        "txn_small": bool(it["txn_small"]),
        "has_deadline_word": bool(it["has_deadline_word"]),
        "billing_cue": bool(it.get("billing_cue")),
        "is_social": bool(it.get("is_social")),
        "is_reply": bool(it.get("is_reply")),
        "booking_cue": bool(it.get("booking_cue")),
        "reservation_ref": bool(it.get("reservation_ref")),
        "is_adv": bool(it.get("is_adv")),
        "promo_like": bool(it.get("promo_like")),
    } for it in items]

    user_msg = RANK_USER_TMPL + "\n\n" + json.dumps(feed, ensure_ascii=False)
    sys_prompt = RANK_SYSTEM + "\n\nReturn ONLY a compact JSON array. Do not add commentary."

    while model_idx < len(MODEL_CANDIDATES):
        model = MODEL_CANDIDATES[model_idx]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            txt = resp.choices[0].message.content.strip()
            top = json.loads(txt)

            allowed = {it["id"] for it in items}
            out = []
            for row in top:
                if isinstance(row, dict) and row.get("id") in allowed:
                    # Normalize all display fields to text (avoid html.escape crashes)
                    t_type = _as_text(row.get("type"))
                    t_cat  = _as_text(row.get("category"))
                    t_urg  = _as_text(row.get("urgency"))
                    t_why  = _as_text(row.get("why"))
                    t_act  = _as_text(row.get("action"))
                    # Debug if coercion happened
                    if DEBUG:
                        for k, v in (("type", row.get("type")), ("category", row.get("category")),
                                     ("urgency", row.get("urgency")), ("why", row.get("why")), ("action", row.get("action"))):
                            if v is not None and not isinstance(v, str):
                                debug_print(f"[Rank] non-string {k}; coerced to empty. value={v!r}")
                    out.append({
                        "id": row["id"],
                        "type": t_type,
                        "category": t_cat,
                        "urgency": t_urg,
                        "why": t_why,
                        "action": t_act,
                    })
                if len(out) >= 5:
                    break
            return out, model
        except Exception as e:
            last_err = e
            debug_print(f"[Rank:{model}] error: {e}")
            model_idx += 1
            time.sleep(backoff)
            backoff *= 2.0

    debug_print(f"[Rank] total failure: {last_err}")
    return [], MODEL_CANDIDATES[min(model_idx, len(MODEL_CANDIDATES)-1)]

# -----------------------------
# Rendering — "old look"
# -----------------------------
def _badge(label: str, bg: str, color: str = "#111"):
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;background:{bg};color:{color};font-size:12px;font-weight:600'>{html.escape(label)}</span>"

def _titlecase_or(raw: Optional[str], fallback: str) -> str:
    s = (raw or "").strip()
    return s[:1].upper() + s[1:] if s else fallback

def render_digest(items: List[dict], top: List[dict], window_start: int, window_end: int, considered_count: int, used_model: str) -> Tuple[str,str]:
    ws = datetime.datetime.fromtimestamp(window_start, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    we = datetime.datetime.fromtimestamp(window_end, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")

    by_id = {it["id"]: it for it in items}

    # Plain text
    plain_lines = [
        f"# AI Email Digest ({datetime.datetime.fromtimestamp(window_end, tz=DISPLAY_TZ).strftime('%Y-%m-%d')})",
        f"Model: {used_model}",
        f"Considered: {considered_count} emails · Window: {ws} → {we}",
        "",
        f"# Today's Top {len(top)} Emails",
    ]
    if not top:
        plain_lines.append("_No eligible emails._")
    else:
        for i, row in enumerate(top, 1):
            it = by_id.get(row["id"], {})
            subj = it.get("subject", "(no subject)")
            frm = it.get("from_raw") or it.get("from_friendly") or it.get("from_domain")
            why = _as_text(row.get("why"))
            act = _as_text(row.get("action"))
            urg = _as_text(row.get("urgency")) or "Low"
            cat = _as_text(row.get("category")) or "Other"
            snip = (it.get("snippet") or "").strip()
            plain_lines.append(f"{i}. {subj} — {frm}")
            plain_lines.append(f"   - Category: {cat} | Urgency: {urg}")
            if why: plain_lines.append(f"   - Why: {why}")
            if act: plain_lines.append(f"   - Action: {act}")
            if snip: plain_lines.append(f"   - Snippet: {snip}")
            plain_lines.append("")
    # Appendix (plain)
    plain_lines.append(f"\n---\n## Appendix — All considered ({len(items)})")
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        plain_lines.append(f"- [{sg_time(it['timestamp'])}] {it.get('from_raw','?')} — {it.get('subject','(no subject)')}")
    plain_txt = "\n".join(plain_lines)

    # HTML
    title = f"AI Email Digest ({datetime.datetime.fromtimestamp(window_end, tz=DISPLAY_TZ).strftime('%Y-%m-%d')})"
    header_html = (
        f"<h2 style='margin:0 0 4px 0;font-size:20px'>{html.escape(title)}</h2>"
        f"<div style='color:#6b7280;font-size:12px;margin-bottom:12px'>"
        f"Model: <strong>{html.escape(used_model)}</strong> · "
        f"Considered: <strong>{considered_count}</strong> emails · "
        f"Window: {html.escape(ws)} → {html.escape(we)}</div>"
    )

    items_html = []
    if not top:
        items_html.append("<p><em>No eligible emails.</em></p>")
    else:
        for i, row in enumerate(top, 1):
            it = by_id.get(row["id"], {})
            subj = html.escape(it.get("subject", "(no subject)"))
            frm = html.escape(it.get("from_raw") or it.get("from_friendly") or it.get("from_domain") or "")
            dom = html.escape(it.get("from_domain", "") or "")
            domain_link = f"<a href='https://{dom}' style='color:#2563eb;text-decoration:none'>{dom}</a>" if dom else ""
            snip = html.escape((it.get("snippet") or "").strip())
            why = html.escape(_as_text(row.get("why")))
            act = html.escape(_as_text(row.get("action")))
            # Normalize badge labels closer to old look
            cat = _titlecase_or(_as_text(row.get("category")), "Other")
            urg = _titlecase_or(_as_text(row.get("urgency")), "Low")
            typ = _titlecase_or(_as_text(row.get("type")), "For Information Only")
            # badge colors (soft)
            type_bg = "#dbeafe" if typ == "For Information Only" else "#fde68a"
            cat_bg_map = {"Legal": "#fecaca", "Government": "#fde68a", "Billing" : "#bbf7d0", "Billing/payment": "#bbf7d0", "Other": "#e5e7eb", "Transaction": "#bbf7d0", "Security": "#e9d5ff"}
            cat_bg = cat_bg_map.get(cat, "#e5e7eb")
            urg_bg_map = {"High": "#fecaca", "Medium": "#fde68a", "Low": "#d1fae5"}
            urg_bg = urg_bg_map.get(urg, "#e5e7eb")

            txn_chip = ""
            if it.get("txn_alert"):
                amt = it.get("txn_amount")
                cur = it.get("txn_currency") or "SGD"
                if isinstance(amt, (int, float)):
                    big = amt >= TXN_HIGH_ALERT_MIN
                    txn_chip = _badge(f"Txn {cur} {amt:.2f}", "#bbf7d0" if big else "#e5e7eb")
                else:
                    txn_chip = _badge("Txn alert", "#e5e7eb")

            items_html.append(f"""
              <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:10px 0;background:#ffffff">
                <div style="font-weight:700;font-size:15px;line-height:1.3;margin-bottom:4px">{i}. {subj}</div>
                <div style="color:#6b7280;font-size:13px;margin-bottom:4px">{domain_link}</div>
                <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{frm}</div>
                <div style="margin-bottom:8px">
                  {_badge(typ, type_bg)}
                  <span style="display:inline-block;width:6px"></span>
                  {_badge(cat, cat_bg)}
                  <span style="display:inline-block;width:6px"></span>
                  {_badge(f"Urgency: {urg}", urg_bg)}
                  {'<span style="display:inline-block;width:6px"></span>' + txn_chip if txn_chip else ''}
                </div>
                {('<div style="font-size:13px;margin-bottom:6px"><strong>Why:</strong> ' + why + '</div>') if why else ''}
                {('<div style="font-size:13px;margin-bottom:6px"><strong>Action:</strong> ' + act + '</div>') if act else ''}
                {('<div style="font-size:12px;color:#6b7280"><strong>Snippet:</strong> ' + snip + '</div>') if snip else ''}
              </div>
            """)

    appendix_rows = []
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        t = html.escape(sg_time(it["timestamp"]))
        f = html.escape(it.get("from_raw") or it.get("from_friendly") or "?")
        s = html.escape(it.get("subject") or "(no subject)")
        appendix_rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;color:#6b7280;font-size:12px;border-bottom:1px solid #f0f0f0'>{t}</td>"
            f"<td style='padding:6px 8px;font-size:12px;border-bottom:1px solid #f0f0f0'>{f}</td>"
            f"<td style='padding:6px 8px;font-size:12px;border-bottom:1px solid #f0f0f0'>{s}</td>"
            "</tr>"
        )
    appendix_html = (
        "<div style='margin-top:14px'>"
        f"<h3 style='margin:0 0 6px 0;font-size:14px'>Appendix — All considered ({len(items)})</h3>"
        "<table style='border-collapse:collapse;width:100%;margin-top:8px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Time</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>From</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Subject</th>"
        "</tr></thead><tbody>"
        + "".join(appendix_rows) +
        "</tbody></table></div>"
    )

    html_body = (
        "<html><body>"
        "<div style='font-family:Inter,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:16px 12px;background:#f8fafc'>"
        + header_html + "".join(items_html) + appendix_html +
        "<div style='color:#9ca3af;font-size:11px;margin-top:12px'>— Generated by your Email Digest Assistant</div>"
        "</div></body></html>"
    )

    return plain_txt, html_body

# -----------------------------
# Email sending (587 STARTTLS or 465 SSL)
# -----------------------------
def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v

def _smtp_connect():
    host = _require("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = _require("SMTP_USER")
    pwd  = _require("SMTP_PASS")

    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=20)
        server.login(user, pwd)
        return server

    server = smtplib.SMTP(host, port, timeout=20)  # STARTTLS
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(user, pwd)
    return server

def send_email(subject: str, plain: str, html_body: str):
    from_addr = _require("MAIL_FROM")
    to_field  = _require("MAIL_TO")
    to_addrs = [x.strip() for x in to_field.split(",") if x.strip()]

    msg = MIMEMultipart('alternative')
    msg['From'] = from_addr
    msg['To'] = ", ".join(to_addrs)
    msg['Subject'] = subject
    msg.attach(MIMEText(plain, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    with _smtp_connect() as server:
        server.sendmail(from_addr, to_addrs, msg.as_string())

def send_digest_email(subject: str, plain: str, html_body: str):
    send_email(subject, plain, html_body)

def send_error_email(subject: str, body: str):
    try:
        send_email(subject, body, f"<pre>{html.escape(body)}</pre>")
    except Exception:
        print("Failed to send error email:", traceback.format_exc())

# -----------------------------
# Watermark handling
# -----------------------------
def _read_last_sent_timestamp() -> Optional[int]:
    try:
        with open("last_sent_ts.txt", "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def _write_last_sent_timestamp(ts: int):
    with open("last_sent_ts.txt", "w") as f:
        f.write(str(ts))

# -----------------------------
# Main
# -----------------------------
def main():
    if DISABLE_DIGEST:
        print("DISABLE_DIGEST=1; skipping.")
        return

    try:
        service = _gmail_service()

        now = int(time.time())
        last_sent = _read_last_sent_timestamp() or (now - 86400)
        since_unix = max(0, last_sent - SINCE_BUFFER_HOURS * 3600)

        print("UTC now:", datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M"))
        print(f"Window (display {DISPLAY_TZ_NAME}): {sg_time(since_unix)} → {sg_time(now)}")

        items = fetch_all_since_v2(service, since_unix, snippet_len=SNIPPET_LEN)
        considered_count = len(items)

        top, used_model = llm_rank(items)

        plain, html_body = render_digest(items, top, since_unix, now, considered_count, used_model)
        subject = f"AI Email Digest ({datetime.datetime.fromtimestamp(now, tz=DISPLAY_TZ).strftime('%Y-%m-%d')})"

        send_digest_email(subject, plain, html_body)
        print("Digest sent.")
        _write_last_sent_timestamp(now)

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
        raise  # fail the workflow visibly

if __name__ == "__main__":
    main()
