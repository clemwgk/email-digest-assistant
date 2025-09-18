# main.py — Gmail AI Email Digest (superset query + local filter + OTP masking + number/card normalization)
# + Restored “old style” Top cards with Subject/Snippet and Appendix table, and show actual model used.

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
OAUTH_B64 = os.getenv("GOOGLE_OAUTH_JSON_B64", "")
TOKEN_B64 = os.getenv("GOOGLE_TOKEN_JSON_B64", "")
ISSUER_DOMAINS = [d.strip() for d in os.getenv("ISSUER_DOMAINS", "").split(",") if d.strip()]
STRICT_INBOX = os.getenv("STRICT_INBOX", "0").strip() == "1"
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Asia/Singapore")
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
DISABLE_DIGEST = os.getenv("DISABLE_DIGEST", "0").strip() == "1"

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
MAX_RETRIES = 4
BASE_BACKOFF = 2.0  # seconds

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Triaging tunables
SNIPPET_LEN = int(os.getenv("SNIPPET_LEN", "500"))            # how much context we send per email
TXN_ALERT_MIN = float(os.getenv("TXN_ALERT_MIN", "100"))      # mark "small" transactions below this
TXN_HIGH_ALERT_MIN = float(os.getenv("TXN_HIGH_ALERT_MIN", "100"))
SINCE_BUFFER_HOURS = int(os.getenv("SINCE_BUFFER_HOURS", "6"))

# -----------------------------
# Utility
# -----------------------------
def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def sg_time(ts: int) -> str:
    tz = ZoneInfo(DISPLAY_TZ)
    return datetime.datetime.fromtimestamp(ts, tz=tz).strftime("%Y-%m-%d %H:%M")

def _extract_headers(msg) -> dict:
    headers = {}
    for h in msg.get('payload', {}).get('headers', []):
        headers[h.get('name', '').lower()] = h.get('value', '')
    return headers

def _get_header(headers: dict, name: str, default: str = "") -> str:
    return headers.get(name.lower(), default)

def _gmail_service():
    # Reconstitute OAuth files from base64 blobs
    creds = None
    if not OAUTH_B64:
        raise RuntimeError("Missing GOOGLE_OAUTH_JSON_B64 secret")
    oauth_path = "credentials.json"
    with open(oauth_path, "wb") as f:
        f.write(base64.b64decode(OAUTH_B64))

    token_path = "token.json"
    if TOKEN_B64:
        with open(token_path, "wb") as f:
            f.write(base64.b64decode(TOKEN_B64))

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                flow = InstalledAppFlow.from_client_secrets_file(oauth_path, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(oauth_path, SCOPES)
            creds = flow.run_local_server(port=0)

    with open(token_path, "w") as f:
        f.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

# -----------------------------
# HTML/text extraction + normalization
# -----------------------------
def _extract_full_body_text(msg) -> str:
    def decode_part(part):
        data = part.get('body', {}).get('data')
        if not data:
            return ""
        try:
            return base64.urlsafe_b64decode(data.encode('UTF-8')).decode('utf-8', errors='replace')
        except Exception:
            return ""
    payload = msg.get('payload', {})
    mimeType = payload.get("mimeType", "")
    if mimeType == "text/plain":
        return _post_process_text(decode_part(payload))
    if mimeType == "text/html":
        return _post_process_text(_html_to_text(decode_part(payload)))
    if mimeType.startswith("multipart/"):
        txt = []
        parts = payload.get("parts", []) or []
        for p in parts:
            mt = p.get("mimeType", "")
            if mt == "text/plain":
                txt.append(decode_part(p))
            elif mt == "text/html":
                txt.append(_html_to_text(decode_part(p)))
            elif mt.startswith("multipart/"):
                for pp in p.get("parts", []) or []:
                    mtt = pp.get("mimeType", "")
                    if mtt == "text/plain":
                        txt.append(decode_part(pp))
                    elif mtt == "text/html":
                        txt.append(_html_to_text(decode_part(pp)))
        return _post_process_text("\n".join([t for t in txt if t]))
    return _post_process_text("")

def _html_to_text(s: str) -> str:
    s = re.sub(r'(?is)<script.*?>.*?</script>', ' ', s)
    s = re.sub(r'(?is)<style.*?>.*?</style>', ' ', s)
    s = re.sub(r'(?is)<br[^>]*>', '\n', s)
    s = re.sub(r'(?is)</p>', '\n', s)
    s = re.sub(r'(?is)<.*?>', ' ', s)
    s = html.unescape(s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _post_process_text(s: str) -> str:
    # 1) Fix broken decimals across line breaks: "25.\n76" -> "25.76"
    s = re.sub(r'(?m)(\d+)\.\s*\n\s*(\d{2})', r'\1.\2', s)
    # 2) Mask card last-4 so they aren't parsed as money
    # common forms: "ending 1234", "ending in 1234", "last 4 1234", "****1234", "xxxx 1234"
    s = re.sub(r'(?i)\b(ending(?: in)?|last(?:\s*4|\s*four)\s*(?:digits?)?)\s*(\d{4})\b', r'\1 ‹last4›', s)
    s = re.sub(r'(?i)(?:\*{4}|x{4})[\s\-]?(\d{4})\b', r'**** ‹last4›', s)
    return s

def _domain_is_issuer(from_domain: str) -> bool:
    d = (from_domain or "").lower()
    return any(d.endswith(x) for x in ISSUER_DOMAINS) if ISSUER_DOMAINS else False

# =========================================================
# OTP/2FA detection (proximity + disclaimer-aware) — mask, don't drop
# =========================================================
OTP_KEYWORDS_RE = re.compile(r'(?i)\b(otp|one[- ]?time password|verification code|2fa|login code|security code)\b')
OTP_CODE_RE = re.compile(r'(?i)(?<![A-Z0-9])([A-Z0-9]{4,8})(?![A-Z0-9])')
OTP_NEG_CUE_RE = re.compile(r'(?i)(do not share|never (?:ask|request)|we will never (?:ask|request)|stay secure|for your security|phishing|scam)')

def classify_and_mask_otp(subject: str, body: str, window: int = 80) -> Tuple[bool, str, str]:
    """
    Return (otp_like, masked_subject, masked_body).
    'otp_like' True only if a keyword occurs within ±window of a code and no strong disclaimer cue is nearby.
    When otp_like=True, mask code-like tokens in both subject and body.
    """
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

# =========================================================
# Amount parsing & context
# =========================================================
AMOUNT_CURR_RE = re.compile(r'(?i)\b(SGD|USD|EUR|GBP|\$)\s?([0-9]{1,3}(?:[,\s][0-9]{3})*|[0-9]+)(\.[0-9]{2})?\b')
AMOUNT_SIMPLE_RE = re.compile(r'(?<![0-9])([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(\.[0-9]{2})\b')
_NEAR_TXN_RE = re.compile(r'(?i)\b(transaction|charged|purchase|merchant|withdrawal|transfer|authorized|unauthorized|declined|failed)\b')
_NEAR_NEG_RE = re.compile(r"(?i)\b(coverage|limit|credit limit|sum insured|policy|premium|price|valued at|apr|interest|markets|news|statement|advice|balance)\b")

def _money_from_groups(cur: Optional[str], num: Optional[str], cents: Optional[str]) -> Optional[Tuple[str,float]]:
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

def _best_amount_by_context(text_lower: str, cands: list[Tuple[str,float,Tuple[int,int]]], is_issuer: bool) -> Optional[Tuple[str,float,Tuple[int,int]]]:
    """
    Score amounts by nearby context:
      +3 if within ~80 chars of a txn verb
      -3 if within ~60 chars of negative contexts
      +1 if issuer domain
      + tiny tie-break for larger amount
    """
    if not cands:
        return None

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
        score = 0.0
        if d_txn <= 80: score += 3
        if d_neg <= 60: score -= 3
        if is_issuer: score += 1
        score += min(amt, 1000) / 100000.0
        if score > best_score:
            best_score = score
            best = (cur, amt, (a,b))
    return best

# -----------------------------
# Snippet extraction
# -----------------------------
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _extract_window(text: str, center: int, radius: int = 220) -> str:
    a = max(0, center - radius)
    b = min(len(text), center + radius)
    seg = text[a:b]
    # Clean up to sentence boundaries if we can
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
        if re.search(r'(?i)\b(due|deadline|expire|expiry|renew|invoice|pay by|payment due|statement)\b', s): score += 1
        if AMOUNT_CURR_RE.search(s): score += 1
        if score: scored.append((score, s.strip()))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return re.sub(r'\s+', ' ', scored[0][1])[:max_len]
    return re.sub(r'\s+', ' ', full_text)[:max_len]

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
    return f'newer_than:{days}d {scope} -in:spam -in:trash -subject:"AI Email Digest —"'

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
    Build the full pool since the last run.
      - Superset Gmail query: newer_than:3d (or 7d fallback) + local internalDate filter >= since_unix
      - Include archived mail by default (STRICT_INBOX=0); exclude spam/trash; exclude past digests
      - Mask OTP-like codes (do NOT drop)
      - Keep latest per thread; keep older items only if they contain deadline words
      - Parse amounts → choose best by context → amount-centered snippet → txn detection
    """
    strict = STRICT_INBOX
    q3 = _build_superset_query(3, strict)
    debug_print(f"gmail.superset.query(primary) = {q3}")
    raw_ids_3 = _list_message_ids(service, q3)
    debug_print(f"gmail.superset.raw_ids(primary) = {len(raw_ids_3)}")

    # local filter by internalDate >= since_unix (note: Gmail internalDate is ms)
    since_ms = int(since_unix) * 1000
    mini_3 = _get_messages_minimal(service, [m["id"] for m in raw_ids_3])
    filtered = [m for m in mini_3 if int(m.get("internalDate", 0)) >= since_ms]

    # Optional fallback if pool suspiciously small
    SMALL_THRESHOLD = 5
    if len(filtered) < SMALL_THRESHOLD:
        q7 = _build_superset_query(7, strict)
        debug_print(f"gmail.superset.pool_small={len(filtered)}<={SMALL_THRESHOLD} → fallback query = {q7}")
        raw_ids_7 = _list_message_ids(service, q7)
        mini_7 = _get_messages_minimal(service, [m["id"] for m in raw_ids_7])
        idmap = {m["id"]: m for m in mini_3}
        idmap.update({m["id"]: m for m in mini_7})
        filtered = [m for m in idmap.values() if int(m.get("internalDate", 0)) >= since_ms]

    # newest first
    filtered.sort(key=lambda m: int(m.get("internalDate", 0)), reverse=True)
    full_msgs = _get_messages_full(service, [m["id"] for m in filtered])
    debug_print(f"gmail.superset.final_pool_size (pre-OTP/thread-dedup) = {len(full_msgs)}; since_unix={since_unix}")

    # Convert to items with OTP masking, parsing, etc.
    items = []
    for msg in full_msgs:
        headers = _extract_headers(msg)
        sender = _get_header(headers, "from")
        subject = _get_header(headers, "subject")
        try:
            ts = int(msg.get('internalDate', '0')) // 1000
        except Exception:
            ts = 0
        thread_id = msg.get('threadId')

        full_body = _extract_full_body_text(msg)

        # OTP / verification (mask, don't drop)
        otp_like, subject, full_body = classify_and_mask_otp(subject, full_body)
        if otp_like:
            debug_print(f"[OTP] masked codes for id={msg.get('id')} subj={subject!r}")

        # From domain (best-effort)
        from_domain = sender
        if '@' in sender:
            try:
                from_domain = sender.split('@')[-1].split('>')[0].strip().lower()
            except Exception:
                pass
        is_issuer = _domain_is_issuer(from_domain)

        # Amount parsing + context
        cands = _parse_amounts_from_text(full_body)
        best = _best_amount_by_context(full_body.lower(), cands, is_issuer)
        parsed_cur = best[0] if best else None
        parsed_amt = best[1] if best else None
        amount_span = best[2] if best else None

        # Txn detection
        is_txn = False
        if parsed_amt is not None:
            if _NEAR_TXN_RE.search(full_body):
                is_txn = True
            if _NEAR_NEG_RE.search(full_body):
                is_txn = False

        cur = parsed_cur
        amt = parsed_amt
        small = None
        if cur and amt is not None:
            small = (cur == "SGD" and amt < TXN_ALERT_MIN)

        snippet = _extract_smart_snippet_v2(f"{subject}\n{full_body}", snippet_len, amount_span)

        item = {
            'id': msg.get('id'),
            'threadId': thread_id,
            'timestamp': ts,
            'from_domain': from_domain,
            'from_raw': sender,
            'subject': subject,
            'snippet': snippet,
            'has_deadline_word': bool(re.search(r'(?i)\b(due|deadline|expire|expiry|renew|invoice|pay by|payment due|statement)\b', (subject + ' ' + full_body).lower())),
            'txn_alert': is_txn,
            'txn_currency': cur,
            'txn_amount': amt,
            'txn_small': bool(small),
            'otp_like': bool(otp_like),
        }
        items.append(item)

    # Thread de-dup: keep latest per thread; include older only if they contain deadline words
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
- Action > FYI; legal/gov/billing outrank others
- Bills / "payment due" are important
- Social notifications last
- When nothing critical, newsletters > promos
- Transactions ≥100 SGD outrank smaller unless fraud cues
Return an array of objects: {id, type, category, urgency, why, action}.
Only choose from provided IDs; if fewer candidates exist, return fewer.
"""

RANK_USER_TMPL = """You are given a list of emails with fields:
id, subject, from_domain, snippet, txn_alert, txn_currency, txn_amount, txn_small, has_deadline_word.
Choose Top-5 strictly from these IDs. Prefer actionability and legal/gov/billing.
"""

def choose_model():
    # Keep simple; fallback handled in llm_rank
    return MODEL_CANDIDATES[0]

def llm_rank(items: List[dict]) -> Tuple[List[dict], str]:
    if not items:
        return [], MODEL_CANDIDATES[0]

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
                    out.append({
                        "id": row["id"],
                        "type": row.get("type"),
                        "category": row.get("category"),
                        "urgency": row.get("urgency"),
                        "why": row.get("why"),
                        "action": row.get("action"),
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
# Email rendering & sending
# -----------------------------
def render_digest(items: List[dict], top: List[dict], window_start: int, window_end: int, considered_count: int, used_model: str) -> Tuple[str,str]:
    tz = ZoneInfo(DISPLAY_TZ)
    ws = datetime.datetime.fromtimestamp(window_start, tz=tz).strftime("%Y-%m-%d %H:%M")
    we = datetime.datetime.fromtimestamp(window_end, tz=tz).strftime("%Y-%m-%d %H:%M")

    # map for quick lookup by id
    by_id = {it["id"]: it for it in items}

    title = f"Gmail - AI Email Digest — {ws.split(' ')[0]} (model: {used_model}; considered {considered_count})"

    # ---------- PLAIN TEXT ----------
    plain_lines = [
        f"Window: {ws} → {we}",
        f"Considered: {considered_count}",
        "",
        "Top:",
    ]
    if not top:
        plain_lines.append("(none)")
    else:
        for i, row in enumerate(top, 1):
            it = by_id.get(row["id"], {})
            subj = it.get("subject", "(no subject)")
            snippet = (it.get("snippet") or "").strip()
            cat = row.get("category")
            urg = row.get("urgency")
            why = row.get("why")
            act = row.get("action")
            plain_lines.append(f"{i}. {subj}")
            if cat or urg:
                plain_lines.append(f"   [{cat or 'other'} | {urg or 'n/a'}]")
            if why: plain_lines.append(f"   Why: {why}")
            if act: plain_lines.append(f"   Action: {act}")
            if snippet: plain_lines.append(f"   Snippet: {snippet}")
            plain_lines.append("")
    plain_lines.append("Appendix:")
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        plain_lines.append(f"- {sg_time(it['timestamp'])} — {it['from_raw']} — {it['subject']}")
    plain_txt = "\n".join(plain_lines)

    def esc(s): return html.escape(s or "")
    # ---------- HTML ----------
    html_parts = [
        f"<h2>{esc(title)}</h2>",
        f"<p><b>Window:</b> {esc(ws)} → {esc(we)} &nbsp; | &nbsp; <b>Considered:</b> {considered_count}</p>",
        "<h3>Top</h3>",
    ]
    if not top:
        html_parts.append("<p>(none)</p>")
    else:
        for i, row in enumerate(top, 1):
            it = by_id.get(row["id"], {})
            subj = esc(it.get("subject", "(no subject)"))
            from_d = esc(it.get("from_domain", "") or it.get("from_raw", ""))
            snippet = esc((it.get("snippet") or "").strip())
            cat = esc(row.get("category") or "other")
            urg = esc(row.get("urgency") or "n/a")
            why = esc(row.get("why") or "")
            act = esc(row.get("action") or "")
            html_parts.append(
                f"""
                <div style="border:1px solid #e7e7e7;border-radius:12px;padding:12px;margin:12px 0;">
                  <div style="font-weight:600;margin-bottom:4px;">{i}. {subj}</div>
                  <div style="color:#666;margin-bottom:8px;">{from_d}</div>
                  <div style="margin:6px 0 10px 0;">
                    <span style="display:inline-block;background:#eef;border:1px solid #dde;border-radius:999px;padding:2px 8px;margin-right:6px;">{cat}</span>
                    <span style="display:inline-block;background:#efe;border:1px solid #ded;border-radius:999px;padding:2px 8px;">Urgency: {urg}</span>
                  </div>
                  {"<div><b>Why:</b> " + why + "</div>" if why else ""}
                  {"<div><b>Action:</b> " + act + "</div>" if act else ""}
                  {"<div><b>Snippet:</b> " + snippet + "</div>" if snippet else ""}
                </div>
                """
            )

    # Appendix table
    html_parts.append("<h3>Appendix — All considered</h3>")
    html_parts.append("""
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th align="left" style="border-bottom:1px solid #ddd;padding:8px;">Time</th>
          <th align="left" style="border-bottom:1px solid #ddd;padding:8px;">From</th>
          <th align="left" style="border-bottom:1px solid #ddd;padding:8px;">Subject</th>
        </tr>
      </thead>
      <tbody>
    """)
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        html_parts.append(
            f"<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;'>{esc(sg_time(it['timestamp']))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;'>{esc(it['from_raw'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #f0f0f0;'>{esc(it['subject'])}</td>"
            f"</tr>"
        )
    html_parts.append("</tbody></table>")

    html_body = "\n".join(html_parts)
    return plain_txt, html_body

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

    # default: STARTTLS (587)
    server = smtplib.SMTP(host, port, timeout=20)
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
# Main flow
# -----------------------------
def main():
    if DISABLE_DIGEST:
        print("DISABLE_DIGEST=1; skipping.")
        return

    try:
        service = _gmail_service()

        now = int(time.time())
        last_sent = _read_last_sent_timestamp() or (now - 86400)
        # watermark with buffer
        since_unix = last_sent - SINCE_BUFFER_HOURS * 3600
        if since_unix < 0:
            since_unix = 0

        tz = ZoneInfo(DISPLAY_TZ)
        print("UTC now:", datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M"))
        print(f"Window (display {DISPLAY_TZ}): {sg_time(since_unix)} → {sg_time(now)}")

        items = fetch_all_since_v2(service, since_unix, snippet_len=SNIPPET_LEN)
        considered_count = len(items)

        top, used_model = llm_rank(items)

        plain, html_body = render_digest(items, top, since_unix, now, considered_count, used_model)
        subject = f"Gmail - AI Email Digest — {datetime.datetime.fromtimestamp(now, tz=tz).strftime('%Y-%m-%d')} (model: {used_model}; considered {considered_count})"

        send_digest_email(subject, plain, html_body)
        print("Digest sent.")
        _write_last_sent_timestamp(now)

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
        raise  # fail the workflow visibly

if __name__ == "__main__":
    main()
