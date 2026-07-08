# main.py — Gmail AI Email Digest (rubric-judge redesign)

import os
import time
import json
import html
import re
import base64
import smtplib
import traceback
import datetime
from typing import List, Tuple, Optional, Dict, NamedTuple

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
from google import genai

# -----------------------------
# Env & constants
# -----------------------------
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Accept either GOOGLE_* or GMAIL_*; else fall back to on-disk files
OAUTH_B64 = os.getenv("GOOGLE_OAUTH_JSON_B64") or os.getenv("GMAIL_CREDENTIALS_JSON_B64") or ""
TOKEN_B64 = os.getenv("GOOGLE_TOKEN_JSON_B64") or os.getenv("GMAIL_TOKEN_JSON_B64") or ""

STRICT_INBOX = os.getenv("STRICT_INBOX", "0").strip() == "1"
DISPLAY_TZ_NAME = os.getenv("DISPLAY_TZ", "Asia/Singapore")
DISPLAY_TZ = ZoneInfo(DISPLAY_TZ_NAME)
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
DISABLE_DIGEST = os.getenv("DISABLE_DIGEST", "0").strip() == "1"

# Dry run: run the full pipeline but print the rendered digest + decisions to the
# log instead of sending SMTP / writing the watermark. Wired to the workflow's
# `dry_run` dispatch input (Phase 6).
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"

# Cross-digest dedup: by default suppressed IDs are removed from KNOW candidacy
# only; ACT items may resurface (Clement's Gate 2 decision — he sometimes misses
# a digest and the short pool window bounds repetition). Flip to "1" for uniform
# suppression. See design notes / DEDUP_SUPPRESS_ACT.
DEDUP_SUPPRESS_ACT = os.getenv("DEDUP_SUPPRESS_ACT", "0").strip() == "1"

# SMTP config (auto-detect 465 vs 587)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = [x.strip() for x in os.getenv("MAIL_TO", "").split(",") if x.strip()]

# LLM Provider config: "gemini" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# OpenAI model fallback order
OPENAI_MODEL_CANDIDATES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

# Gemini model (single model, no fallback needed for free tier)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

BASE_BACKOFF = 2.0  # seconds

# OpenAI client (initialized if needed)
openai_client = None
if LLM_PROVIDER == "openai" or os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gemini client (initialized if needed)
gemini_client = None
if LLM_PROVIDER == "gemini" and os.getenv("GEMINI_API_KEY"):
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Tunables
LLM_BODY_LEN = int(os.getenv("LLM_BODY_LEN", "2000"))   # masked chars sent to the judge
PREVIEW_LEN = int(os.getenv("PREVIEW_LEN", "200"))      # masked chars shown in the card
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000"))
SINCE_BUFFER_HOURS = int(os.getenv("SINCE_BUFFER_HOURS", "6"))
RUBRIC_PATH = os.getenv("RUBRIC_PATH", "rubric.md")

# Selection budget
TOTAL_BUDGET = 7
KNOW_CAP = 3

# -----------------------------
# Errors
# -----------------------------
class RubricError(RuntimeError):
    """Raised when rubric.md is missing or empty — never run rubric-less."""


class LLMExhaustedError(RuntimeError):
    """Raised when every configured LLM provider failed to produce a parseable
    judgment. Propagates to main() → error email + failed run. A genuinely empty
    (but successfully parsed) judgment is NOT this error."""


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
    s = s.replace("‌", "").replace("​", "")
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
# OTP/2FA detection — narrowed mask (do not drop)
#
# The mask's job is privacy: never transmit a live-looking verification code.
# Relevance (OTP emails are categorically noise) is now the rubric/LLM's job.
# Narrowed vs. the old mask: only tokens that CONTAIN A DIGIT (length 4–8),
# sitting within an OTP-keyword window, are masked — so marketing/transaction
# text with plain uppercase words ("VISA", "REVOLUTION", "CODE") survives.
# -----------------------------
OTP_KEYWORDS_RE = re.compile(r'(?i)\b(otp|one[- ]?time password|verification code|2fa|login code|security code)\b')
# A 4–8 char alphanumeric token, bounded by non-alnum. Digit-bearing check is
# applied in code (see _token_is_code) so only codes — not plain words — match.
OTP_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9])([A-Za-z0-9]{4,8})(?![A-Za-z0-9])')
OTP_NEG_CUE_RE = re.compile(r'(?i)(do not share|never (?:ask|request)|we will never (?:ask|request)|stay secure|for your security|phishing|scam)')

def _token_is_code(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)

def _has_code_token(s: str) -> bool:
    return any(_token_is_code(m.group(1)) for m in OTP_TOKEN_RE.finditer(s or ""))

def _mask_codes(s: str) -> str:
    return OTP_TOKEN_RE.sub(lambda m: "‹code›" if _token_is_code(m.group(1)) else m.group(1), s or "")

def classify_and_mask_otp(subject: str, body: str, window: int = 80) -> Tuple[bool, str, str]:
    text = (subject or "") + "\n" + (body or "")
    otp_like = False
    for m in OTP_KEYWORDS_RE.finditer(text):
        left = max(0, m.start() - window)
        right = min(len(text), m.end() + window)
        window_text = text[left:right]
        if OTP_NEG_CUE_RE.search(window_text):
            continue
        if _has_code_token(window_text):
            otp_like = True
            break
    if not otp_like:
        return False, subject, body

    return True, _mask_codes(subject), _mask_codes(body)

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

# Deadline cue — used only as a thread-dedup tiebreak (keep an older message in a
# thread if it carries a deadline the newest one dropped). NOT fed to the judge.
DEADLINE_RE = re.compile(r'(?i)\b(due|deadline|expire|expiry|renew|invoice|pay by|payment due|statement)\b')

def _item_has_deadline(it: dict) -> bool:
    return bool(DEADLINE_RE.search((it.get('subject', '') or '') + ' ' + (it.get('body_for_llm', '') or '')))

def fetch_all_since_v2(service, since_unix: int):
    """
    Build pool since watermark:
      - Superset Gmail query (3d; 7d fallback) + local filter internalDate >= since_unix
      - Include archived mail by default; exclude spam/trash; exclude past digests; exclude Sent/from:me
      - OTP/code masking (narrowed; do not drop)
      - Thread de-dup (keep latest; keep older w/ deadline words)
      - Per item: masked body_for_llm (~LLM_BODY_LEN) + body_preview (~PREVIEW_LEN of the SAME text)
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
        from_address = (addr or "").strip().lower()
        from_name = (display_name or "").strip() or from_address

        subject = _get_header(headers, "subject")
        try:
            ts = int(msg.get('internalDate', '0')) // 1000
        except Exception:
            ts = 0
        thread_id = msg.get('threadId')

        full_body = _extract_full_body_text(msg)

        # OTP masking (narrowed) — both body_for_llm and body_preview derive from
        # the SAME masked string so the prompt and the card never disagree.
        otp_like, subject, masked_body = classify_and_mask_otp(subject, full_body)
        if otp_like:
            # Avoid leaking subjects (PII) into logs; note only the message id.
            debug_print(f"[OTP] masked codes for id={msg.get('id')}")

        from_domain = (from_address.split('@')[-1] if from_address else (sender_hdr or "").lower()).strip()

        item = {
            'id': msg.get('id'),
            'threadId': thread_id,
            'timestamp': ts,
            'from_name': from_name,
            'from_address': from_address,
            'from_domain': from_domain,
            'from_raw': sender_hdr,             # "Name <email>" — appendix/card display
            'subject': subject,
            'body_for_llm': (masked_body or "")[:LLM_BODY_LEN],
            'body_preview': (masked_body or "")[:PREVIEW_LEN],
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
            if _item_has_deadline(it) and not _item_has_deadline(by_thread[tid]):
                by_thread[tid] = it
    deduped = list(by_thread.values())

    debug_print(f"[Pool] after thread de-dup: {len(deduped)}")
    return deduped

# -----------------------------
# Rubric (Clement-owned judgment file)
# -----------------------------
def load_rubric() -> str:
    """Load rubric.md; missing or empty is a loud failure (never a rubric-less run)."""
    try:
        with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        raise RubricError(f"Rubric file not found at '{RUBRIC_PATH}'. Refusing to run rubric-less.")
    if not content:
        raise RubricError(f"Rubric file '{RUBRIC_PATH}' is empty. Refusing to run rubric-less.")
    return content

# -----------------------------
# LLM judgment layer
# -----------------------------
SYSTEM_PROMPT = """You are the judgment layer of Clement's daily email digest. You receive his personal rubric and a
JSON array of candidate emails from roughly the last day. Decide which emails he must ACT on and
which he would materially want to KNOW about today — and exclude everything else. Returning zero
items is a correct and common outcome: on a quiet day, an empty selection is a better answer than
a padded one. Nothing is included just to fill space.

THE CORE QUESTION, per email:
Does Clement need to DO something about this (ACT)? Would he materially want to see it today even
with nothing to do (KNOW)? Or neither (exclude)?
When an email asks for action, ask WHOSE interest the action serves. "Pay your bill", "your
flight changed", "confirm this login was you" are Clement's obligations — ACT. "Sign up",
"redeem", "don't miss out", "claim your reward" are the sender's wishes — marketing; exclude it
no matter how urgent the wording sounds.

CLEMENT'S RUBRIC (his own definitions — these override your general intuitions):
{RUBRIC_MD}

DECISION PROCEDURE, per email:
1. Work out what the email actually is from its body text, not just its subject.
2. Check the rubric's Noise list first. If it matches a noise category, exclude — the only
   exception is the credit-card-statement override written in the rubric itself.
3. Otherwise test it against ACT, then KNOW. Include only if it clearly fits. When genuinely
   unsure, exclude: a missed marginal item costs less than restored noise.
4. Sender identity is decided at full-address level, not by domain. Only the exact addresses the
   rubric allowlists qualify — e.g. Nate's nateszerotoai@substack.com AND natesnewsletter@substack.com
   are both allowlisted; a different mailbox at substack.com (another author, or a platform no-reply) is not.

TRANSACTION RULE (this replaced a regex that kept failing — be precise):
An amount is a transaction only if the email reports money actually moving out of Clement's
accounts: charged, debited, paid, transferred out, withdrawn. Amounts appearing in marketing
offers, prices, insurance or deposit-coverage statements ("insured up to S$100,000"), credit
limits, or promotional targets are NOT transactions. A genuine outflow of S$100 or more is ACT
(fraud guardrail). Smaller outflows and micro-receipts are excluded.

OUTPUT — return ONLY this JSON object, no commentary:
{"items": [ ... ], "omitted_act_count": 0}
Each item:
  "id"       — copied exactly from the input; never invent or repeat an id
  "tier"     — "ACT" or "KNOW"
  "category" — one of: Billing, Government, Transaction, Travel, Security, Newsletter, Account, Other
  "reason"   — one concrete sentence citing what in the email makes it signal
  "action"   — one short imperative sentence; ACT items only, omit for KNOW
  "amount"   — like "SGD 3103.00"; only when a genuine transaction amount is central, else omit

BUDGET:
At most 7 items total, ordered most-important-first. ACT items take priority; KNOW items fill at
most 3 of the remaining slots. If more than 7 emails genuinely qualify as ACT, include the 7 most
important and set omitted_act_count to the number left out; otherwise omitted_act_count is 0.

FINAL RULES:
- Reasons must be verifiable from the email text. Never cite an amount as significant unless the
  transaction rule was satisfied.
- OTP and verification-code emails are always excluded — their codes expired long before this
  digest is read.
- Do not pad. A short or empty list on a quiet day is the system working correctly."""

# Structured-output schema (attempted first; falls back to plain generation).
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tier": {"type": "string", "enum": ["ACT", "KNOW"]},
                    "category": {"type": "string"},
                    "reason": {"type": "string"},
                    "action": {"type": "string"},
                    "amount": {"type": "string"},
                },
                "required": ["id", "tier", "category", "reason"],
            },
        },
        "omitted_act_count": {"type": "integer"},
    },
    "required": ["items", "omitted_act_count"],
}


class ParsedJudgment(NamedTuple):
    act: List[dict]
    know: List[dict]
    model_omitted: int


def _extract_json_object_text(txt: str) -> str:
    """Extract first balanced JSON object from model text, handling code fences."""
    s = (txt or "").strip().replace("﻿", "")
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return s[start:]


def _load_json_object(raw: str) -> dict:
    attempts = []
    for label, cand in (("raw", raw), ("extracted", _extract_json_object_text(raw))):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            attempts.append(f"{label}:not-an-object({type(obj).__name__})")
        except json.JSONDecodeError as err:
            attempts.append(f"{label}:{err}")
    preview = (raw or "")[:300].replace("\n", "\\n")
    raise ValueError(f"LLM JSON object parse failed ({'; '.join(attempts)}) preview={preview}")


def _parse_llm_response(txt: str, allowed_ids: set) -> ParsedJudgment:
    """Parse the judgment JSON object. Validates ids and tiers, splits into
    ACT/KNOW preserving the model's order. A valid-but-empty result is legitimate.
    Raises ValueError on unparseable / malformed output (caller decides retry)."""
    obj = _load_json_object((txt or "").strip())

    items_field = obj.get("items")
    if not isinstance(items_field, list):
        raise ValueError("LLM response object missing 'items' array")

    model_omitted = obj.get("omitted_act_count", 0)
    try:
        model_omitted = max(0, int(model_omitted))
    except (TypeError, ValueError):
        model_omitted = 0

    act: List[dict] = []
    know: List[dict] = []
    seen = set()
    rejected = 0

    for row in items_field:
        if not isinstance(row, dict):
            rejected += 1
            continue
        rid = row.get("id")
        if rid not in allowed_ids or rid in seen:
            rejected += 1
            continue
        tier = _as_text(row.get("tier")).strip().upper()
        if tier not in ("ACT", "KNOW"):
            rejected += 1
            continue
        out_row = {
            "id": rid,
            "tier": tier,
            "category": _as_text(row.get("category")).strip() or "Other",
            "reason": _as_text(row.get("reason")).strip(),
            "action": _as_text(row.get("action")).strip(),
            "amount": _as_text(row.get("amount")).strip(),
        }
        seen.add(rid)
        (act if tier == "ACT" else know).append(out_row)

    debug_print(f"[Judge] parsed act={len(act)} know={len(know)} rejected={rejected} model_omitted={model_omitted}")
    return ParsedJudgment(act, know, model_omitted)


def select_top(parsed: ParsedJudgment, suppressed_ids: Optional[set] = None,
               suppress_act: bool = False, total: int = TOTAL_BUDGET,
               know_cap: int = KNOW_CAP) -> Tuple[List[dict], int]:
    """Apply cross-digest dedup suppression + selection budget.

    ACT fills first (model's importance order); KNOW gets min(know_cap, total-#ACT)
    of the remaining slots. Suppressed IDs are removed from KNOW candidacy by
    default; ACT only when suppress_act. omitted_act_count counts model-reported
    overflow PLUS any ACT rows truncated by the budget (not dedup-suppressed ones).
    """
    suppressed_ids = suppressed_ids or set()

    act = parsed.act
    if suppress_act and suppressed_ids:
        act = [r for r in act if r["id"] not in suppressed_ids]
    know = [r for r in parsed.know if r["id"] not in suppressed_ids]

    act_selected = act[:total]
    omitted = parsed.model_omitted + max(0, len(act) - len(act_selected))

    know_slots = max(0, min(know_cap, total - len(act_selected)))
    know_selected = know[:know_slots]

    return act_selected + know_selected, omitted


def _gemini_finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        finish = getattr(cand, "finish_reason", None)
        if finish is not None:
            name = getattr(finish, "name", None)
            return str(name) if name else str(finish)
    return ""


def _llm_rank_gemini(sys_prompt: str, user_msg: str, allowed_ids: set) -> Tuple[ParsedJudgment, str]:
    """Judge with Gemini. Attempt structured output first, then plain generation.
    Returns a parsed judgment (possibly empty) on success; raises LLMExhaustedError
    after retries are exhausted."""
    model = GEMINI_MODEL
    full_prompt = f"{sys_prompt}\n\n{user_msg}"
    backoff = BASE_BACKOFF
    max_retries = 3
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        use_schema = (attempt == 0)  # first try structured; fall back to plain on any error
        try:
            config = {"temperature": 0.1, "max_output_tokens": LLM_MAX_OUTPUT_TOKENS}
            if use_schema:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = GEMINI_RESPONSE_SCHEMA
            response = gemini_client.models.generate_content(
                model=model, contents=full_prompt, config=config,
            )
            txt = (response.text or "").strip()
            finish = _gemini_finish_reason(response)
            print(f"[LLM] Gemini attempt={attempt+1}/{max_retries} schema={use_schema} "
                  f"finish={finish} chars={len(txt)}")
            parsed = _parse_llm_response(txt, allowed_ids)
            print(f"[LLM] Gemini model={model} judged act={len(parsed.act)} know={len(parsed.know)}")
            return parsed, model
        except Exception as e:
            last_err = e
            print(f"[Rank:Gemini:{model}] attempt {attempt+1} error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2.0

    raise LLMExhaustedError(f"Gemini failed after {max_retries} attempts: {type(last_err).__name__}: {last_err}")


def _llm_rank_openai(sys_prompt: str, user_msg: str, allowed_ids: set) -> Tuple[ParsedJudgment, str]:
    """Judge with OpenAI (model fallback). Returns parsed judgment on success;
    raises LLMExhaustedError if every candidate model failed."""
    backoff = BASE_BACKOFF
    last_err: Optional[Exception] = None

    for model in OPENAI_MODEL_CANDIDATES:
        try:
            print(f"[LLM] Trying OpenAI model candidate: {model}")
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
            )
            txt = resp.choices[0].message.content.strip()
            parsed = _parse_llm_response(txt, allowed_ids)
            print(f"[LLM] OpenAI model={model} judged act={len(parsed.act)} know={len(parsed.know)}")
            return parsed, model
        except Exception as e:
            last_err = e
            print(f"[LLM] OpenAI model={model} failed: {type(e).__name__}: {e}")
            time.sleep(backoff)
            backoff *= 2.0

    raise LLMExhaustedError(f"OpenAI failed all candidates: {type(last_err).__name__}: {last_err}")


def llm_rank(items: List[dict], rubric: str) -> Tuple[ParsedJudgment, str]:
    """Judge the whole pool in one request. Gemini first; a valid result (even
    empty) is returned WITHOUT calling OpenAI. Gemini failure → OpenAI chain.
    Both exhausted → LLMExhaustedError (→ error email + failed run)."""
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    print(f"[LLM] provider={LLM_PROVIDER} gemini_key_present={has_gemini_key} "
          f"openai_key_present={has_openai_key} gemini_model={GEMINI_MODEL}")

    if not items:
        default_model = GEMINI_MODEL if LLM_PROVIDER == "gemini" else OPENAI_MODEL_CANDIDATES[0]
        return ParsedJudgment([], [], 0), default_model

    allowed_ids = {it["id"] for it in items}
    payload = [{
        "id": it["id"],
        "from_name": it["from_name"],
        "from_address": it["from_address"],
        "subject": it["subject"],
        "body_for_llm": it["body_for_llm"],
    } for it in items]

    sys_prompt = SYSTEM_PROMPT.replace("{RUBRIC_MD}", rubric)
    today = datetime.datetime.now(DISPLAY_TZ).strftime("%A, %Y-%m-%d")
    user_msg = (
        f"Today is {today} ({DISPLAY_TZ_NAME}).\n\n"
        f"Candidate emails (JSON array):\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    errors: List[str] = []

    # Gemini first (if configured)
    if LLM_PROVIDER == "gemini" and gemini_client:
        try:
            return _llm_rank_gemini(sys_prompt, user_msg, allowed_ids)
        except LLMExhaustedError as e:
            print(f"[Rank] Gemini exhausted, falling back to OpenAI: {e}")
            errors.append(f"gemini: {e}")

    # OpenAI (primary when configured that way, or fallback)
    if openai_client:
        try:
            return _llm_rank_openai(sys_prompt, user_msg, allowed_ids)
        except LLMExhaustedError as e:
            errors.append(f"openai: {e}")

    raise LLMExhaustedError(
        "All LLM providers exhausted: " + " | ".join(errors)
        if errors else "No LLM provider configured (no API keys)."
    )

# -----------------------------
# Rendering — two tiers, "old look"
# -----------------------------
CATEGORY_BG = {
    "Billing": "#bbf7d0",
    "Government": "#fde68a",
    "Transaction": "#bbf7d0",
    "Travel": "#bfdbfe",
    "Security": "#e9d5ff",
    "Newsletter": "#e0e7ff",
    "Account": "#fed7aa",
    "Other": "#e5e7eb",
    # legacy labels tolerated
    "Legal": "#fecaca",
}

FOOTER_TEXT = "Reply 'noise: X' or 'missed: X' — I'll fold it into rubric.md."

def _badge(label: str, bg: str, color: str = "#111"):
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;background:{bg};color:{color};font-size:12px;font-weight:600'>{html.escape(label)}</span>"

def _titlecase_or(raw: Optional[str], fallback: str) -> str:
    s = (raw or "").strip()
    return s[:1].upper() + s[1:] if s else fallback

def _digest_ids_marker(selected: List[dict]) -> str:
    ids = [r["id"] for r in selected]
    return ",".join(ids) if ids else "none"

def _section_plain(title: str, rows: List[dict], by_id: dict) -> List[str]:
    lines = [f"## {title} ({len(rows)})"]
    if not rows:
        lines.append("_None._")
        lines.append("")
        return lines
    for i, row in enumerate(rows, 1):
        it = by_id.get(row["id"], {})
        subj = it.get("subject", "(no subject)")
        frm = it.get("from_raw") or it.get("from_name") or it.get("from_address")
        cat = row.get("category") or "Other"
        why = row.get("reason", "")
        act = row.get("action", "")
        amt = row.get("amount", "")
        prev = (it.get("body_preview") or "").strip()
        lines.append(f"{i}. {subj} — {frm}")
        lines.append(f"   - Category: {cat}" + (f" | Txn: {amt}" if amt else ""))
        if why:
            lines.append(f"   - Why: {why}")
        if act:
            lines.append(f"   - Action: {act}")
        if prev:
            lines.append(f"   - Preview: {prev}")
        lines.append("")
    return lines

def _card_html(i: int, row: dict, it: dict) -> str:
    subj = html.escape(it.get("subject", "(no subject)"))
    frm = html.escape(it.get("from_raw") or it.get("from_name") or it.get("from_address") or "")
    dom = html.escape(it.get("from_domain", "") or "")
    domain_link = f"<a href='https://{dom}' style='color:#2563eb;text-decoration:none'>{dom}</a>" if dom else ""
    prev = html.escape((it.get("body_preview") or "").strip())
    why = html.escape(row.get("reason", ""))
    act = html.escape(row.get("action", ""))
    cat = _titlecase_or(row.get("category"), "Other")
    cat_bg = CATEGORY_BG.get(cat, "#e5e7eb")

    txn_chip = ""
    amt = row.get("amount", "")
    if amt:
        txn_chip = _badge(f"Txn {amt}", "#bbf7d0")

    return f"""
      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:10px 0;background:#ffffff">
        <div style="font-weight:700;font-size:15px;line-height:1.3;margin-bottom:4px">{i}. {subj}</div>
        <div style="color:#6b7280;font-size:13px;margin-bottom:4px">{domain_link}</div>
        <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{frm}</div>
        <div style="margin-bottom:8px">
          {_badge(cat, cat_bg)}
          {'<span style="display:inline-block;width:6px"></span>' + txn_chip if txn_chip else ''}
        </div>
        {('<div style="font-size:13px;margin-bottom:6px"><strong>Why:</strong> ' + why + '</div>') if why else ''}
        {('<div style="font-size:13px;margin-bottom:6px"><strong>Action:</strong> ' + act + '</div>') if act else ''}
        {('<div style="font-size:12px;color:#6b7280"><strong>Preview:</strong> ' + prev + '</div>') if prev else ''}
      </div>
    """

def _section_html(title: str, rows: List[dict], by_id: dict) -> str:
    head = f"<h3 style='margin:16px 0 6px 0;font-size:15px'>{html.escape(title)} ({len(rows)})</h3>"
    if not rows:
        return head + "<p style='color:#6b7280;font-size:13px;margin:0 0 8px 0'><em>None.</em></p>"
    cards = "".join(_card_html(i, row, by_id.get(row["id"], {})) for i, row in enumerate(rows, 1))
    return head + cards

def _appendix_plain(items: List[dict]) -> List[str]:
    lines = [f"\n---\n## Appendix — All considered ({len(items)})"]
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        lines.append(f"- [{sg_time(it['timestamp'])}] {it.get('from_raw','?')} — {it.get('subject','(no subject)')}")
    return lines

def _appendix_html(items: List[dict]) -> str:
    rows = []
    for it in sorted(items, key=lambda x: x["timestamp"], reverse=True):
        t = html.escape(sg_time(it["timestamp"]))
        f = html.escape(it.get("from_raw") or it.get("from_name") or "?")
        s = html.escape(it.get("subject") or "(no subject)")
        rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;color:#6b7280;font-size:12px;border-bottom:1px solid #f0f0f0'>{t}</td>"
            f"<td style='padding:6px 8px;font-size:12px;border-bottom:1px solid #f0f0f0'>{f}</td>"
            f"<td style='padding:6px 8px;font-size:12px;border-bottom:1px solid #f0f0f0'>{s}</td>"
            "</tr>"
        )
    return (
        "<div style='margin-top:14px'>"
        f"<h3 style='margin:0 0 6px 0;font-size:14px'>Appendix — All considered ({len(items)})</h3>"
        "<table style='border-collapse:collapse;width:100%;margin-top:8px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Time</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>From</th>"
        "<th style='text-align:left;padding:6px 8px;font-size:12px;color:#374151'>Subject</th>"
        "</tr></thead><tbody>"
        + "".join(rows) +
        "</tbody></table></div>"
    )

def render_digest(items: List[dict], selected: List[dict], omitted_act_count: int,
                  window_start: int, window_end: int, considered_count: int,
                  used_model: str) -> Tuple[str, str]:
    ws = datetime.datetime.fromtimestamp(window_start, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    we = datetime.datetime.fromtimestamp(window_end, tz=DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    date_str = datetime.datetime.fromtimestamp(window_end, tz=DISPLAY_TZ).strftime("%Y-%m-%d")

    by_id = {it["id"]: it for it in items}
    act_rows = [r for r in selected if r.get("tier") == "ACT"]
    know_rows = [r for r in selected if r.get("tier") == "KNOW"]
    marker = _digest_ids_marker(selected)

    # ---- Plain text ----
    plain_lines = [
        f"# AI Email Digest ({date_str})",
        f"Model: {used_model}",
        f"Considered: {considered_count} emails · Window: {ws} → {we}",
        "",
    ]
    if not selected:
        plain_lines.append(f"Nothing needs you today — considered {considered_count} emails.")
        plain_lines.append("")
    else:
        plain_lines += _section_plain("Action needed", act_rows, by_id)
        if omitted_act_count > 0:
            plain_lines.append(f"+{omitted_act_count} more action items today — see appendix.")
            plain_lines.append("")
        plain_lines += _section_plain("Worth knowing", know_rows, by_id)

    plain_lines += _appendix_plain(items)
    plain_lines.append("")
    plain_lines.append(FOOTER_TEXT)
    plain_lines.append(f"[digest-ids] {marker}")   # MUST remain the last line
    plain_txt = "\n".join(plain_lines)

    # ---- HTML ----
    title = f"AI Email Digest ({date_str})"
    header_html = (
        f"<h2 style='margin:0 0 4px 0;font-size:20px'>{html.escape(title)}</h2>"
        f"<div style='color:#6b7280;font-size:12px;margin-bottom:12px'>"
        f"Model: <strong>{html.escape(used_model)}</strong> · "
        f"Considered: <strong>{considered_count}</strong> emails · "
        f"Window: {html.escape(ws)} → {html.escape(we)}</div>"
    )

    if not selected:
        body_html = (
            f"<p style='font-size:15px;margin:8px 0 4px 0'>Nothing needs you today — "
            f"considered <strong>{considered_count}</strong> emails.</p>"
        )
    else:
        body_html = _section_html("Action needed", act_rows, by_id)
        if omitted_act_count > 0:
            body_html += (
                f"<p style='font-size:13px;color:#6b7280;margin:6px 0'>"
                f"+{omitted_act_count} more action items today — see appendix.</p>"
            )
        body_html += _section_html("Worth knowing", know_rows, by_id)

    footer_html = (
        f"<div style='color:#6b7280;font-size:12px;margin-top:12px'>{html.escape(FOOTER_TEXT)}</div>"
        f"<div style='color:#9ca3af;font-size:11px;margin-top:4px'>— Generated by your Email Digest Assistant</div>"
    )

    html_body = (
        "<html><body>"
        "<div style='font-family:Inter,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:16px 12px;background:#f8fafc'>"
        + header_html + body_html + _appendix_html(items) + footer_html +
        "</div>"
        f"<!-- digest-ids: {marker} -->"
        "</body></html>"
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

def _find_last_sent_timestamp(service) -> Optional[int]:
    """
    Find the most recent digest in Gmail Sent and return its internalDate (unix seconds).
    """
    user_id = "me"
    query = 'in:sent subject:"AI Email Digest ("'
    try:
        resp = _with_retries(
            lambda: service.users().messages().list(
                userId=user_id,
                q=query,
                maxResults=1,
                includeSpamTrash=False,
            ).execute(),
            what="messages.list[sent-digest]"
        )
        msgs = resp.get("messages", []) or []
        if not msgs:
            return None
        msg_id = msgs[0]["id"]
        msg = _with_retries(
            lambda: service.users().messages().get(
                userId=user_id,
                id=msg_id,
                format="minimal",
            ).execute(),
            what=f"messages.get[sent-digest:{msg_id}]"
        )
        ts_ms = int(msg.get("internalDate", "0"))
        return ts_ms // 1000 if ts_ms else None
    except Exception as exc:
        debug_print(f"[Watermark] Sent digest lookup failed: {exc}")
        return None

def _write_last_sent_timestamp(ts: int):
    with open("last_sent_ts.txt", "w") as f:
        f.write(str(ts))

# -----------------------------
# Cross-digest dedup (marker read-back from Sent)
# -----------------------------
DIGEST_IDS_PLAIN_RE = re.compile(r'\[digest-ids\]\s*([A-Za-z0-9,]+|none)', re.I)
DIGEST_IDS_HTML_RE = re.compile(r'digest-ids:\s*([A-Za-z0-9,]+|none)', re.I)

def _extract_raw_text_parts(msg) -> Tuple[str, str]:
    """Return (plain_concat, html_concat) decoded but WITHOUT html-to-text
    stripping — so the HTML comment marker survives for fallback parsing."""
    plain: List[str] = []
    html_parts: List[str] = []

    def decode(d: Optional[str]) -> str:
        if not d:
            return ""
        try:
            return base64.urlsafe_b64decode(d.encode("UTF-8")).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def walk(p):
        mt = p.get("mimeType", "")
        body = p.get("body", {}) or {}
        if mt == "text/plain":
            plain.append(decode(body.get("data")))
        elif mt == "text/html":
            html_parts.append(decode(body.get("data")))
        for sp in (p.get("parts") or []):
            walk(sp)

    walk(msg.get("payload", {}) or {})
    return "\n".join(plain), "\n".join(html_parts)

def _extract_digest_ids_from_message(msg) -> Optional[set]:
    """Recover the [digest-ids] set from a sent digest. Plain-text marker is
    primary; HTML comment is fallback. Returns None if unparseable (fail-open)."""
    plain, html_raw = _extract_raw_text_parts(msg)
    m = (
        DIGEST_IDS_PLAIN_RE.search(plain)
        or DIGEST_IDS_PLAIN_RE.search(html_raw)
        or DIGEST_IDS_HTML_RE.search(html_raw)
        or DIGEST_IDS_HTML_RE.search(plain)
    )
    if not m:
        return None
    val = m.group(1).strip().lower()
    if val == "none":
        return set()
    return {x for x in val.split(",") if x}

def _load_suppressed_ids(service, n: int = 3) -> set:
    """Fetch the last n Sent digests (format=full) and union their marker IDs.
    Any failure or unparseable digest fails open (contributes nothing)."""
    suppressed: set = set()
    user_id = "me"
    query = 'in:sent subject:"AI Email Digest ("'
    try:
        resp = _with_retries(
            lambda: service.users().messages().list(
                userId=user_id, q=query, maxResults=n, includeSpamTrash=False,
            ).execute(),
            what="messages.list[dedup-sent]"
        )
        msgs = resp.get("messages", []) or []
    except Exception as e:
        debug_print(f"[Dedup] Sent list failed, fail-open: {e}")
        return suppressed

    for m in msgs:
        mid = m["id"]
        try:
            full = _with_retries(
                lambda mid=mid: service.users().messages().get(
                    userId=user_id, id=mid, format="full",
                ).execute(),
                what=f"messages.get[dedup:{mid}]"
            )
        except Exception as e:
            print(f"[Dedup] fetch failed for sent digest id={mid}; fail-open: {e}")
            continue
        ids = _extract_digest_ids_from_message(full)
        if ids is None:
            print(f"[Dedup] unparseable marker in sent digest id={mid}; fail-open, contributes nothing")
            continue
        suppressed |= ids

    debug_print(f"[Dedup] suppressed_ids={len(suppressed)} (from {len(msgs)} sent digests)")
    return suppressed

# -----------------------------
# Main
# -----------------------------
def main():
    if DISABLE_DIGEST:
        print("DISABLE_DIGEST=1; skipping.")
        return

    try:
        # Rubric is a hard precondition for any run (incl. empty/heartbeat days).
        rubric = load_rubric()

        service = _gmail_service()

        now = int(time.time())
        last_sent = (
            _find_last_sent_timestamp(service)
            or _read_last_sent_timestamp()
            or (now - 86400)
        )
        since_unix = max(0, last_sent - SINCE_BUFFER_HOURS * 3600)

        print("UTC now:", datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M"))
        print(f"Window (display {DISPLAY_TZ_NAME}): {sg_time(since_unix)} → {sg_time(now)}")

        items = fetch_all_since_v2(service, since_unix)
        considered_count = len(items)

        suppressed_ids = _load_suppressed_ids(service)

        parsed, used_model = llm_rank(items, rubric)
        selected, omitted_act_count = select_top(parsed, suppressed_ids, DEDUP_SUPPRESS_ACT)

        provider_used = "gemini" if used_model.startswith("gemini") else ("openai" if used_model.startswith("gpt-") else "none")
        print(f"[LLM] Summary: provider_used={provider_used} model_used={used_model} "
              f"act={sum(1 for r in selected if r['tier']=='ACT')} "
              f"know={sum(1 for r in selected if r['tier']=='KNOW')} "
              f"omitted_act={omitted_act_count} suppressed={len(suppressed_ids)}")

        plain, html_body = render_digest(
            items, selected, omitted_act_count, since_unix, now, considered_count, used_model
        )
        subject = f"AI Email Digest ({datetime.datetime.fromtimestamp(now, tz=DISPLAY_TZ).strftime('%Y-%m-%d')})"

        if DRY_RUN:
            print("=== DRY_RUN: rendered digest (plain text) ===")
            print(plain)
            print("=== DRY_RUN: decisions ===")
            for r in selected:
                print(json.dumps(r, ensure_ascii=False))
            print(f"=== DRY_RUN: omitted_act_count={omitted_act_count} "
                  f"suppressed_ids={len(suppressed_ids)} subject={subject!r} ===")
            print("DRY_RUN=1; skipping SMTP send and watermark write.")
            return

        send_digest_email(subject, plain, html_body)
        print("Digest sent.")
        _write_last_sent_timestamp(now)

    except Exception as e:
        send_error_email("Email Digest ERROR", f"{e}\n\n{traceback.format_exc()}")
        print("Digest failed:", e)
        raise  # fail the workflow visibly

if __name__ == "__main__":
    main()
