"""Shared test harness for the rubric-judge digest.

No network: Gmail and the LLM are faked. `main` is imported once; module-level
client globals are None under a bare env and get monkeypatched per test.
"""
import base64
import types

import pytest

import main


# -----------------------------
# Encoding helpers
# -----------------------------
def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii")


# -----------------------------
# Gmail-API-shaped message (format=full)
# -----------------------------
def build_gmail_message(msg_id, thread_id=None, from_hdr="Sender <sender@example.com>",
                        subject="", plain=None, html=None, internal_date_ms=1751846400000):
    """Return a dict shaped like users().messages().get(format='full')."""
    headers = [
        {"name": "From", "value": from_hdr},
        {"name": "Subject", "value": subject},
    ]
    have = [("text/plain", plain), ("text/html", html)]
    have = [(mt, body) for mt, body in have if body is not None]

    if len(have) == 1:
        mt, body = have[0]
        payload = {"mimeType": mt, "headers": headers, "body": {"data": b64(body)}}
    else:
        payload = {
            "mimeType": "multipart/alternative",
            "headers": headers,
            "parts": [{"mimeType": mt, "body": {"data": b64(body)}} for mt, body in have],
        }
    return {
        "id": msg_id,
        "threadId": thread_id or msg_id,
        "internalDate": str(internal_date_ms),
        "payload": payload,
    }


# -----------------------------
# Fake Gmail service (list / get chain)
# -----------------------------
class _Exec:
    def __init__(self, val):
        self._val = val

    def execute(self):
        return self._val


class _FakeMessagesApi:
    def __init__(self, msgs):
        self._msgs = msgs
        self._by_id = {m["id"]: m for m in msgs}

    def list(self, userId=None, q=None, pageToken=None, maxResults=None, includeSpamTrash=None):
        limit = maxResults or len(self._msgs)
        return _Exec({"messages": [{"id": m["id"]} for m in self._msgs[:limit]]})

    def get(self, userId=None, id=None, format=None):
        m = self._by_id[id]
        if format == "minimal":
            return _Exec({"id": id, "internalDate": m["internalDate"], "threadId": m.get("threadId")})
        return _Exec(m)


class _FakeUsersApi:
    def __init__(self, msgs):
        self._m = _FakeMessagesApi(msgs)

    def messages(self):
        return self._m


class FakeGmailService:
    def __init__(self, msgs):
        self._u = _FakeUsersApi(msgs)

    def users(self):
        return self._u


# -----------------------------
# Fake LLM clients
# -----------------------------
class _FakeGeminiResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = []


class _FakeGeminiModels:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.calls = 0

    def generate_content(self, model=None, contents=None, config=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return _FakeGeminiResponse(self._text)


class FakeGeminiClient:
    def __init__(self, text=None, exc=None):
        self.models = _FakeGeminiModels(text=text, exc=exc)


class _FakeOpenAIChatCompletions:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.calls = 0

    def create(self, model=None, messages=None, temperature=None, max_tokens=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        msg = types.SimpleNamespace(content=self._text)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    def __init__(self, text=None, exc=None):
        self.chat = types.SimpleNamespace(completions=_FakeOpenAIChatCompletions(text=text, exc=exc))


# -----------------------------
# Item builder (post-fetch schema)
# -----------------------------
def make_item(msg_id, subject="Subject", from_raw="Name <a@b.com>", from_domain="b.com",
              from_name="Name", from_address="a@b.com", body="Body text", ts=1751846400):
    return {
        "id": msg_id,
        "threadId": msg_id,
        "timestamp": ts,
        "from_name": from_name,
        "from_address": from_address,
        "from_domain": from_domain,
        "from_raw": from_raw,
        "subject": subject,
        "body_for_llm": (body or "")[:main.LLM_BODY_LEN],
        "body_preview": (body or "")[:main.PREVIEW_LEN],
    }


# -----------------------------
# Judgment row / ParsedJudgment builders
# -----------------------------
def act_row(msg_id, category="Billing", reason="reason", action="do it", amount=""):
    return {"id": msg_id, "tier": "ACT", "category": category,
            "reason": reason, "action": action, "amount": amount}


def know_row(msg_id, category="Newsletter", reason="reason"):
    return {"id": msg_id, "tier": "KNOW", "category": category,
            "reason": reason, "action": "", "amount": ""}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Neutralize retry backoffs so provider-failure tests run fast."""
    monkeypatch.setattr(main.time, "sleep", lambda *a, **k: None)


@pytest.fixture
def rubric_file(tmp_path, monkeypatch):
    """Point main.RUBRIC_PATH at a temp rubric with real content."""
    p = tmp_path / "rubric.md"
    p.write_text("# Rubric\nACT: bills.\nKNOW: nate.\nNoise: everything else.\n", encoding="utf-8")
    monkeypatch.setattr(main, "RUBRIC_PATH", str(p))
    return p
