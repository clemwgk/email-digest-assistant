"""§7 #8 provider exhaustion, #11 valid-empty short-circuit, #14 rubric-missing."""
import pytest

import main
from conftest import FakeGeminiClient, FakeOpenAIClient, make_item


def test_valid_empty_gemini_does_not_call_openai(monkeypatch):
    # #11: a valid empty judgment returns without touching OpenAI.
    g = FakeGeminiClient(text='{"items": [], "omitted_act_count": 0}')
    o = FakeOpenAIClient(exc=AssertionError("openai must not be called"))
    monkeypatch.setattr(main, "gemini_client", g)
    monkeypatch.setattr(main, "openai_client", o)
    monkeypatch.setattr(main, "LLM_PROVIDER", "gemini")

    parsed, model = main.llm_rank([make_item("aa11")], "rubric text")
    assert parsed.act == [] and parsed.know == []
    assert g.models.calls == 1
    assert o.chat.completions.calls == 0


def test_empty_pool_makes_no_llm_call(monkeypatch):
    g = FakeGeminiClient(exc=AssertionError("no LLM call on empty pool"))
    monkeypatch.setattr(main, "gemini_client", g)
    monkeypatch.setattr(main, "LLM_PROVIDER", "gemini")
    parsed, model = main.llm_rank([], "rubric text")
    assert parsed.act == [] and parsed.know == []
    assert g.models.calls == 0


def test_gemini_failure_falls_back_to_openai(monkeypatch):
    g = FakeGeminiClient(exc=RuntimeError("gemini down"))
    o = FakeOpenAIClient(text='{"items": [{"id":"aa11","tier":"ACT","category":"Billing","reason":"r","action":"pay"}], "omitted_act_count": 0}')
    monkeypatch.setattr(main, "gemini_client", g)
    monkeypatch.setattr(main, "openai_client", o)
    monkeypatch.setattr(main, "LLM_PROVIDER", "gemini")

    parsed, model = main.llm_rank([make_item("aa11")], "rubric text")
    assert [r["id"] for r in parsed.act] == ["aa11"]
    assert o.chat.completions.calls == 1


def test_both_providers_exhausted_raises(monkeypatch):
    # #8: both dead → LLMExhaustedError (NOT an empty result).
    monkeypatch.setattr(main, "gemini_client", FakeGeminiClient(exc=RuntimeError("boom")))
    monkeypatch.setattr(main, "openai_client", FakeOpenAIClient(exc=RuntimeError("boom2")))
    monkeypatch.setattr(main, "LLM_PROVIDER", "gemini")
    with pytest.raises(main.LLMExhaustedError):
        main.llm_rank([make_item("aa11")], "rubric text")


def test_main_provider_exhaustion_sends_error_no_digest(monkeypatch):
    sent = {"error": 0, "digest": 0, "watermark": 0}
    monkeypatch.setattr(main, "DISABLE_DIGEST", False)
    monkeypatch.setattr(main, "DRY_RUN", False)
    monkeypatch.setattr(main, "load_rubric", lambda: "rubric text")
    monkeypatch.setattr(main, "_gmail_service", lambda: object())
    monkeypatch.setattr(main, "_find_last_sent_timestamp", lambda s: 1751846400)
    monkeypatch.setattr(main, "fetch_all_since_v2", lambda s, since: [make_item("aa11")])
    monkeypatch.setattr(main, "_load_suppressed_ids", lambda s: set())

    def boom(items, rubric):
        raise main.LLMExhaustedError("both providers dead")
    monkeypatch.setattr(main, "llm_rank", boom)
    monkeypatch.setattr(main, "send_error_email", lambda subj, body: sent.__setitem__("error", sent["error"] + 1))
    monkeypatch.setattr(main, "send_digest_email", lambda *a: sent.__setitem__("digest", sent["digest"] + 1))
    monkeypatch.setattr(main, "_write_last_sent_timestamp", lambda ts: sent.__setitem__("watermark", sent["watermark"] + 1))

    with pytest.raises(main.LLMExhaustedError):
        main.main()
    assert sent["error"] == 1
    assert sent["digest"] == 0      # NO digest, NO heartbeat
    assert sent["watermark"] == 0   # watermark not advanced on failure


# ---- rubric loud-fail (#14) ----

def test_load_rubric_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RUBRIC_PATH", str(tmp_path / "nope.md"))
    with pytest.raises(main.RubricError):
        main.load_rubric()


def test_load_rubric_empty_raises(monkeypatch, tmp_path):
    p = tmp_path / "r.md"
    p.write_text("   \n\n", encoding="utf-8")
    monkeypatch.setattr(main, "RUBRIC_PATH", str(p))
    with pytest.raises(main.RubricError):
        main.load_rubric()


def test_main_missing_rubric_sends_error_no_digest(monkeypatch, tmp_path):
    sent = {"error": 0, "digest": 0}
    monkeypatch.setattr(main, "DISABLE_DIGEST", False)
    monkeypatch.setattr(main, "DRY_RUN", False)
    monkeypatch.setattr(main, "RUBRIC_PATH", str(tmp_path / "nope.md"))
    monkeypatch.setattr(main, "_gmail_service", lambda: object())
    monkeypatch.setattr(main, "send_error_email", lambda subj, body: sent.__setitem__("error", sent["error"] + 1))
    monkeypatch.setattr(main, "send_digest_email", lambda *a: sent.__setitem__("digest", sent["digest"] + 1))

    with pytest.raises(main.RubricError):
        main.main()
    assert sent["error"] == 1
    assert sent["digest"] == 0
