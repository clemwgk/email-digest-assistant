"""§7 #12 body_preview derives from the same masked text; item schema."""
import main
from conftest import FakeGmailService, build_gmail_message

LEGACY_KEYS = {"snippet", "txn_alert", "txn_small", "txn_amount", "txn_currency",
               "is_adv", "promo_like", "is_social", "billing_cue", "booking_cue",
               "reservation_ref", "has_deadline_word", "otp_like"}


def test_item_schema_and_preview_invariant():
    long_body = "This is the actionable content. " + ("filler " * 500)
    msg = build_gmail_message("m1", from_hdr="HSBC <ebanking@mail.hsbc.com.sg>",
                              subject="Your eStatement", html=f"<p>{long_body}</p>")
    service = FakeGmailService([msg])
    items = main.fetch_all_since_v2(service, since_unix=0)
    assert len(items) == 1
    it = items[0]

    # new schema present
    for k in ("from_name", "from_address", "body_for_llm", "body_preview"):
        assert k in it
    # legacy flags gone
    assert LEGACY_KEYS.isdisjoint(it.keys())

    # #12: preview and llm-body derive from the SAME masked string
    assert it["body_preview"] == it["body_for_llm"][:main.PREVIEW_LEN]
    assert it["body_for_llm"].startswith(it["body_preview"])
    assert len(it["body_for_llm"]) <= main.LLM_BODY_LEN
    assert len(it["body_preview"]) <= main.PREVIEW_LEN
    assert it["from_address"] == "ebanking@mail.hsbc.com.sg"


def test_mask_applied_in_fetch():
    msg = build_gmail_message("m2", subject="Your login code",
                              plain="Your login code is 738201 please use it now")
    service = FakeGmailService([msg])
    items = main.fetch_all_since_v2(service, since_unix=0)
    it = items[0]
    assert "738201" not in it["body_for_llm"]
    assert "‹code›" in it["body_for_llm"]


def test_thread_dedup_keeps_one_per_thread():
    m1 = build_gmail_message("m1", thread_id="t1", subject="First", plain="hello one",
                             internal_date_ms=1751846400000)
    m2 = build_gmail_message("m2", thread_id="t1", subject="Second", plain="hello two",
                             internal_date_ms=1751846460000)
    service = FakeGmailService([m1, m2])
    items = main.fetch_all_since_v2(service, since_unix=0)
    assert len(items) == 1  # same thread collapses
