"""§7 #15 — no lingering references to deleted regex-classifier symbols."""
import os
import re

import main

MAIN_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")

DELETED_SYMBOLS = [
    "RANK_SYSTEM", "RANK_USER_TMPL", "_display_why",
    "_extract_smart_snippet_v2", "_parse_amounts_from_text", "_best_amount_by_context",
    "AMOUNT_CURR_RE", "AMOUNT_SIMPLE_RE", "SOCIAL_DOMAINS", "GeminiTruncatedOutput",
    "looks_adv", "promo_like_near_amount", "_is_social_notification",
    "txn_alert", "txn_small", "is_adv", "promo_like", "billing_cue",
    "booking_cue", "reservation_ref", "has_deadline_word",
    "_extract_json_array_text",
]


def test_no_deleted_symbols_in_main():
    src = open(MAIN_SRC, encoding="utf-8").read()
    offenders = [sym for sym in DELETED_SYMBOLS if re.search(r"\b" + re.escape(sym) + r"\b", src)]
    assert offenders == [], f"deleted symbols still present in main.py: {offenders}"


def test_no_snippet_field():
    src = open(MAIN_SRC, encoding="utf-8").read().lower()
    assert "snippet" not in src, "the 'snippet' concept should be fully removed"


def test_new_surface_present():
    # sanity: the redesigned surface exists
    for name in ("load_rubric", "select_top", "LLMExhaustedError", "RubricError",
                 "SYSTEM_PROMPT", "ParsedJudgment", "_extract_digest_ids_from_message"):
        assert hasattr(main, name), f"missing expected symbol {name}"
