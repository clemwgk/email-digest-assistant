"""§7 #13 mask precision, #40 OTP masking, card last-4 (KEEP)."""
import main


def test_marketing_uppercase_tokens_survive_without_otp_keywords():
    # #13 first half: no OTP keyword present → nothing masked, code-like tokens survive.
    otp_like, subj, body = main.classify_and_mask_otp(
        "Get up to 40% off with CODE40", "Shop now and save with promo CODE40."
    )
    assert otp_like is False
    assert "CODE40" in subj
    assert "CODE40" in body
    assert "‹code›" not in (subj + body)


def test_real_otp_code_is_masked():
    # #40 / #13 second half: OTP keyword + digit-bearing token → masked.
    otp_like, subj, body = main.classify_and_mask_otp(
        "Your login code", "Your login code is 738201 and expires in 10 minutes."
    )
    assert otp_like is True
    assert "738201" not in body
    assert "‹code›" in body
    # plain words near the code are untouched
    assert "expires" in body


def test_transaction_alert_not_shredded():
    # fixture 37 regression: a txn alert with no OTP keyword stays readable.
    otp_like, subj, body = main.classify_and_mask_otp(
        "Transaction Alert: SGD 3,103.00",
        "A transaction of SGD 3,103.00 was made on your VISA REVOLUTION card.",
    )
    assert otp_like is False
    assert "3,103.00" in body
    assert "VISA" in body
    assert "REVOLUTION" in body


def test_negative_cue_prevents_masking():
    # Security-awareness lecture ("for your security" / "never share") is not treated as OTP.
    otp_like, subj, body = main.classify_and_mask_otp(
        "For your security", "For your security, never share your verification code 999888."
    )
    assert otp_like is False
    assert body.count("‹code›") == 0


def test_only_digit_bearing_tokens_masked():
    # Within a genuine OTP context, letter-only 4-8 tokens survive; digit-bearing ones mask.
    masked = main._mask_codes("PLAIN WORDS and A1B2C3 and 1234")
    assert "PLAIN" in masked and "WORDS" in masked
    assert "A1B2C3" not in masked
    assert "1234" not in masked
    assert masked.count("‹code›") == 2


def test_card_last4_still_masked():
    # main.py:165-166 preserved — card last-4 not parsed as money.
    assert "‹last4›" in main._post_process_text("Your card ending 1234 was charged")
    assert "‹last4›" in main._post_process_text("card **** 5678")
