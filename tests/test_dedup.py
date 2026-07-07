"""§7 #9 dedup marker round-trip, #10 cross-day suppression."""
import main
from conftest import build_gmail_message, make_item, act_row, know_row

WS = 1751846400
WE = 1751846400 + 3600


def _render(selected, items, omitted=0):
    return main.render_digest(items, selected, omitted, WS, WE, len(items), "gemini-3.1-flash-lite")


def test_marker_roundtrip_plain_and_html():
    selected = [act_row("aa11"), know_row("bb22")]
    items = [make_item("aa11"), make_item("bb22")]
    plain, html = _render(selected, items)
    # marker is the last plain line
    assert plain.rstrip().splitlines()[-1] == "[digest-ids] aa11,bb22"
    assert "<!-- digest-ids: aa11,bb22 -->" in html

    msg = build_gmail_message("sent1", plain=plain, html=html)
    assert main._extract_digest_ids_from_message(msg) == {"aa11", "bb22"}


def test_marker_recovered_from_html_only():
    selected = [act_row("cc33")]
    items = [make_item("cc33")]
    _, html = _render(selected, items)
    msg = build_gmail_message("sent2", html=html)  # no plain part
    assert main._extract_digest_ids_from_message(msg) == {"cc33"}


def test_marker_recovered_from_plain_only():
    selected = [know_row("dd44")]
    items = [make_item("dd44")]
    plain, _ = _render(selected, items)
    msg = build_gmail_message("sent3", plain=plain)
    assert main._extract_digest_ids_from_message(msg) == {"dd44"}


def test_heartbeat_marker_is_none():
    items = [make_item("ee55")]
    plain, html = _render([], items)
    assert plain.rstrip().splitlines()[-1] == "[digest-ids] none"
    msg = build_gmail_message("sent4", plain=plain, html=html)
    assert main._extract_digest_ids_from_message(msg) == set()


def test_unparseable_marker_fails_open():
    msg = build_gmail_message("sent5", plain="Some digest with no marker at all.")
    assert main._extract_digest_ids_from_message(msg) is None


def test_know_only_suppression_by_default():
    # #10: an ACT item resurfaces; a KNOW item is suppressed.
    parsed = main.ParsedJudgment(act=[act_row("act1")], know=[know_row("kno1"), know_row("kno2")], model_omitted=0)
    selected, _ = main.select_top(parsed, suppressed_ids={"act1", "kno1"}, suppress_act=False)
    ids = {r["id"] for r in selected}
    assert "act1" in ids          # ACT not suppressed by default
    assert "kno1" not in ids       # KNOW suppressed
    assert "kno2" in ids


def test_uniform_suppression_with_flag():
    parsed = main.ParsedJudgment(act=[act_row("act1")], know=[know_row("kno1")], model_omitted=0)
    selected, _ = main.select_top(parsed, suppressed_ids={"act1", "kno1"}, suppress_act=True)
    assert selected == []          # both suppressed when flag on
