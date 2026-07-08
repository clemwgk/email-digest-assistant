"""§7 #9 dedup marker round-trip, #10 cross-day suppression."""
import main
from conftest import (
    FakeGmailService,
    build_gmail_message,
    make_item,
    act_row,
    know_row,
)

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


def test_load_suppressed_ids_end_to_end_fail_open(capsys):
    """#9/#10 glue: _load_suppressed_ids reads the last Sent digests, unions the
    IDs of the parseable ones, and fails open (contributes nothing + logs) on an
    unparseable digest. Exercises the real production entry point, not just the
    extractor it calls."""
    # A genuine sent digest, rendered through the real pipeline.
    good_plain, good_html = _render([act_row("gd01"), know_row("gd02")],
                                    [make_item("gd01"), make_item("gd02")])
    good = build_gmail_message("good1", subject="AI Email Digest (Jul 8)",
                               plain=good_plain, html=good_html)
    # An unparseable sent digest: subject matches the dedup query but no marker.
    broken = build_gmail_message("unparse1", subject="AI Email Digest (Jul 7)",
                                 plain="A digest body with no digest-ids marker at all.")

    service = FakeGmailService([good, broken])
    suppressed = main._load_suppressed_ids(service)

    # Union comes only from the parseable digest; the broken one contributes nothing.
    assert suppressed == {"gd01", "gd02"}
    # Fail-open is observable (no silent failure): a log line names the broken digest.
    out = capsys.readouterr().out
    assert "unparseable marker in sent digest id=unparse1" in out


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
