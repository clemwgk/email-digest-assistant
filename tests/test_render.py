"""§7 #7 heartbeat, tier sections, +N overflow line."""
import main
from conftest import make_item, act_row, know_row

WS = 1751846400
WE = 1751846400 + 3600


def test_heartbeat_when_empty():
    items = [make_item("aa11"), make_item("bb22")]
    plain, html = main.render_digest(items, [], 0, WS, WE, 2, "gemini-3.1-flash-lite")
    assert plain.startswith("# AI Email Digest (")
    assert "Nothing needs you today — considered 2 emails." in plain
    assert plain.rstrip().splitlines()[-1] == "[digest-ids] none"
    # appendix still present in a heartbeat
    assert "Appendix — All considered (2)" in plain
    assert "Nothing needs you today" in html
    assert "digest-ids: none" in html


def test_sections_and_counts():
    items = [make_item("aa11", subject="Pay me"), make_item("bb22", subject="Read me")]
    selected = [act_row("aa11", reason="bill due", action="pay it"),
                know_row("bb22", reason="newsletter")]
    plain, html = main.render_digest(items, selected, 0, WS, WE, 2, "gemini-3.1-flash-lite")
    assert "## Action needed (1)" in plain
    assert "## Worth knowing (1)" in plain
    assert "Why: bill due" in plain
    assert "Action: pay it" in plain
    assert "Action needed (1)" in html
    assert "Worth knowing (1)" in html


def test_overflow_line_present_only_when_omitted():
    items = [make_item(f"id{i:02d}") for i in range(7)]
    selected = [act_row(f"id{i:02d}") for i in range(7)]
    plain_no, _ = main.render_digest(items, selected, 0, WS, WE, 7, "m")
    assert "more action items today" not in plain_no

    plain_yes, html_yes = main.render_digest(items, selected, 3, WS, WE, 10, "m")
    assert "+3 more action items today — see appendix." in plain_yes
    assert "+3 more action items today" in html_yes


def test_txn_chip_and_preview_render():
    items = [make_item("aa11", body="Charged SGD 3103.00 to your card. " + "x" * 400)]
    selected = [act_row("aa11", category="Transaction", amount="SGD 3103.00", reason="charge")]
    plain, html = main.render_digest(items, selected, 0, WS, WE, 1, "m")
    assert "Txn: SGD 3103.00" in plain
    assert "Txn SGD 3103.00" in html
    # preview text (first PREVIEW_LEN chars) shows in the card
    assert items[0]["body_preview"][:20] in html
