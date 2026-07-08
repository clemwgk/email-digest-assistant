"""§7 #6 parser + selection budget."""
import pytest

import main
from conftest import act_row, know_row

IDS = {f"id{i:02d}" for i in range(20)}


def _obj(items, omitted=0):
    import json
    return json.dumps({"items": items, "omitted_act_count": omitted})


def test_parse_empty_is_legitimate():
    parsed = main._parse_llm_response(_obj([]), IDS)
    assert parsed.act == [] and parsed.know == [] and parsed.model_omitted == 0


@pytest.mark.parametrize("n", [1, 5, 6, 7])
def test_parse_n_items(n):
    items = [{"id": f"id{i:02d}", "tier": "ACT", "category": "Billing", "reason": "r"} for i in range(n)]
    parsed = main._parse_llm_response(_obj(items), IDS)
    assert len(parsed.act) == n


def test_invalid_tier_row_rejected():
    items = [
        {"id": "id00", "tier": "MAYBE", "category": "Other", "reason": "r"},
        {"id": "id01", "tier": "ACT", "category": "Billing", "reason": "r"},
    ]
    parsed = main._parse_llm_response(_obj(items), IDS)
    assert [r["id"] for r in parsed.act] == ["id01"]
    assert parsed.know == []


def test_unknown_and_duplicate_ids_rejected():
    items = [
        {"id": "NOPE", "tier": "ACT", "category": "Other", "reason": "r"},
        {"id": "id02", "tier": "ACT", "category": "Other", "reason": "r"},
        {"id": "id02", "tier": "KNOW", "category": "Other", "reason": "r"},  # dup
    ]
    parsed = main._parse_llm_response(_obj(items), IDS)
    assert [r["id"] for r in parsed.act] == ["id02"]
    assert parsed.know == []


def test_code_fenced_json_parses():
    fenced = "```json\n" + _obj([{"id": "id00", "tier": "KNOW", "category": "Newsletter", "reason": "r"}]) + "\n```"
    parsed = main._parse_llm_response(fenced, IDS)
    assert [r["id"] for r in parsed.know] == ["id00"]


def test_malformed_raises():
    with pytest.raises(ValueError):
        main._parse_llm_response("not json at all", IDS)
    with pytest.raises(ValueError):
        main._parse_llm_response('{"no_items_key": true}', IDS)


# ---- selection budget (select_top) ----

def _pj(act, know, model_omitted=0):
    return main.ParsedJudgment(act=act, know=know, model_omitted=model_omitted)


def test_budget_no_overflow():
    parsed = _pj([act_row(f"id{i:02d}") for i in range(5)], [])
    selected, omitted = main.select_top(parsed)
    assert len(selected) == 5
    assert omitted == 0


@pytest.mark.parametrize("n_act,expected_omitted", [(7, 0), (8, 1), (10, 3)])
def test_act_overflow_truncates_to_seven(n_act, expected_omitted):
    parsed = _pj([act_row(f"id{i:02d}") for i in range(n_act)], [])
    selected, omitted = main.select_top(parsed)
    assert len(selected) == 7
    assert all(r["tier"] == "ACT" for r in selected)
    assert omitted == expected_omitted


def test_model_reported_omitted_is_added():
    parsed = _pj([act_row(f"id{i:02d}") for i in range(7)], [], model_omitted=2)
    selected, omitted = main.select_top(parsed)
    assert len(selected) == 7
    assert omitted == 2


def test_know_fills_remaining_slots_capped_at_three():
    parsed = _pj([act_row("id00"), act_row("id01")],
                 [know_row(f"id{i:02d}") for i in range(5, 10)])
    selected, omitted = main.select_top(parsed)
    # 2 ACT + min(3, 7-2)=3 KNOW
    assert sum(1 for r in selected if r["tier"] == "ACT") == 2
    assert sum(1 for r in selected if r["tier"] == "KNOW") == 3


def test_know_only_capped_at_three():
    parsed = _pj([], [know_row(f"id{i:02d}") for i in range(5)])
    selected, omitted = main.select_top(parsed)
    assert len(selected) == 3
    assert omitted == 0


def test_full_act_leaves_no_room_for_know():
    parsed = _pj([act_row(f"id{i:02d}") for i in range(7)], [know_row("id19")])
    selected, omitted = main.select_top(parsed)
    assert len(selected) == 7
    assert all(r["tier"] == "ACT" for r in selected)
