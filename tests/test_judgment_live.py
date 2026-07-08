"""§7 #1-5, #11-KNOW — judgment quality.

These CANNOT be validated with a mocked LLM (you'd only be testing the mock), so
they call the REAL Gemini API against the REAL rubric.md + prompt. Skipped unless
GEMINI_API_KEY is set. This is the automated companion to the M4 human read of a
real dry-run. Expected tiers come from golden-set-expectations.md (★ rows).

flash-lite judgment is probabilistic; assertions here cover only the clearest ★
cases. Treat a failure as a signal to iterate the prompt/rubric, not a hard bug.
"""
import os

import pytest

import main
from conftest import make_item

pytestmark = pytest.mark.live

if not os.getenv("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set; skipping live judgment tests", allow_module_level=True)


# (id, from_address, subject, body, expected) — expected ∈ {"ACT","KNOW","EXCLUDE"}
GOLDEN = [
    ("g01", "hsbc@notification.hsbc.com.hk", "Transaction Alert",
     "A transfer of SGD 3,103.00 was debited from your account today.", "ACT"),
    ("g04", "ebanking@mail.hsbc.com.sg", "HSBC VISA REVOLUTION eStatement",
     "Your HSBC VISA REVOLUTION credit card eStatement is ready. Total amount due SGD 812.40 by 20 Jul.", "ACT"),
    ("g05", "security@mail.instagram.com", "Did you just add an account?",
     "We noticed a new account was added to your Accounts Centre. If this wasn't you, secure your account.", "ACT"),
    ("g07", "notification@e.cpf.gov.sg", "View your CPF statement",
     "Your CPF monthly statement is now available to view. Log in to see your balances.", "EXCLUDE"),
    ("g11", "nateszerotoai@substack.com", "AI's next act",
     "This week in AI: the state of agents, and what it means for builders.", "KNOW"),
    ("g13", "sabrinaramonov@substack.com", "Growth tactics",
     "My best growth playbook for creators this month.", "EXCLUDE"),
    ("g17", "sdic@notice.sg", "Deposit insurance",
     "Your deposits are insured up to S$100,000 by SDIC. No action required.", "EXCLUDE"),
]


@pytest.fixture(scope="module")
def judged():
    rubric = main.load_rubric()
    items = [make_item(gid, subject=subj, from_raw=f"<{addr}>", from_address=addr,
                       from_domain=addr.split("@")[-1], from_name=addr, body=body)
             for (gid, addr, subj, body, _exp) in GOLDEN]
    parsed, model = main.llm_rank(items, rubric)
    tier_by_id = {}
    for r in parsed.act:
        tier_by_id[r["id"]] = "ACT"
    for r in parsed.know:
        tier_by_id[r["id"]] = "KNOW"
    return tier_by_id


@pytest.mark.parametrize("gid,addr,subj,body,expected", GOLDEN)
def test_golden_row(judged, gid, addr, subj, body, expected):
    got = judged.get(gid, "EXCLUDE")
    assert got == expected, f"{addr} :: expected {expected}, got {got}"
