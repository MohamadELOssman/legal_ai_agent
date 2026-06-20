"""Tests for the trust / groundedness report (Workstream C1)."""

from src.utils.trust import compute_trust_report, filter_grounded_provisions


def test_fully_grounded():
    r = compute_trust_report({"grounded": 5, "ungrounded": 0}, {"verified": 4, "unverified": 0})
    assert r["hallucination_rate"] == 0.0
    assert r["grounding_rate"] == 1.0
    assert r["fully_grounded"] is True


def test_partial_hallucination():
    r = compute_trust_report({"grounded": 5, "ungrounded": 1}, {"verified": 4, "unverified": 1})
    # 2 flagged out of 11 total claims
    assert r["provisions_total"] == 6
    assert r["citations_total"] == 5
    assert r["hallucination_rate"] == round(2 / 11, 3)
    assert r["fully_grounded"] is False


def test_empty_inputs():
    r = compute_trust_report({}, {})
    assert r["hallucination_rate"] == 0.0
    assert r["grounding_rate"] == 1.0
    assert r["fully_grounded"] is False  # nothing to be confident about


def test_filter_grounded_provisions():
    provs = [
        {"article_number": "1", "grounded": True},
        {"article_number": "2", "grounded": False},
        {"article_number": "3"},  # missing flag -> treated as grounded (backward compat)
    ]
    kept = filter_grounded_provisions(provs)
    assert [p["article_number"] for p in kept] == ["1", "3"]
