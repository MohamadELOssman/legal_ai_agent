"""Tests for objective citation extraction + metrics (Workstream D2)."""

from src.evaluation.comparison import extract_cited_articles, citation_metrics


def test_extract_ascii_arabic():
    assert extract_cited_articles("المادة 549 من قانون العقوبات") == {"549"}


def test_extract_arabic_indic_digits():
    # ٥٤٧ / ٥٤٩ must normalize to ASCII so they match gold.
    assert extract_cited_articles("بموجب المادة ٥٤٧ والمادة ٥٤٩") == {"547", "549"}


def test_extract_plural_and_multiple():
    assert extract_cited_articles("تطبّق المواد 453 و 456 و 459") == {"453", "456", "459"}


def test_extract_english_plural():
    assert extract_cited_articles("Articles 638 and 639 of the Penal Code") == {"638", "639"}


def test_extract_dual_form():
    assert extract_cited_articles("المادتين 564 و 565") == {"564", "565"}


def test_extract_ignores_years_and_amounts():
    assert extract_cited_articles("in 2024 the amount was 150000") == set()


def test_citation_metrics_partial():
    m = citation_metrics({"547", "548", "549"}, {"547", "549", "700"})
    assert m["citation_precision"] == round(2 / 3, 3)  # 547,549 correct of 3 cited
    assert m["citation_recall"] == round(2 / 3, 3)     # 547,549 found of 3 gold
    assert 0 < m["citation_f1"] <= 1


def test_citation_metrics_no_gold_returns_empty():
    assert citation_metrics({"549"}, set()) == {}
