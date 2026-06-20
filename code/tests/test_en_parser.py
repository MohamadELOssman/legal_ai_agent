"""Tests for the English Penal Code article-number normalization (Workstream A)."""

from scripts.parse_penal_code_en import _normalize_article_number, MAX_ARTICLE


def test_valid_number_passthrough():
    assert _normalize_article_number("549") == "549"
    assert _normalize_article_number(str(MAX_ARTICLE)) == str(MAX_ARTICLE)


def test_footnote_superscript_recovery():
    # "Article 547¹" extracted as "5471" -> recover 547; same for 548.
    assert _normalize_article_number("5471") == "547"
    assert _normalize_article_number("5481") == "548"


def test_above_max_is_trimmed_to_plausible():
    # Drops trailing digits until <= MAX_ARTICLE.
    assert int(_normalize_article_number("771")) <= MAX_ARTICLE
