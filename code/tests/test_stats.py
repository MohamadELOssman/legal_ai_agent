"""Tests for the statistics helpers (Workstream D3)."""

from src.evaluation.stats import mean_ci, paired_test


def test_mean_ci_basic():
    r = mean_ci([0.6, 0.7, 0.8, 0.65])
    assert r["n"] == 4
    assert r["mean"] == 0.688  # mean 0.6875 rounded to 3 dp
    assert r["ci95"] > 0
    # ci95 must be a plain float (JSON-serializable, not numpy)
    assert isinstance(r["ci95"], float)


def test_mean_ci_single_value():
    r = mean_ci([0.5])
    assert r["n"] == 1 and r["mean"] == 0.5 and r["ci95"] == 0.0


def test_mean_ci_empty():
    r = mean_ci([])
    assert r["n"] == 0 and r["mean"] is None


def test_paired_test_detects_difference():
    a = [0.8, 0.9, 0.85, 0.7]
    b = [0.5, 0.6, 0.55, 0.5]
    r = paired_test(a, b)
    assert r["available"] is True
    assert r["mean_diff"] > 0
    assert r["ttest_p"] is not None


def test_paired_test_identical():
    r = paired_test([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert r["available"] is True
    assert r["mean_diff"] == 0.0
    assert r["wilcoxon_p"] == 1.0


def test_paired_test_too_few():
    assert paired_test([0.5], [0.4])["available"] is False
