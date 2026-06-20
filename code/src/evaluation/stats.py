"""
Statistical helpers for system comparison (Workstream D3).

Provides mean +/- 95% confidence intervals and paired significance tests
(Wilcoxon signed-rank + paired t-test) so claims like "multi-agent beats the
single-agent baseline" are backed by statistics, not single-run anecdotes.
"""

import math
from typing import List, Dict, Sequence

try:
    from scipy import stats as _sp
    SCIPY = True
except Exception:  # pragma: no cover
    SCIPY = False


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Dict:
    """Mean with a confidence interval (t-based; falls back to z=1.96 sans scipy)."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": round(mean, 3), "std": 0.0, "ci95": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    tcrit = float(_sp.t.ppf(0.5 + confidence / 2, n - 1)) if SCIPY else 1.96
    margin = tcrit * std / math.sqrt(n)
    return {"mean": round(mean, 3), "std": round(std, 3), "ci95": round(float(margin), 3), "n": n}


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Paired significance test between two systems' per-item metric values.

    `a` and `b` must be aligned (same items, same order). Returns Wilcoxon and
    paired-t p-values plus the mean difference (a - b).
    """
    a = list(a)
    b = list(b)
    if len(a) != len(b) or len(a) < 2:
        return {"available": False, "reason": "need >=2 paired observations"}
    mean_diff = round((sum(a) - sum(b)) / len(a), 3)
    if not SCIPY:
        return {"available": False, "reason": "scipy not installed", "mean_diff": mean_diff}
    out = {"available": True, "mean_diff": mean_diff, "n_pairs": len(a)}
    try:
        if any(x != y for x, y in zip(a, b)):
            out["wilcoxon_p"] = round(float(_sp.wilcoxon(a, b).pvalue), 4)
        else:
            out["wilcoxon_p"] = 1.0  # identical samples
    except Exception as e:
        out["wilcoxon_p"] = None
        out["wilcoxon_error"] = str(e)
    try:
        out["ttest_p"] = round(float(_sp.ttest_rel(a, b).pvalue), 4)
    except Exception as e:
        out["ttest_p"] = None
        out["ttest_error"] = str(e)
    return out
