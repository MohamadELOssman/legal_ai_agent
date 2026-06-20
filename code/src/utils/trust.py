"""
Trust / groundedness reporting (Workstream C1).

Combines the two trust signals the pipeline already produces — provision
grounding (Analysis agent) and citation verification (Citation agent) — into a
single, surfaceable report with a hallucination rate. This is the headline
trustworthiness metric for the system and for the thesis evaluation.
"""

from typing import Dict, Any, List


def compute_trust_report(grounding: Dict[str, Any], citation_validation: Dict[str, Any]) -> Dict[str, Any]:
    """Build a trust report from analysis grounding + citation validation.

    grounding:            {"grounded": int, "ungrounded": int, ...}
    citation_validation:  {"verified": int, "unverified": int, ...}
    """
    pg = int((grounding or {}).get("grounded", 0) or 0)
    pu = int((grounding or {}).get("ungrounded", 0) or 0)
    cv = int((citation_validation or {}).get("verified", 0) or 0)
    cu = int((citation_validation or {}).get("unverified", 0) or 0)

    prov_total = pg + pu
    cit_total = cv + cu
    denom = prov_total + cit_total
    flagged = pu + cu

    return {
        "provisions_total": prov_total,
        "provisions_grounded": pg,
        "provisions_ungrounded": pu,
        "citations_total": cit_total,
        "citations_verified": cv,
        "citations_unverified": cu,
        # Fraction of all legal claims (provisions + citations) NOT backed by the corpus.
        "hallucination_rate": round(flagged / denom, 3) if denom else 0.0,
        "grounding_rate": round((pg + cv) / denom, 3) if denom else 1.0,
        "fully_grounded": flagged == 0 and denom > 0,
    }


def filter_grounded_provisions(provisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only provisions grounded in the retrieved corpus.

    Used when grounding enforcement is enabled so the final memorandum is built
    solely on provisions whose article numbers appear in the retrieved documents.
    A provision missing the `grounded` key is treated as grounded (backward
    compatible with outputs produced before grounding existed).
    """
    return [p for p in provisions if p.get("grounded", True)]
