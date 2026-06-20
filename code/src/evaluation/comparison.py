"""
System comparison + LLM-as-judge scoring.

Shared evaluation logic used by both the headless runner
(scripts/run_evaluation.py) and the in-app Benchmark tab. Keeping it here means
the script and the UI score identically.

Systems:
  multi_agent  — the full 7-agent LegalAIPipeline
  single_agent — SingleAgentBaseline (one LLM call + RAG)
  no_rag       — NoRAGBaseline (one LLM call, model knowledge only)
"""

import re
import json
import time
from typing import List, Dict, Callable, Optional, Any

from loguru import logger

from src.config import get_config, DEFAULT_MODEL


JUDGE_DIMENSIONS = ["legal_correctness", "citation_quality", "completeness", "clarity"]

JUDGE_PROMPT = """You are a Lebanese legal evaluation expert. Score the legal memorandum below.

User Query: "{query}"

Legal Memorandum:
{memorandum}

Score each dimension 1-5 (1 = poor, 5 = excellent):
- legal_correctness: Is the law stated correctly for Lebanese law?
- citation_quality: Are article citations specific, relevant, and plausibly correct?
- completeness: Does it fully address the question?
- clarity: Is it well-structured and clear?

Return ONLY valid JSON (no prose, no code fences):
{{"legal_correctness":N,"citation_quality":N,"completeness":N,"clarity":N,"explanation":"one sentence"}}"""


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def build_judge(model: str = DEFAULT_MODEL) -> Callable[[str, str], dict]:
    """Return a function that scores a memorandum for a query."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    cfg = get_config()
    judge = ChatAnthropic(model=model, temperature=0.0, max_tokens=400,
                          anthropic_api_key=cfg.anthropic_api_key)

    def score(query: str, memorandum: str) -> dict:
        if not memorandum:
            return {}
        resp = judge.invoke([HumanMessage(content=JUDGE_PROMPT.format(
            query=query, memorandum=memorandum[:6000]))])
        s = extract_json(resp.content)
        vals = [s.get(d, 0) for d in JUDGE_DIMENSIONS]
        s["avg_score"] = round(sum(vals) / len(JUDGE_DIMENSIONS), 2) if any(vals) else 0
        return s

    return score


# ── System runners ───────────────────────────────────────────────────────────────

def _run_multi_agent(cases, score_fn, vectorstore, model, progress) -> List[Dict[str, Any]]:
    from src.orchestrator.coordinator import LegalAIPipeline
    pipeline = LegalAIPipeline(model=model, vectorstore=vectorstore,
                               load_vectorstore=(vectorstore is None))
    records = []
    for i, tc in enumerate(cases):
        if progress:
            progress("multi_agent", i, len(cases), tc)
        r = pipeline.process_query(tc["query"])
        rec = {
            "id": tc.get("id", f"TC{i+1}"), "query": tc["query"], "system": "multi_agent",
            "success": r.get("success", False),
            "memorandum": r.get("memorandum", ""),
            "latency_s": r.get("total_latency_s"),
            "documents_retrieved": r.get("documents_retrieved"),
            "num_citations": r.get("validation", {}).get("num_citations"),
            "num_verified_citations": r.get("validation", {}).get("num_verified_citations"),
            "grounding": r.get("grounding"),
            "cost_usd": r.get("usage", {}).get("totals", {}).get("cost_usd"),
            "total_tokens": r.get("usage", {}).get("totals", {}).get("total_tokens"),
        }
        if score_fn:
            rec["judge"] = score_fn(tc["query"], rec["memorandum"])
        records.append(rec)
    return records


def _run_baseline(cases, score_fn, system, vectorstore, model, progress) -> List[Dict[str, Any]]:
    if system == "single_agent":
        from src.baselines.single_agent_baseline import SingleAgentBaseline
        bl = SingleAgentBaseline(model=model, vectorstore=vectorstore)
    else:
        from src.baselines.no_rag_baseline import NoRAGBaseline
        bl = NoRAGBaseline(model=model)

    records = []
    for i, tc in enumerate(cases):
        if progress:
            progress(system, i, len(cases), tc)
        t0 = time.time()
        r = bl.process_query(tc["query"])
        lat = round(time.time() - t0, 2)
        rec = {
            "id": tc.get("id", f"TC{i+1}"), "query": tc["query"], "system": system,
            "success": r.get("success", False),
            "memorandum": r.get("memorandum", ""),
            "latency_s": lat,
            "documents_retrieved": r.get("documents_retrieved"),
        }
        if score_fn:
            rec["judge"] = score_fn(tc["query"], rec["memorandum"])
        records.append(rec)
    return records


def run_system(
    system: str,
    cases: List[Dict],
    score_fn: Optional[Callable] = None,
    vectorstore: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
    progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Run one system over the cases and return normalized per-query records.

    `progress(system, index, total, case)` is called before each query (for UI).
    """
    logger.info(f"Running system '{system}' over {len(cases)} cases")
    if system == "multi_agent":
        return _run_multi_agent(cases, score_fn, vectorstore, model, progress)
    if system in ("single_agent", "no_rag"):
        return _run_baseline(cases, score_fn, system, vectorstore, model, progress)
    raise ValueError(f"Unknown system: {system}")


def summarize(records: List[Dict]) -> Dict[str, Dict]:
    """Aggregate records into per-system summary stats."""
    by_system: Dict[str, list] = {}
    for r in records:
        by_system.setdefault(r["system"], []).append(r)

    summary = {}
    for system, recs in by_system.items():
        ok = [r for r in recs if r.get("success")]
        scores = [r["judge"]["avg_score"] for r in recs if r.get("judge", {}).get("avg_score")]
        lats = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
        costs = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
        dim_avgs = {}
        for d in JUDGE_DIMENSIONS:
            vals = [r["judge"].get(d, 0) for r in recs if r.get("judge")]
            vals = [v for v in vals if v]
            dim_avgs[d] = round(sum(vals) / len(vals), 2) if vals else None
        summary[system] = {
            "n": len(recs), "ok": len(ok),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else None,
            "avg_cost_usd": round(sum(costs) / len(costs), 5) if costs else None,
            "dimension_averages": dim_avgs,
        }
    return summary
