"""
System comparison + LLM-as-judge scoring.

Shared evaluation logic used by both the headless runner
(scripts/run_evaluation.py) and the in-app Benchmark tab. Keeping it here means
the script and the UI score identically.

Systems:
  multi_agent  — the full 7-agent LegalAIPipeline
  agentic      — the chat assistant (orchestrator that calls sub-agents as needed)
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

JUDGE_PROMPT = """# CONTEXT
You are evaluating one answer produced by a Lebanese criminal-law AI system (Penal Code + Code
of Criminal Procedure). No human reference answer is available for this item, so judge on legal
soundness and quality alone.

# ROLE
You are an impartial Lebanese legal evaluation expert acting as an LLM-as-judge.

# ACTION
Read the user query and the AI's legal memorandum, then score four dimensions from 1 (poor) to
5 (excellent):
- legal_correctness: is the law stated correctly for Lebanese law?
- citation_quality: are article citations specific, relevant, and plausibly correct?
- completeness: does it fully address the question?
- clarity: is it well-structured and clear?

User Query: "{query}"

Legal Memorandum:
{memorandum}

# FORMAT
Return ONLY valid JSON (no prose, no code fences):
{{"legal_correctness":N,"citation_quality":N,"completeness":N,"clarity":N,"explanation":"one sentence"}}"""


# Used when the user supplied a ground-truth ("source of truth") answer: the judge
# scores the AI answer RELATIVE to that reference.
JUDGE_PROMPT_REF = """# CONTEXT
You are evaluating one answer produced by a Lebanese criminal-law AI system, and a human expert
has supplied a REFERENCE (ground-truth) answer. The reference is the standard of correctness —
judge the AI answer RELATIVE to it, not against your own opinion.

# ROLE
You are an impartial Lebanese legal evaluation expert acting as an LLM-as-judge.

# ACTION
Compare the AI answer against the reference and score four dimensions from 1 (poor) to 5
(excellent), judged AGAINST the reference:
- legal_correctness: does the AI answer agree with the reference on the law and the conclusion?
- citation_quality: are the cited articles consistent with the reference?
- completeness: does it cover what the reference covers?
- clarity: is it well-structured and clear?

User Query: "{query}"

REFERENCE (ground-truth) answer:
{reference}

AI Answer to evaluate:
{memorandum}

# FORMAT
Return ONLY valid JSON (no prose, no code fences):
{{"legal_correctness":N,"citation_quality":N,"completeness":N,"clarity":N,"explanation":"one sentence"}}"""


def _content_to_text(content) -> str:
    """Normalize an LLM response to plain text. Reasoning models (e.g. claude-sonnet-5)
    return `content` as a LIST of blocks (thinking + text) rather than a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") in (None, "text") and "text" in b)
    return str(content or "")


def extract_json(text) -> dict:
    """Best-effort JSON extraction from an LLM response (string or block list)."""
    text = _content_to_text(text).strip()
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


# ── Objective citation metrics (vs verified gold article numbers) ────────────────

# Arabic-Indic digits -> ASCII, so "المادة ٥٤٧" matches gold "547".
_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# A citation keyword (Arabic singular/dual/plural of "article", or English
# article/art.) followed by a short run that may list several numbers joined by
# connectors, e.g. "المواد 638 و 639", "Articles 453, 456 and 459".
_CITE_KEYWORD = re.compile(
    r"(?:الماد[ةتّ]\w*|الموادّ?|المواد|articles?|art\.?)\s*([0-9][0-9\s,،/\-وandAND&]{0,40})",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d{1,3}")


def extract_cited_articles(text: str) -> set:
    """Extract the set of article numbers a memorandum cites (AR + EN forms).

    Normalizes Arabic-Indic digits and captures multiple numbers per reference
    (e.g. "المواد 638 و 639") so the citation metric is not silently undercounted.
    """
    text = (text or "").translate(_AR_INDIC)
    out = set()
    for m in _CITE_KEYWORD.finditer(text):
        out.update(_NUM.findall(m.group(1)))
    return out


def citation_metrics(cited: set, gold: set) -> dict:
    """Precision / recall / F1 of cited articles against the gold article set."""
    if not gold:
        return {}
    tp = len(cited & gold)
    precision = tp / len(cited) if cited else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "citation_precision": round(precision, 3),
        "citation_recall": round(recall, 3),
        "citation_f1": round(f1, 3),
        "cited_articles": sorted(cited),
        "gold_articles": sorted(gold),
    }


def _attach_citation_metrics(rec: dict, tc: dict) -> dict:
    """Add objective citation metrics to a record when the case carries gold."""
    gold = set(tc.get("relevant_articles") or tc.get("gold_articles") or [])
    if gold:
        rec.update(citation_metrics(extract_cited_articles(rec.get("memorandum", "")), gold))
    return rec


def build_judge(model: str = DEFAULT_MODEL) -> Callable[[str, str], dict]:
    """Return a function that scores a memorandum for a query."""
    from langchain_core.messages import HumanMessage
    from src.utils.llm import make_chat

    cfg = get_config()
    judge = make_chat(model=model, api_key=cfg.anthropic_api_key,
                      temperature=0.0, max_tokens=400)

    def score(query: str, memorandum: str, reference: str = None) -> dict:
        if not memorandum:
            return {}
        if reference and reference.strip():
            prompt = JUDGE_PROMPT_REF.format(
                query=query, reference=reference[:4000], memorandum=memorandum[:6000])
        else:
            prompt = JUDGE_PROMPT.format(query=query, memorandum=memorandum[:6000])
        resp = judge.invoke([HumanMessage(content=prompt)])
        s = extract_json(resp.content)
        vals = [s.get(d, 0) for d in JUDGE_DIMENSIONS]
        s["avg_score"] = round(sum(vals) / len(JUDGE_DIMENSIONS), 2) if any(vals) else 0
        s["reference_based"] = bool(reference and reference.strip())
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
        _attach_citation_metrics(rec, tc)
        if score_fn:
            rec["judge"] = score_fn(tc["query"], rec["memorandum"], tc.get("reference_answer"))
        records.append(rec)
    return records


def _run_agentic(cases, score_fn, vectorstore, model, progress) -> List[Dict[str, Any]]:
    """The agentic chat assistant (orchestrator that calls sub-agents as needed)."""
    from src.orchestrator.agentic import AgenticLegalAssistant
    assistant = AgenticLegalAssistant(model=model, vectorstore=vectorstore)
    records = []
    for i, tc in enumerate(cases):
        if progress:
            progress("agentic", i, len(cases), tc)
        r = assistant.chat([], tc["query"])
        cits = r.get("citations", {}) or {}
        usage = r.get("usage", {}) or {}
        rec = {
            "id": tc.get("id", f"TC{i+1}"), "query": tc["query"], "system": "agentic",
            "success": True,
            "memorandum": r.get("answer", ""),
            "latency_s": r.get("latency_s"),
            "num_citations": len(cits.get("cited", [])),
            "num_verified_citations": len(cits.get("verified", [])),
            "cost_usd": usage.get("cost_usd"),
            "total_tokens": usage.get("total_tokens"),
        }
        _attach_citation_metrics(rec, tc)
        if score_fn:
            rec["judge"] = score_fn(tc["query"], rec["memorandum"], tc.get("reference_answer"))
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
        _attach_citation_metrics(rec, tc)
        if score_fn:
            rec["judge"] = score_fn(tc["query"], rec["memorandum"], tc.get("reference_answer"))
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
    if system == "agentic":
        return _run_agentic(cases, score_fn, vectorstore, model, progress)
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

        def _avg(key):
            vals = [r[key] for r in recs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        summary[system] = {
            "n": len(recs), "ok": len(ok),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else None,
            "avg_cost_usd": round(sum(costs) / len(costs), 5) if costs else None,
            # Objective citation metrics vs gold (when cases carry relevant_articles).
            "citation_precision": _avg("citation_precision"),
            "citation_recall": _avg("citation_recall"),
            "citation_f1": _avg("citation_f1"),
            "dimension_averages": dim_avgs,
        }
    return summary
