#!/usr/bin/env python3
"""Full expert-set benchmark: Chat (agentic) vs Full Pipeline, judged against
the author-provided gold answers. Writes results incrementally so a mid-run
failure still leaves partial data.

Usage:  python scripts/run_expert_benchmark.py
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import LegalVectorStore
from src.evaluation.comparison import build_judge

MODEL = "claude-sonnet-5"
JUDGE = "claude-sonnet-5"
SRC = Path("data_processed/expert_benchmark_set.json")
OUT = Path("experiments/expert_run.json")
SUM = Path("experiments/expert_run_summary.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

cases = json.loads(SRC.read_text(encoding="utf-8"))["cases"]

# ── Resume: keep already-good records, only (re)run the failed/missing ones ──
def _good(r):
    return r.get("score") is not None and "error" not in r and (r.get("answer") or "")

results = []
done = set()
if OUT.exists():
    try:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        results = [r for r in prev if _good(r)]
        done = {(r["system"], r["id"]) for r in results}
        logger.info(f"Resuming: kept {len(done)} good records, will (re)run the rest.")
    except Exception:
        pass

vs = LegalVectorStore(); vs.load_vectorstore()
judge = build_judge(JUDGE)

from src.orchestrator.agentic import AgenticLegalAssistant
from src.orchestrator.coordinator import LegalAIPipeline
_asst = AgenticLegalAssistant(model=MODEL, vectorstore=vs)
_pipe = LegalAIPipeline(model=MODEL, vectorstore=vs, load_vectorstore=False)


def run_chat(c):
    r = _asst.chat([], c["query"])
    return (r.get("answer", ""), r.get("latency_s"),
            (r.get("usage", {}) or {}).get("cost_usd"),
            len(r.get("citations", {}).get("verified", [])))


def run_pipe(c):
    r = _pipe.process_query(c["query"])
    return (r.get("memorandum", ""), r.get("total_latency_s"),
            (r.get("usage", {}) or {}).get("totals", {}).get("cost_usd"),
            (r.get("validation", {}) or {}).get("num_verified_citations"))


# Hard per-case wall-clock cap so a hung/outage call can't run for hours.
import signal
class _CaseTimeout(Exception):
    pass
def _on_alarm(signum, frame):
    raise _CaseTimeout("case exceeded the 220s time budget")
signal.signal(signal.SIGALRM, _on_alarm)

t0 = time.time()
for sysname, runner in [("agentic", run_chat), ("multi_agent", run_pipe)]:
    for i, c in enumerate(cases):
        if (sysname, c["id"]) in done:
            continue
        rec = {"system": sysname, "id": c["id"], "type": c["user_type"], "query": c["query"]}
        signal.alarm(220)
        try:
            ans, lat, cost, vc = runner(c)
            sc = judge(c["query"], ans, c.get("reference_answer"))
            rec.update({
                "score": sc.get("avg_score"),
                "dims": {k: sc.get(k) for k in
                         ("legal_correctness", "citation_quality", "completeness", "clarity")},
                "explanation": sc.get("explanation", ""),
                "latency": lat, "cost": cost, "verified": vc,
                "answer": ans,
            })
            logger.info(f"[{sysname}] {c['id']} → score={rec['score']} ({i+1}/{len(cases)})")
        except Exception as e:
            rec["error"] = str(e)
            logger.warning(f"[{sysname}] {c['id']} FAILED: {e}")
        finally:
            signal.alarm(0)
        results.append(rec)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

# ── Summary ──────────────────────────────────────────────────────────────────
def agg(recs, key):
    v = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
    return round(sum(v) / len(v), 3) if v else None

summary = {}
for sysname in ("agentic", "multi_agent"):
    rs = [r for r in results if r["system"] == sysname]
    ok = [r for r in rs if "error" not in r]
    per_type = {}
    for t in ("citizen", "case_study", "lawyer"):
        tr = [r for r in ok if r["type"] == t]
        per_type[t] = {"n": len(tr), "avg_score": agg(tr, "score")}
    summary[sysname] = {
        "n": len(rs), "ok": len(ok), "errors": len(rs) - len(ok),
        "avg_score": agg(ok, "score"),
        "avg_latency_s": agg(ok, "latency"),
        "avg_cost_usd": agg(ok, "cost"),
        "avg_verified_citations": agg(ok, "verified"),
        "by_type": per_type,
    }
summary["_elapsed_min"] = round((time.time() - t0) / 60, 1)
SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info("=" * 60)
logger.info("SUMMARY: " + json.dumps(summary, ensure_ascii=False))
logger.info(f"✓ Done in {summary['_elapsed_min']} min → {OUT} , {SUM}")
