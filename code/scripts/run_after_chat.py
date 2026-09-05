#!/usr/bin/env python3
"""AFTER-fix full expert benchmark — Chat (agentic) over ALL 20 questions with the
completed corpus + grounded-citation orchestrator. Judged against the gold answers.

Writes incrementally and resumes, so the recurring API network drops can't lose
progress or run for hours (220s hard cap per case).

Usage:  python scripts/run_after_chat.py
"""
import sys, json, time, signal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import LegalVectorStore
from src.evaluation.comparison import build_judge

MODEL = "claude-sonnet-5"
SRC = Path("data_processed/expert_benchmark_set.json")
OUT = Path("experiments/expert_after.json")
SUM = Path("experiments/expert_after_summary.json")
BEFORE = Path("experiments/expert_run.before.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

cases = json.loads(SRC.read_text(encoding="utf-8"))["cases"]

# before-scores (citizen only had valid scores pre-fix) for the delta column
before = {}
if BEFORE.exists():
    for r in json.loads(BEFORE.read_text(encoding="utf-8")):
        if r.get("system") == "agentic" and r.get("score") is not None:
            before[r["id"]] = r["score"]

def _good(r):
    return r.get("score") is not None and "error" not in r and (r.get("answer") or "")

results, done = [], set()
if OUT.exists():
    try:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        results = [r for r in prev if _good(r)]
        done = {r["id"] for r in results}
        logger.info(f"Resuming: kept {len(done)} good records.")
    except Exception:
        pass

vs = LegalVectorStore(); vs.load_vectorstore()
judge = build_judge(MODEL)
from src.orchestrator.agentic import AgenticLegalAssistant
asst = AgenticLegalAssistant(model=MODEL, vectorstore=vs)

class _T(Exception):
    pass
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_T("case exceeded 300s")))

t0 = time.time()
for i, c in enumerate(cases):
    if c["id"] in done:
        continue
    rec = {"id": c["id"], "type": c["user_type"], "query": c["query"]}
    signal.alarm(300)
    try:
        r = asst.chat([], c["query"])
        sc = judge(c["query"], r.get("answer", ""), c.get("reference_answer"))
        gold = [str(g) for g in c.get("gold_articles", [])]
        verified = r.get("citations", {}).get("verified", [])
        rec.update({
            "score": sc.get("avg_score"),
            "before": before.get(c["id"]),
            "dims": {k: sc.get(k) for k in ("legal_correctness", "citation_quality", "completeness", "clarity")},
            "gold": gold,
            "cited_gold": [g for g in gold if g in verified],
            "verified": verified,
            "unverified": r.get("citations", {}).get("unverified", []),
            "latency": r.get("latency_s"),
            "cost": (r.get("usage", {}) or {}).get("cost_usd"),
            "answer": r.get("answer", ""),
        })
        logger.info(f"{c['id']} → {rec['score']} (before={rec['before']}) "
                    f"cited_gold={rec['cited_gold']} ({i+1}/{len(cases)})")
    except Exception as e:
        rec["error"] = str(e)
        logger.warning(f"{c['id']} FAILED: {e}")
    finally:
        signal.alarm(0)
    results.append(rec)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

# ── Summary ──
def agg(recs, k):
    v = [r.get(k) for r in recs if isinstance(r.get(k), (int, float))]
    return round(sum(v) / len(v), 3) if v else None

ok = [r for r in results if _good(r)]
per_type = {}
for t in ("citizen", "case_study", "lawyer"):
    tr = [r for r in ok if r["type"] == t]
    tb = [r for r in tr if isinstance(r.get("before"), (int, float))]
    per_type[t] = {"n": len(tr), "after": agg(tr, "score"),
                   "before": agg(tb, "before"),
                   "n_with_before": len(tb),
                   "gold_recall": round(sum(1 for r in tr if r.get("cited_gold")) / len(tr), 3) if tr else None}
withb = [r for r in ok if isinstance(r.get("before"), (int, float))]
summary = {
    "n": len(ok), "errors": len(results) - len(ok),
    "after_mean": agg(ok, "score"),
    "before_mean_where_available": agg(withb, "before"),
    "after_mean_same_subset": agg(withb, "score"),
    "delta_same_subset": (round(agg(withb, "score") - agg(withb, "before"), 3)
                          if withb else None),
    "avg_cost_usd": agg(ok, "cost"), "avg_latency_s": agg(ok, "latency"),
    "by_type": per_type, "_elapsed_min": round((time.time() - t0) / 60, 1),
}
SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info("SUMMARY: " + json.dumps(summary, ensure_ascii=False))
logger.info(f"✓ Done in {summary['_elapsed_min']} min → {OUT}, {SUM}")
