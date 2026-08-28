#!/usr/bin/env python3
"""Ablation over the expert set: the Chat Assistant FULL vs one sub-agent dropped
at a time, judged against the gold answers. Runs a representative subset (2 per
type) across 3 configs. Resumes and hard-caps each case (survives API hiccups).

Usage:  python scripts/run_expert_ablation.py
"""
import sys, json, time, signal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import LegalVectorStore
from src.evaluation.comparison import build_judge

MODEL = "claude-sonnet-5"
JUDGE = "claude-sonnet-5"
SRC = Path("data_processed/expert_benchmark_set.json")
OUT = Path("experiments/expert_ablation.json")
SUM = Path("experiments/expert_ablation_summary.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Meaningful configs (dropping Research leaves retrieval via Analysis, so it is
# less interpretable and omitted here).
CONFIGS = [("full", set()), ("drop_analysis", {"analysis_agent"}),
           ("drop_citation", {"citation_agent"})]

# Representative subset: first 2 of each type.
allcases = json.loads(SRC.read_text(encoding="utf-8"))["cases"]
subset = []
for t in ("citizen", "case_study", "lawyer"):
    subset += [c for c in allcases if c["user_type"] == t][:2]

# Resume: keep good records.
def _good(r):
    return r.get("score") is not None and "error" not in r and (r.get("answer") or "")
results, done = [], set()
if OUT.exists():
    try:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        results = [r for r in prev if _good(r)]
        done = {(r["config"], r["id"]) for r in results}
        logger.info(f"Resuming: kept {len(done)} good records.")
    except Exception:
        pass

vs = LegalVectorStore(); vs.load_vectorstore()
judge = build_judge(JUDGE)
from src.orchestrator.agentic import AgenticLegalAssistant

class _CaseTimeout(Exception):
    pass
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_CaseTimeout("case exceeded 220s")))

t0 = time.time()
for name, disabled in CONFIGS:
    asst = AgenticLegalAssistant(model=MODEL, vectorstore=vs, disabled_tools=disabled)
    for c in subset:
        if (name, c["id"]) in done:
            continue
        rec = {"config": name, "id": c["id"], "type": c["user_type"], "query": c["query"]}
        signal.alarm(220)
        try:
            r = asst.chat([], c["query"])
            sc = judge(c["query"], r.get("answer", ""), c.get("reference_answer"))
            rec.update({
                "score": sc.get("avg_score"),
                "tools_used": r.get("tools_used"),
                "tools": [t.get("tool") for t in r.get("trace", [])],
                "verified": len(r.get("citations", {}).get("verified", [])),
                "latency": r.get("latency_s"),
                "cost": (r.get("usage", {}) or {}).get("cost_usd"),
                "answer": r.get("answer", ""),
            })
            logger.info(f"[{name}] {c['id']} → score={rec['score']}")
        except Exception as e:
            rec["error"] = str(e)
            logger.warning(f"[{name}] {c['id']} FAILED: {e}")
        finally:
            signal.alarm(0)
        results.append(rec)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

# ── Summary: mean score per config + contribution (full − dropped) ──
def agg(recs, k):
    v = [r.get(k) for r in recs if isinstance(r.get(k), (int, float))]
    return round(sum(v) / len(v), 3) if v else None
summary = {}
for name, _ in CONFIGS:
    rs = [r for r in results if r["config"] == name and "error" not in r]
    summary[name] = {"n": len(rs), "avg_score": agg(rs, "score"),
                     "avg_latency_s": agg(rs, "latency"), "avg_cost_usd": agg(rs, "cost"),
                     "avg_verified": agg(rs, "verified")}
full = summary.get("full", {}).get("avg_score")
if full is not None:
    for name, _ in CONFIGS:
        s = summary[name].get("avg_score")
        summary[name]["contribution_vs_full"] = (round(full - s, 3) if s is not None else None)
summary["_elapsed_min"] = round((time.time() - t0) / 60, 1)
SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info("SUMMARY: " + json.dumps(summary, ensure_ascii=False))
logger.info(f"✓ Done in {summary['_elapsed_min']} min → {OUT} , {SUM}")
