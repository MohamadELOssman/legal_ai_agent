#!/usr/bin/env python3
"""
Statistical evaluation study (Workstream D2 + D3).

Runs each system R times over the gold set and reports, per system:
  • objective citation metrics (precision/recall/F1 vs verified gold articles)
  • LLM-judge score (optional)
each as mean +/- 95% CI, plus paired significance tests vs the multi-agent system.

This is the quantitative backbone for the thesis comparison. Gold comes from
experiments/retrieval_gold.json (queries carry verified relevant_articles).

Note on runtime: the multi-agent system makes ~6 LLM calls per query, so a full
R x 15-query run is slow. Use --systems / --limit / --reps to size the run; the
baselines are fast and exercise the whole harness.

Usage:
  python scripts/run_study.py --systems single_agent no_rag --reps 3 --no-judge
  python scripts/run_study.py --reps 3                      # all systems, with judge
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import DEFAULT_MODEL
from src.evaluation.comparison import build_judge, run_system
from src.evaluation.stats import mean_ci, paired_test

GOLD_FILE = Path("experiments/retrieval_gold.json")
METRICS = ["citation_f1", "citation_precision", "citation_recall", "judge_score"]


def record_metric(rec: dict, metric: str):
    if metric == "judge_score":
        return rec.get("judge", {}).get("avg_score")
    return rec.get(metric)


def per_query_means(records, metric):
    """Average a metric per query id across repetitions -> {query_id: value}."""
    buckets = defaultdict(list)
    for r in records:
        v = record_metric(r, metric)
        if v is not None:
            buckets[r["id"]].append(v)
    return {qid: sum(vs) / len(vs) for qid, vs in buckets.items() if vs}


def main():
    ap = argparse.ArgumentParser(description="Statistical evaluation study")
    ap.add_argument("--systems", nargs="+",
                    default=["multi_agent", "single_agent", "no_rag"],
                    choices=["multi_agent", "single_agent", "no_rag"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", dest="judge", action="store_true", default=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--primary-metric", default="citation_f1", choices=METRICS)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cases = json.load(open(GOLD_FILE, encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[:args.limit]

    score_fn = build_judge(DEFAULT_MODEL) if args.judge else None

    # Share the vector store across systems/reps.
    from src.rag.vectorstore import LegalVectorStore
    vs = LegalVectorStore(); vs.load_vectorstore()

    logger.info(f"Study: systems={args.systems}, reps={args.reps}, "
                f"cases={len(cases)}, judge={'on' if args.judge else 'off'}")

    all_records = []
    for system in args.systems:
        for rep in range(args.reps):
            logger.info(f"[{system}] rep {rep+1}/{args.reps}")
            recs = run_system(system, cases, score_fn=score_fn, vectorstore=vs,
                              model=DEFAULT_MODEL)
            for r in recs:
                r["rep"] = rep
            all_records += recs

    # Aggregate per system: mean +/- CI for each metric over per-query means.
    by_system = defaultdict(list)
    for r in all_records:
        by_system[r["system"]].append(r)

    report = {}
    for system, recs in by_system.items():
        report[system] = {}
        for metric in METRICS:
            pqm = per_query_means(recs, metric)
            if pqm:
                report[system][metric] = mean_ci(list(pqm.values()))

    # Significance vs multi_agent on the primary metric (paired by query).
    significance = {}
    if "multi_agent" in by_system:
        base_pqm = per_query_means(by_system["multi_agent"], args.primary_metric)
        for system, recs in by_system.items():
            if system == "multi_agent":
                continue
            other_pqm = per_query_means(recs, args.primary_metric)
            shared = sorted(set(base_pqm) & set(other_pqm))
            if len(shared) >= 2:
                significance[f"multi_agent_vs_{system}"] = paired_test(
                    [base_pqm[q] for q in shared], [other_pqm[q] for q in shared])

    # ── Print ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"STATISTICAL STUDY  (reps={args.reps}, {len(cases)} queries, "
          f"judge={'on' if args.judge else 'off'})")
    print("-" * 88)
    print(f"{'System':<16}{'cit_F1 (mean±ci)':>22}{'cit_recall':>14}{'judge (mean±ci)':>22}")
    print("-" * 88)
    for system, m in report.items():
        f1 = m.get("citation_f1", {})
        rc = m.get("citation_recall", {})
        jd = m.get("judge_score", {})
        f1s = f"{f1.get('mean','-')}±{f1.get('ci95','-')}" if f1 else "-"
        rcs = f"{rc.get('mean','-')}" if rc else "-"
        jds = f"{jd.get('mean','-')}±{jd.get('ci95','-')}" if jd else "-"
        print(f"{system:<16}{f1s:>22}{rcs:>14}{jds:>22}")
    print("=" * 88)
    if significance:
        print(f"Paired significance vs multi_agent on {args.primary_metric}:")
        for k, v in significance.items():
            if v.get("available"):
                print(f"  {k}: mean_diff={v['mean_diff']} "
                      f"wilcoxon_p={v.get('wilcoxon_p')} ttest_p={v.get('ttest_p')} (n={v['n_pairs']})")
            else:
                print(f"  {k}: {v.get('reason', 'n/a')}")
        print()

    out = Path(args.output or
               f"experiments/study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "report": report, "significance": significance,
               "records": all_records}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved {out}")


if __name__ == "__main__":
    main()
