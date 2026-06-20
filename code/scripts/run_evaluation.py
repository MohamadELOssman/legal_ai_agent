#!/usr/bin/env python3
"""
Batch Evaluation Runner — Multi-Agent vs Baselines

Runs a set of legal queries through one or more systems and (optionally) scores
each final memorandum with an LLM-as-judge. Produces a JSON file of raw results
plus a printed summary table — the core comparison for the thesis.

Shared scoring/runner logic lives in src/evaluation/comparison.py, so this script
and the in-app Benchmark tab evaluate identically.

Usage:
  python scripts/run_evaluation.py                         # all systems, default cases, with judge
  python scripts/run_evaluation.py --systems multi_agent   # one system only
  python scripts/run_evaluation.py --limit 3 --no-judge    # quick smoke run
  python scripts/run_evaluation.py --cases my_cases.json   # custom test set
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import DEFAULT_MODEL
from src.evaluation.comparison import build_judge, run_system, summarize


# ── Default test set (Arabic criminal/penal — matches the in-app benchmark) ──────

DEFAULT_CASES = [
    {"id": "TC1", "query": "ما العقوبة التي يستحقها من قتل إنساناً قصداً بالأشغال الشاقة؟", "domain": "criminal"},
    {"id": "TC2", "query": "ما هي حالات القتل العمد المشددة التي تستوجب الإعدام أو الأشغال الشاقة المؤبدة؟", "domain": "criminal"},
    {"id": "TC3", "query": "ما هي الأفعال التي تُعدّ دفاعاً مشروعاً عن النفس والأموال ضد السرقة والدخول ليلاً؟", "domain": "criminal"},
    {"id": "TC4", "query": "ما عقوبة من يُكره شخصاً على الجماع بالعنف أو التهديد أو يستغل عجزه الجسدي أو النفسي؟", "domain": "criminal"},
    {"id": "TC5", "query": "ما هي عقوبة السرقة المشددة بالكسر والخلع من المصارف والمؤسسات العامة؟", "domain": "criminal"},
    {"id": "TC6", "query": "ما هي عقوبة تزوير الأوراق الرسمية سواء أقدم عليه موظف عام أم شخص عادي؟", "domain": "criminal"},
    {"id": "TC7", "query": "ما هي أركان جريمة الاحتيال بالمناورات الاحتيالية أو الادعاءات الكاذبة أو الاسم المستعار؟", "domain": "criminal"},
    {"id": "TC8", "query": "هل يستفيد من العذر المخفف من يفاجئ زوجه في جرم الزنا المشهود ويقدم على قتله؟", "domain": "criminal"},
    {"id": "TC9", "query": "ما هي عقوبة القتل غير العمد والإيذاء الناتج عن الإهمال وقلة الاحتراز؟", "domain": "criminal"},
    {"id": "TC10", "query": "موكلي ضُبط وبحوزته سيارة مسروقة ويدّعي أنه اشتراها بحسن نية. كيف يمكنني الدفاع عنه؟", "domain": "criminal"},
]


def print_table(summary: dict):
    print("\n" + "=" * 78)
    print(f"{'System':<16}{'N':>4}{'OK':>4}{'AvgScore':>10}{'AvgLat(s)':>11}{'AvgCost$':>10}")
    print("-" * 78)
    for system, row in summary.items():
        print(f"{system:<16}{row['n']:>4}{row['ok']:>4}"
              f"{str(row['avg_score'] or '-'):>10}"
              f"{str(row['avg_latency_s'] or '-'):>11}"
              f"{str(row['avg_cost_usd'] or '-'):>10}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation: multi-agent vs baselines")
    parser.add_argument("--systems", nargs="+",
                        default=["multi_agent", "single_agent", "no_rag"],
                        choices=["multi_agent", "single_agent", "no_rag"])
    parser.add_argument("--cases", type=str, default=None,
                        help="Path to a JSON list of {id, query, domain}. Defaults to built-in set.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cases")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8")) if args.cases else DEFAULT_CASES
    if args.limit:
        cases = cases[:args.limit]

    logger.info(f"Evaluating {len(cases)} cases across systems: {args.systems} "
                f"(judge={'on' if args.judge else 'off'})")

    score_fn = build_judge(args.judge_model) if args.judge else None

    def progress(system, i, total, tc):
        logger.info(f"[{system}] {tc.get('id', i+1)} ({i+1}/{total})")

    all_records = []
    for system in args.systems:
        logger.info(f"=== Running system: {system} ===")
        all_records += run_system(system, cases, score_fn=score_fn, progress=progress)

    summary = summarize(all_records)
    print_table(summary)

    out_path = Path(args.output or
                    f"experiments/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": all_records, "cases": cases,
               "systems": args.systems, "judge_model": args.judge_model if args.judge else None},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
