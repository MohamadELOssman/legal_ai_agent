#!/usr/bin/env python3
"""
Ablation study (Workstream D5).

Runs the multi-agent pipeline under several configurations to quantify each
component's contribution, scored on the verified gold set with the objective
citation metrics (precision/recall/F1) plus latency.

Variants (each a LegalAIPipeline config):
  full               — evidence-based defaults (semantic retrieval, reasoning on)
  no_reasoning       — skip the Reasoning agent (does CoT reasoning help?)
  enforce_grounding  — only corpus-grounded provisions reach the writer
  retrieval_hybrid   — hybrid (BM25+dense) instead of semantic

Note: each variant runs the full pipeline per query (~minutes/query), so use
--limit / --variants to size the run; this script is the harness and the user
runs the full study offline.

Usage:
  python scripts/run_ablation.py --limit 3
  python scripts/run_ablation.py --variants full no_reasoning --limit 5
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
from src.evaluation.comparison import extract_cited_articles, citation_metrics
from src.evaluation.stats import mean_ci

GOLD_FILE = Path("experiments/retrieval_gold.json")

VARIANTS = {
    "full": {},
    "no_reasoning": {"skip_reasoning": True},
    "enforce_grounding": {"enforce_grounding": True},
    "retrieval_hybrid": {"retrieval_strategy": "hybrid", "use_reranking": False},
}


def run_variant(name: str, kwargs: dict, cases: list, vs) -> list:
    from src.orchestrator.coordinator import LegalAIPipeline
    pipe = LegalAIPipeline(model=DEFAULT_MODEL, vectorstore=vs,
                           load_vectorstore=False, **kwargs)
    rows = []
    for i, tc in enumerate(cases):
        logger.info(f"[{name}] {tc.get('id', i+1)} ({i+1}/{len(cases)})")
        r = pipe.process_query(tc["query"])
        gold = set(tc.get("relevant_articles", []))
        cm = citation_metrics(extract_cited_articles(r.get("memorandum", "")), gold)
        rows.append({
            "id": tc.get("id"), "variant": name,
            "success": r.get("success"),
            "latency_s": r.get("total_latency_s"),
            "hallucination_rate": r.get("trust_report", {}).get("hallucination_rate"),
            "citation_f1": cm.get("citation_f1"),
            "citation_recall": cm.get("citation_recall"),
            "citation_precision": cm.get("citation_precision"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ablation study for the multi-agent pipeline")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cases = json.load(open(GOLD_FILE, encoding="utf-8"))["cases"][:args.limit]

    from src.rag.vectorstore import LegalVectorStore
    vs = LegalVectorStore(); vs.load_vectorstore()

    logger.info(f"Ablation: variants={args.variants}, cases={len(cases)}")

    all_rows = []
    for name in args.variants:
        all_rows += run_variant(name, VARIANTS[name], cases, vs)

    # Aggregate per variant.
    by_variant = defaultdict(list)
    for r in all_rows:
        by_variant[r["variant"]].append(r)

    report = {}
    for name, rows in by_variant.items():
        report[name] = {
            "citation_f1": mean_ci([r["citation_f1"] for r in rows if r["citation_f1"] is not None]),
            "citation_recall": mean_ci([r["citation_recall"] for r in rows if r["citation_recall"] is not None]),
            "latency_s": mean_ci([r["latency_s"] for r in rows if r["latency_s"] is not None]),
            "hallucination_rate": mean_ci([r["hallucination_rate"] for r in rows if r["hallucination_rate"] is not None]),
        }

    print("\n" + "=" * 86)
    print(f"ABLATION STUDY  ({len(cases)} queries)")
    print("-" * 86)
    print(f"{'Variant':<20}{'cit_F1':>16}{'cit_recall':>14}{'latency_s':>16}{'halluc.':>12}")
    print("-" * 86)
    for name, m in report.items():
        f1, rc, lat, hl = (m["citation_f1"], m["citation_recall"],
                           m["latency_s"], m["hallucination_rate"])
        f1_str = f"{f1.get('mean', '-')}±{f1.get('ci95', '-')}"
        rc_str = str(rc.get("mean", "-"))
        lat_str = str(lat.get("mean", "-"))
        hl_str = str(hl.get("mean", "-"))
        print(f"{name:<20}{f1_str:>16}{rc_str:>14}{lat_str:>16}{hl_str:>12}")
    print("=" * 86)

    out = Path(args.output or
               f"experiments/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"variants": args.variants, "report": report, "rows": all_rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved {out}")


if __name__ == "__main__":
    main()
