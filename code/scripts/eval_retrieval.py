#!/usr/bin/env python3
"""
Retrieval evaluation harness (Workstream B2).

Computes standard IR metrics — precision@k, recall@k, MRR, nDCG@k — for the
Research retrieval layer against a verified ground-truth set
(experiments/retrieval_gold.json), and compares retrieval configurations so the
default strategy is chosen from evidence rather than assumption.

Relevance is judged by ARTICLE NUMBER (language-agnostic): a retrieved chunk is
relevant if its article_number is in the query's gold set — so an English query
that correctly retrieves the Arabic article still scores as a hit.

Usage:
  python scripts/eval_retrieval.py                 # compare default config matrix
  python scripts/eval_retrieval.py --k 5
"""

import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

GOLD_FILE = Path("experiments/retrieval_gold.json")

# Configurations to compare: (label, strategy, use_reranking)
CONFIG_MATRIX = [
    ("semantic",            "semantic", False),
    ("semantic+rerank",     "semantic", True),
    ("hybrid",              "hybrid",   False),
    ("hybrid+rerank",       "hybrid",   True),
    ("bm25",                "bm25",     False),
]


# ── Metrics ────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in set(top) if r in relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieved_article_numbers(vs, query, k, strategy, use_reranking):
    """Return the ordered list of article numbers for a query under one config."""
    docs = vs.search(query=query, k=k, strategy=strategy,
                     use_reranking=use_reranking, score_threshold=0.0,
                     filter_dict={"source_type": "legal_code"})
    nums = []
    for d in docs:
        num = str(d.metadata.get("article_number", "")).strip()
        if num:
            nums.append(num)
    return nums


def evaluate_config(vs, cases, label, strategy, use_reranking, k):
    rows = []
    for c in cases:
        gold = set(c["relevant_articles"])
        got = retrieved_article_numbers(vs, c["query"], k, strategy, use_reranking)
        rows.append({
            "id": c["id"], "lang": c.get("lang"),
            "retrieved": got, "gold": sorted(gold),
            "p@k": precision_at_k(got, gold, k),
            "r@k": recall_at_k(got, gold, k),
            "mrr": reciprocal_rank(got, gold),
            "ndcg": ndcg_at_k(got, gold, k),
        })
    n = len(rows)
    agg = {
        "config": label, "strategy": strategy, "rerank": use_reranking,
        "precision@k": round(sum(r["p@k"] for r in rows) / n, 3),
        "recall@k": round(sum(r["r@k"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "ndcg@k": round(sum(r["ndcg"] for r in rows) / n, 3),
    }
    return agg, rows


def main():
    parser = argparse.ArgumentParser(description="Retrieval metrics across configs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gold", type=str, default=str(GOLD_FILE))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cases = json.load(open(args.gold, encoding="utf-8"))["cases"]
    logger.info(f"Loaded {len(cases)} gold queries; evaluating @k={args.k}")

    from src.rag.vectorstore import LegalVectorStore
    vs = LegalVectorStore()
    vs.load_vectorstore()

    aggregates, details = [], {}
    for label, strategy, rerank in CONFIG_MATRIX:
        logger.info(f"Evaluating config: {label}")
        try:
            agg, rows = evaluate_config(vs, cases, label, strategy, rerank, args.k)
        except Exception as e:
            logger.warning(f"  {label} failed: {e}")
            continue
        aggregates.append(agg)
        details[label] = rows

    # Print comparison table.
    print("\n" + "=" * 72)
    print(f"RETRIEVAL METRICS  (k={args.k}, {len(cases)} queries)")
    print("-" * 72)
    print(f"{'Config':<18}{'P@k':>8}{'R@k':>8}{'MRR':>8}{'nDCG@k':>9}")
    print("-" * 72)
    for a in sorted(aggregates, key=lambda x: x["ndcg@k"], reverse=True):
        print(f"{a['config']:<18}{a['precision@k']:>8}{a['recall@k']:>8}"
              f"{a['mrr']:>8}{a['ndcg@k']:>9}")
    print("=" * 72)
    if aggregates:
        best = max(aggregates, key=lambda x: x["ndcg@k"])
        print(f"Best by nDCG@k: {best['config']}\n")

    out = Path(args.output or
               f"experiments/retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"k": args.k, "n_queries": len(cases), "aggregates": aggregates,
               "details": details}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"✓ Results saved to {out}")


if __name__ == "__main__":
    main()
