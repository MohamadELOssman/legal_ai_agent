#!/usr/bin/env python3
"""
Embedding-model study (Workstream B3).

Rebuilds the index under each candidate embedding model in a temporary directory
and scores retrieval on the verified gold set (experiments/retrieval_gold.json)
using the winning config (semantic, no reranking). Reuses the ingestion loaders
(build_index) and IR metrics (eval_retrieval) so results are comparable to the
main pipeline.

Note: intfloat/multilingual-e5-* models expect "query:" / "passage:" prefixes for
best results; we report a caveat for those rather than silently under-using them.

Usage:
  python scripts/study_embeddings.py            # default candidate set
  python scripts/study_embeddings.py --models sentence-transformers/LaBSE
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from scripts.build_index import discover_code_files, load_code_file, load_rulings
from scripts.eval_retrieval import (
    precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k,
)

GOLD_FILE = Path("experiments/retrieval_gold.json")

DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # current baseline
    "sentence-transformers/LaBSE",                                  # 109 languages
    "intfloat/multilingual-e5-large",                              # strong multilingual
]

E5_PREFIX_MODELS = ("e5-",)  # models that benefit from query:/passage: prefixes


def build_documents():
    code_docs = []
    for path in discover_code_files():
        code_docs += load_code_file(path)
    return code_docs + load_rulings()


def evaluate_model(model_name: str, docs, cases, k: int) -> dict:
    """Build a temp index with this embedding model and score it on the gold set."""
    from src.rag.vectorstore import LegalVectorStore

    needs_prefix = any(tag in model_name for tag in E5_PREFIX_MODELS)
    with tempfile.TemporaryDirectory(prefix="embed_study_") as tmp:
        vs = LegalVectorStore(persist_directory=tmp,
                              embedding_provider="huggingface",
                              embedding_model=model_name,
                              use_reranking=False)
        vs.build_vectorstore(docs)

        rows = []
        for c in cases:
            gold = set(c["relevant_articles"])
            results = vs.search(query=c["query"], k=k, strategy="semantic",
                                use_reranking=False, score_threshold=0.0,
                                filter_dict={"source_type": "legal_code"})
            got = [str(d.metadata.get("article_number", "")).strip()
                   for d in results if d.metadata.get("article_number")]
            rows.append({
                "p@k": precision_at_k(got, gold, k),
                "r@k": recall_at_k(got, gold, k),
                "mrr": reciprocal_rank(got, gold),
                "ndcg": ndcg_at_k(got, gold, k),
            })

    n = len(rows)
    return {
        "model": model_name,
        "needs_prefix_caveat": needs_prefix,
        "precision@k": round(sum(r["p@k"] for r in rows) / n, 3),
        "recall@k": round(sum(r["r@k"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "ndcg@k": round(sum(r["ndcg"] for r in rows) / n, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Embedding-model retrieval study")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cases = json.load(open(GOLD_FILE, encoding="utf-8"))["cases"]
    docs = build_documents()
    logger.info(f"Study: {len(args.models)} models on {len(docs)} docs / "
                f"{len(cases)} gold queries @k={args.k}")

    results = []
    for model in args.models:
        logger.info(f"=== {model} (downloading if needed; re-embedding {len(docs)} docs) ===")
        try:
            results.append(evaluate_model(model, docs, cases, args.k))
        except Exception as e:
            logger.warning(f"  {model} failed: {e}")

    print("\n" + "=" * 92)
    print(f"EMBEDDING STUDY  (semantic, no rerank, k={args.k}, {len(cases)} queries)")
    print("-" * 92)
    print(f"{'Model':<58}{'P@k':>7}{'R@k':>7}{'MRR':>7}{'nDCG':>7}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: x["ndcg@k"], reverse=True):
        tag = " *" if r["needs_prefix_caveat"] else ""
        print(f"{r['model'][:56]:<58}{r['precision@k']:>7}{r['recall@k']:>7}"
              f"{r['mrr']:>7}{r['ndcg@k']:>7}{tag}")
    print("=" * 92)
    if any(r["needs_prefix_caveat"] for r in results):
        print("* e5 models tested without query:/passage: prefixes — may understate their quality.")

    out = Path(args.output or
               f"experiments/embedding_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"k": args.k, "n_queries": len(cases), "n_docs": len(docs),
               "results": results}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved {out}")


if __name__ == "__main__":
    main()
