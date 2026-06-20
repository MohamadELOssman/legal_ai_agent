#!/usr/bin/env python3
"""
Chunking study (Workstream B4).

Compares chunking strategies for the legal corpus on the gold set (mpnet,
semantic, no rerank). Legal articles are atomic citation units, so the
hypothesis is that article-level chunking beats fixed-size splitting that breaks
articles apart. This script tests that hypothesis rather than assuming it.

Strategies:
  article         — one chunk per article (current default)
  fixed_300_50    — RecursiveCharacterTextSplitter(300, overlap 50)
  fixed_600_100   — RecursiveCharacterTextSplitter(600, overlap 100)

Usage:
  python scripts/study_chunking.py
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.build_index import discover_code_files, load_code_file, load_rulings
from scripts.eval_retrieval import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

GOLD = Path("experiments/retrieval_gold.json")
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def base_documents():
    docs = []
    for p in discover_code_files():
        docs += load_code_file(p)
    return docs + load_rulings()


def rechunk(docs, size, overlap):
    """Split each document, preserving metadata (esp. article_number) per sub-chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for d in docs:
        for piece in splitter.split_text(d.page_content):
            out.append(Document(page_content=piece, metadata=dict(d.metadata)))
    return out


def score(docs, cases, k):
    from src.rag.vectorstore import LegalVectorStore
    with tempfile.TemporaryDirectory(prefix="chunk_study_") as tmp:
        vs = LegalVectorStore(persist_directory=tmp, embedding_provider="huggingface",
                              embedding_model=MODEL, use_reranking=False)
        vs.build_vectorstore(docs)
        rows = []
        for c in cases:
            gold = set(c["relevant_articles"])
            res = vs.search(query=c["query"], k=k, strategy="semantic",
                            use_reranking=False, score_threshold=0.0,
                            filter_dict={"source_type": "legal_code"})
            got = [str(x.metadata.get("article_number", "")).strip()
                   for x in res if x.metadata.get("article_number")]
            rows.append((precision_at_k(got, gold, k), recall_at_k(got, gold, k),
                         reciprocal_rank(got, gold), ndcg_at_k(got, gold, k)))
    n = len(rows)
    return {"precision@k": round(sum(r[0] for r in rows)/n, 3),
            "recall@k": round(sum(r[1] for r in rows)/n, 3),
            "mrr": round(sum(r[2] for r in rows)/n, 3),
            "ndcg@k": round(sum(r[3] for r in rows)/n, 3)}


def main():
    k = 5
    cases = json.load(open(GOLD, encoding="utf-8"))["cases"]
    base = base_documents()
    logger.info(f"Chunking study on {len(base)} article-level docs / {len(cases)} queries")

    strategies = {
        "article": base,
        "fixed_300_50": rechunk(base, 300, 50),
        "fixed_600_100": rechunk(base, 600, 100),
    }
    results = {}
    for name, docs in strategies.items():
        logger.info(f"=== {name}: {len(docs)} chunks ===")
        results[name] = {"chunks": len(docs), **score(docs, cases, k)}

    print("\n" + "=" * 70)
    print(f"CHUNKING STUDY  (mpnet, semantic, k={k}, {len(cases)} queries)")
    print("-" * 70)
    print(f"{'Strategy':<16}{'Chunks':>8}{'P@k':>8}{'R@k':>8}{'MRR':>8}{'nDCG':>8}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ndcg@k"], reverse=True):
        print(f"{name:<16}{r['chunks']:>8}{r['precision@k']:>8}{r['recall@k']:>8}"
              f"{r['mrr']:>8}{r['ndcg@k']:>8}")
    print("=" * 70)

    out = Path(f"experiments/chunking_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"k": k, "results": results}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved {out}")


if __name__ == "__main__":
    main()
