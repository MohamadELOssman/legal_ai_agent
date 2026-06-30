#!/usr/bin/env python3
"""Tune hybrid retrieval (BM25/dense weights) and report recall@5/@10 — CPU only, no API."""

import sys, json, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import LegalVectorStore

GOLD = json.load(open("experiments/qa_benchmark_200.json", encoding="utf-8"))["cases"]


def metrics(vs, k_eval=(5, 10)):
    out = {f"recall@{k}": [] for k in k_eval}
    out["mrr"] = []; out["ndcg@5"] = []
    for c in GOLD:
        gold = set(c["relevant_articles"])
        res = vs.search(query=c["query"], k=max(k_eval), strategy="hybrid",
                        use_reranking=False, score_threshold=0.0,
                        filter_dict={"source_type": "legal_code"})
        got = [str(x.metadata.get("article_number", "")) for x in res]
        for k in k_eval:
            top = got[:k]
            out[f"recall@{k}"].append(sum(1 for g in gold if g in top) / len(gold) if gold else 0)
        rr = next((1 / i for i, g in enumerate(got, 1) if g in gold), 0)
        out["mrr"].append(rr)
        dcg = sum(1 / math.log2(i + 1) for i, g in enumerate(got[:5], 1) if g in gold)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 5) + 1))
        out["ndcg@5"].append(dcg / idcg if idcg else 0)
    return {m: round(sum(v) / len(v), 3) for m, v in out.items()}


def main():
    vs = LegalVectorStore(); vs.load_vectorstore()
    # Pull all docs once to rebuild BM25/ensemble with different weights.
    coll = vs.vectorstore._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata=m or {}) for t, m in
            zip(coll["documents"], coll["metadatas"])]

    weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
    print(f"\n{'BM25/dense':<12}{'Recall@5':>10}{'Recall@10':>11}{'MRR':>8}{'nDCG@5':>9}")
    print("-" * 50)
    results = {}
    for w in weights:
        vs.build_hybrid_retriever(docs, weights=w)
        m = metrics(vs)
        results[f"{w[0]}/{w[1]}"] = m
        print(f"{w[0]}/{w[1]:<8}{m['recall@5']:>10}{m['recall@10']:>11}{m['mrr']:>8}{m['ndcg@5']:>9}")
    best = max(results, key=lambda k: results[k]["recall@10"])
    print("-" * 50)
    print(f"Best by Recall@10: {best} -> {results[best]}")
    json.dump(results, open("experiments/tune_retrieval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
