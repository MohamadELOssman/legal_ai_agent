#!/usr/bin/env python3
"""
Build vector store from the two canonical source files:
  data_processed/documents/panel_code_AR.json   — 434 penal-code articles (Arabic)
  data_processed/documents/use_cases.json        — 54 court rulings (preprocessed)

One chunk = one article / one ruling.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from langchain_core.documents import Document
from src.rag.vectorstore import LegalVectorStore


PANEL_CODE_FILE = Path("data_processed/documents/panel_code_AR.json")
USE_CASES_FILE  = Path("data_processed/documents/use_cases.json")
VECTORSTORE_DIR = "data_processed/vectorstore"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_articles() -> list[Document]:
    """Convert every article in panel_code_AR.json to a LangChain Document."""

    with open(PANEL_CODE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    doc_meta = data["document"]["metadata"]   # language, document_type, ...
    articles = data["articles"]
    documents = []

    for article in articles:
        status = article.get("status", "active")
        text   = article.get("text", "").strip()

        # Skip repealed articles and articles with no usable text
        if status == "repealed" or not text:
            logger.debug(f"Skipping article {article.get('article_number')} [{status}]")
            continue

        metadata = {
            "source_type":       "legal_code",
            "document_language": doc_meta.get("language", "ar"),
            "document_type":     doc_meta.get("document_type", "penal_code"),
            "article_number":    str(article.get("article_number", "")),
            "document_id":       article.get("id", ""),
            "book":              article.get("book",       "") or "",
            "chapter":           article.get("chapter",    "") or "",
            "section":           article.get("section",    "") or "",
            "subsection":        article.get("subsection", "") or "",
            "status":            status,
            "keywords":          ", ".join(article.get("keywords", [])),
            "topics":            ", ".join(article.get("topics",   [])),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Loaded {len(documents)} articles from {PANEL_CODE_FILE.name}")
    return documents


def load_rulings() -> list[Document]:
    """Convert every case in use_cases.json to a LangChain Document."""

    with open(USE_CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    documents = []

    for case in cases:
        court   = case.get("court", {})
        charges = case.get("charges", {})
        outcome = case.get("outcome", {})
        lr      = case.get("legal_reasoning", {})

        # Build rich page content: embedding_text + ratio + outcome sentence
        embedding_text = case.get("embedding_text", "").strip()
        ratio          = lr.get("ratio_decidendi", "")
        verdict        = outcome.get("verdict", "")
        sentence_final = outcome.get("sentence_final", "")

        content_parts = [embedding_text]
        if ratio:
            content_parts.append(f"Legal reasoning: {ratio[:600]}")
        if verdict or sentence_final:
            content_parts.append(f"Verdict: {verdict} | Sentence: {sentence_final}")

        page_content = "\n\n".join(p for p in content_parts if p)

        # Flatten applicable_articles to a comma-separated string
        applicable_articles_str = ", ".join(
            a.get("article", "") for a in charges.get("applicable_articles", [])
            if a.get("article")
        )

        # Flatten legal_domain list
        legal_domain_str = ", ".join(charges.get("legal_domain", []))

        # Flatten search_tags strings
        search_tags = case.get("search_tags", {})
        crime_types = search_tags.get("crime_type", [])
        legal_concepts = search_tags.get("legal_concepts", [])

        metadata = {
            "source_type":          "court_ruling",
            "document_language":    court.get("language", "ar"),
            "document_id":          case.get("case_id", ""),
            "case_number":          court.get("decision_number", ""),
            "decision_number":      court.get("decision_number", ""),
            "court":                court.get("name", ""),
            "chamber":              court.get("chamber", ""),
            "decision_date":        court.get("decision_date", ""),
            "outcome":              verdict,
            "sentence_final":       sentence_final,
            "applicable_articles":  applicable_articles_str,
            "legal_domain":         legal_domain_str,
            "case_type":            charges.get("case_type", ""),
            "crime_types":          ", ".join(crime_types),
            "legal_concepts":       ", ".join(legal_concepts),
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"Loaded {len(documents)} court rulings from {USE_CASES_FILE.name}")
    return documents


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("BUILDING VECTOR STORE")
    logger.info(f"  Articles : {PANEL_CODE_FILE}")
    logger.info(f"  Rulings  : {USE_CASES_FILE}")
    logger.info(f"  Output   : {VECTORSTORE_DIR}")
    logger.info("=" * 70)

    # Validate inputs
    for path in (PANEL_CODE_FILE, USE_CASES_FILE):
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)

    # Load documents
    articles = load_articles()
    rulings  = load_rulings()
    all_docs  = articles + rulings

    logger.info(f"\n✓ Total: {len(all_docs)} documents "
                f"({len(articles)} articles + {len(rulings)} rulings)")

    # Build vector store
    vs = LegalVectorStore(
        persist_directory=VECTORSTORE_DIR,
        embedding_provider="huggingface",
        use_reranking=True,
    )

    logger.info("\nEmbedding and indexing — this may take a few minutes...")
    vs.build_vectorstore(all_docs)
    vs.build_hybrid_retriever(all_docs, weights=(0.5, 0.5))

    logger.info("\n" + "=" * 70)
    logger.info("✓ VECTOR STORE BUILT SUCCESSFULLY")
    logger.info("=" * 70)

    # ── Quick smoke test ──────────────────────────────────────────────────────
    logger.info("\nSmoke tests:")

    test_cases = [
        ("عقوبة القتل العمد",          "hybrid",   {"source_type": "legal_code"},    2),
        ("premeditated murder filicide", "semantic", {"source_type": "court_ruling"}, 2),
        ("الظروف المخففة",              "hybrid",   None,                             3),
    ]

    for query, strategy, filt, k in test_cases:
        try:
            if filt and strategy == "hybrid":
                results = vs.search(query, k=k, strategy="semantic",
                                    filter_dict=filt, score_threshold=0.0)
            else:
                results = vs.search(query, k=k, strategy=strategy,
                                    filter_dict=filt, score_threshold=0.0)
            logger.info(f"  [{strategy}] '{query[:45]}' → {len(results)} docs")
            for doc in results[:1]:
                src = doc.metadata.get("source_type", "?")
                art = doc.metadata.get("article_number", "")
                cid = doc.metadata.get("document_id", "")
                label = f"art {art}" if src == "legal_code" else f"case {cid}"
                logger.info(f"    top: [{src}] {label}")
        except Exception as e:
            logger.warning(f"  Test failed: {e}")

    logger.info(f"\n✓ Vector store ready at: {VECTORSTORE_DIR}/")
    logger.info(f"  {len(articles)} penal-code articles  |  {len(rulings)} court rulings")


if __name__ == "__main__":
    main()
