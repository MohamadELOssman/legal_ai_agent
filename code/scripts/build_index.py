#!/usr/bin/env python3
"""
Canonical, reproducible corpus ingestion pipeline.

Discovers every processed source under data_processed/documents/, builds the
Chroma vector store (+ BM25 hybrid retriever), regenerates the citation article
index, and writes a manifest describing exactly what was indexed.

Sources discovered automatically:
  • Legal codes  — any *.json with the {"document": {...}, "articles": [...]}
                   schema (e.g. panel_code_AR.json, penal_code_EN.json, and any
                   future code_obligations_*.json / *_FR.json you add).
  • Court rulings — use_cases.json (a JSON list of ruling objects).

Drop a new processed code file in data_processed/documents/ using the same
schema and it is indexed on the next run — no code changes needed.

Usage:
  python scripts/build_index.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from langchain_core.documents import Document

from src.rag.vectorstore import LegalVectorStore

DOCS_DIR = Path("data_processed/documents")
USE_CASES_FILE = DOCS_DIR / "use_cases.json"
VECTORSTORE_DIR = "data_processed/vectorstore"
ARTICLES_INDEX = Path("data_processed/articles_index.json")
MANIFEST = Path("data_processed/index_manifest.json")

MAX_ARTICLE = 770  # plausibility bound for penal-code article numbers


# ── Discovery ────────────────────────────────────────────────────────────────────

def discover_code_files() -> list[Path]:
    """Find legal-code JSON files (schema: dict with 'document' + 'articles')."""
    found = []
    for path in sorted(DOCS_DIR.glob("*.json")):
        if path.name == USE_CASES_FILE.name:
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "articles" in data and "document" in data:
            found.append(path)
    return found


# ── Loaders ──────────────────────────────────────────────────────────────────────

def load_code_file(path: Path) -> list[Document]:
    """Load one legal-code file into LangChain Documents with unified metadata."""
    data = json.load(open(path, encoding="utf-8"))
    doc_meta = data["document"].get("metadata", {})
    language = doc_meta.get("language", "ar")
    doc_type = doc_meta.get("document_type", "penal_code")

    documents = []
    for article in data["articles"]:
        if article.get("status") == "repealed":
            continue
        text = (article.get("text") or "").strip()
        if not text:
            continue

        metadata = {
            "source_type": "legal_code",
            "document_language": language,
            "document_type": doc_type,
            "article_number": str(article.get("article_number", "")),
            "document_id": article.get("id", ""),
            "status": article.get("status", "active"),
            "source_file": path.name,  # provenance
        }
        # Optional structural / enrichment fields (present in the Arabic code).
        for key in ("book", "chapter", "section", "subsection"):
            if article.get(key):
                metadata[key] = article[key]
        for key in ("keywords", "topics"):
            val = article.get(key)
            if isinstance(val, list) and val:
                metadata[key] = ", ".join(val)
            elif isinstance(val, str) and val:
                metadata[key] = val

        documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f"  {path.name}: {len(documents)} articles  [{language}/{doc_type}]")
    return documents


def load_rulings() -> list[Document]:
    """Convert each case in use_cases.json to a LangChain Document."""
    if not USE_CASES_FILE.exists():
        logger.warning(f"{USE_CASES_FILE} not found — skipping rulings")
        return []

    cases = json.load(open(USE_CASES_FILE, encoding="utf-8"))
    documents = []
    for case in cases:
        court = case.get("court", {})
        charges = case.get("charges", {})
        outcome = case.get("outcome", {})
        lr = case.get("legal_reasoning", {})

        embedding_text = (case.get("embedding_text") or "").strip()
        ratio = lr.get("ratio_decidendi", "")
        verdict = outcome.get("verdict", "")
        sentence_final = outcome.get("sentence_final", "")

        parts = [embedding_text]
        if ratio:
            parts.append(f"Legal reasoning: {ratio[:600]}")
        if verdict or sentence_final:
            parts.append(f"Verdict: {verdict} | Sentence: {sentence_final}")
        page_content = "\n\n".join(p for p in parts if p)
        if not page_content.strip():
            continue

        applicable = ", ".join(
            a.get("article", "") for a in charges.get("applicable_articles", []) if a.get("article")
        )
        tags = case.get("search_tags", {})

        metadata = {
            "source_type": "court_ruling",
            "document_language": court.get("language", "ar"),
            "document_id": case.get("case_id", ""),
            "case_number": court.get("decision_number", ""),
            "decision_number": court.get("decision_number", ""),
            "court": court.get("name", ""),
            "chamber": court.get("chamber", ""),
            "decision_date": court.get("decision_date", ""),
            "outcome": verdict,
            "sentence_final": sentence_final,
            "applicable_articles": applicable,
            "legal_domain": ", ".join(charges.get("legal_domain", [])),
            "case_type": charges.get("case_type", ""),
            "crime_types": ", ".join(tags.get("crime_type", [])),
            "legal_concepts": ", ".join(tags.get("legal_concepts", [])),
            "source_file": USE_CASES_FILE.name,
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"  {USE_CASES_FILE.name}: {len(documents)} rulings")
    return documents


# ── Article index (citation ground truth) ────────────────────────────────────────

def build_articles_index(code_docs: list[Document], rulings: list[Document]) -> dict:
    """Article numbers per document_type, merged across languages."""
    import re
    index: dict[str, set] = {}
    for d in code_docs:
        dtype = d.metadata.get("document_type", "penal_code")
        m = re.search(r"\d+", d.metadata.get("article_number", ""))
        if m and int(m.group(0)) <= MAX_ARTICLE:
            index.setdefault(dtype, set()).add(m.group(0))
    # Include article numbers cited by rulings (still penal-code references).
    for d in rulings:
        for tok in d.metadata.get("applicable_articles", "").split(","):
            m = re.search(r"\d+", tok)
            if m and int(m.group(0)) <= MAX_ARTICLE:
                index.setdefault("penal_code", set()).add(m.group(0))

    serializable = {k: sorted(v, key=int) for k, v in index.items()}
    json.dump(serializable, open(ARTICLES_INDEX, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"✓ Article index → {ARTICLES_INDEX} "
                f"({ {k: len(v) for k, v in serializable.items()} })")
    return serializable


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("CORPUS INGESTION PIPELINE")
    logger.info("=" * 70)

    code_files = discover_code_files()
    if not code_files:
        logger.error(f"No legal-code files found in {DOCS_DIR}/")
        sys.exit(1)
    logger.info(f"Discovered {len(code_files)} legal-code file(s): "
                f"{[p.name for p in code_files]}")

    code_docs: list[Document] = []
    per_source = {}
    for path in code_files:
        docs = load_code_file(path)
        code_docs += docs
        per_source[path.name] = len(docs)

    rulings = load_rulings()
    per_source[USE_CASES_FILE.name] = len(rulings)
    all_docs = code_docs + rulings

    # Language / type breakdown for the manifest.
    by_language, by_type = {}, {}
    for d in all_docs:
        by_language[d.metadata.get("document_language", "?")] = \
            by_language.get(d.metadata.get("document_language", "?"), 0) + 1
        key = d.metadata.get("document_type", d.metadata.get("source_type", "?"))
        by_type[key] = by_type.get(key, 0) + 1

    logger.info(f"\n✓ Total: {len(all_docs)} documents "
                f"({len(code_docs)} articles + {len(rulings)} rulings)")
    logger.info(f"  by language: {by_language}")
    logger.info(f"  by type:     {by_type}")

    # Build vector store (rebuilds from scratch for reproducibility).
    vs = LegalVectorStore(persist_directory=VECTORSTORE_DIR,
                          embedding_provider="huggingface", use_reranking=True)
    logger.info("\nEmbedding and indexing — this may take a few minutes...")
    vs.build_vectorstore(all_docs)
    vs.build_hybrid_retriever(all_docs, weights=(0.5, 0.5))

    index = build_articles_index(code_docs, rulings)

    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "total_documents": len(all_docs),
        "articles": len(code_docs),
        "rulings": len(rulings),
        "per_source": per_source,
        "by_language": by_language,
        "by_type": by_type,
        "article_index_counts": {k: len(v) for k, v in index.items()},
        "vectorstore_dir": VECTORSTORE_DIR,
    }
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✓ Manifest → {MANIFEST}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ INGESTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
