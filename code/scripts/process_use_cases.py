#!/usr/bin/env python3
"""
Process Lebanese court-ruling use cases — one complete document per case.

Design:
  - One vector-store entry per case (not one per section).
  - The stored text = a short structured summary (good for embedding within the
    model's 128-token window) followed by the full cleaned body (preserved in
    the Chroma SQLite store for the agent to read).
  - The Research agent does two separate searches:
      1. Legal codes  (filter source_type == "legal_code")  → relevant articles
      2. Court rulings (filter source_type == "court_ruling") → similar cases
  - This way the agent always gets complete case context, never a fragment.

Input:  data/processed_use_cases/full_documents/{name}_full.json
Output: data/processed_use_cases/vector_store_ready/all_documents.jsonl
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# ── helpers ──────────────────────────────────────────────────────────────────

def _flat(value) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(", ".join(str(v) for v in item.values()))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return "" if value is None else str(value)


def _articles_str(applicable_articles) -> str:
    if not applicable_articles:
        return ""
    parts = []
    for a in applicable_articles:
        if isinstance(a, dict):
            art  = a.get("article", "")
            para = a.get("paragraph", "")
            parts.append(f"{art}({para})" if para else art)
        else:
            parts.append(str(a))
    return ", ".join(parts)


def build_base_metadata(doc_meta: dict, pdf_name: str) -> dict:
    return {
        "source":              "lebanese_legal_corpus",
        "source_type":         "court_ruling",
        "document_language":   doc_meta.get("document_language", "ar"),
        "document_id":         pdf_name,
        "court":               doc_meta.get("court", ""),
        "chamber":             doc_meta.get("chamber", ""),
        "case_number":         doc_meta.get("case_number", ""),
        "decision_number":     doc_meta.get("decision_number", ""),
        "decision_date":       doc_meta.get("decision_date", ""),
        "case_type":           doc_meta.get("case_type", ""),
        "legal_domain":        _flat(doc_meta.get("legal_domain", [])),
        "applicable_articles": _articles_str(doc_meta.get("applicable_articles", [])),
        "applicable_laws":     _flat(doc_meta.get("applicable_laws", [])),
        "outcome":             doc_meta.get("outcome", ""),
        "sentence_original":   doc_meta.get("sentence_original", ""),
        "sentence_final":      doc_meta.get("sentence_final", ""),
        "processed_at":        datetime.now().isoformat(),
        "chunk_type":          "court_ruling",
    }


def build_case_document(full_doc_path: Path) -> dict:
    """
    Return a single vector-store entry for one court ruling.

    The text field contains:
      • A compact structured summary (≤ ~100 tokens) — this is what the
        multilingual embedding model encodes for similarity search.
      • The full cleaned_body — preserved verbatim in Chroma's SQLite store
        so the Research agent can return it to Analysis without losing context.
    """
    with open(full_doc_path, encoding="utf-8") as f:
        data = json.load(f)

    doc_meta     = data.get("document_metadata", {})
    cleaned_body = data.get("cleaned_body", "")
    pdf_name     = full_doc_path.stem.replace("_full", "")

    # Short summary — fits in the embedding model's 128-token window
    summary = (
        f"المحكمة: {doc_meta.get('court', '')} | "
        f"القضية: {doc_meta.get('case_number', '')} | "
        f"التاريخ: {doc_meta.get('decision_date', '')} | "
        f"النوع: {doc_meta.get('case_type', '')} | "
        f"المجال: {_flat(doc_meta.get('legal_domain', []))} | "
        f"النتيجة: {doc_meta.get('outcome', '')} | "
        f"العقوبة: {doc_meta.get('sentence_final', '')} | "
        f"المواد: {_articles_str(doc_meta.get('applicable_articles', []))}"
    )

    # Combined text: summary for embedding quality + full body for agent context
    text = summary + "\n\n---\n\n" + cleaned_body

    meta = build_base_metadata(doc_meta, pdf_name)

    logger.info(f"  {pdf_name}: 1 document ({len(cleaned_body)} chars body)")
    return {
        "chunk_id":   f"{pdf_name}_case",
        "chunk_type": "court_ruling",
        "text":       text,
        "metadata":   meta,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    project_root  = Path(__file__).parent.parent
    use_cases_dir = project_root / "data" / "use_cases"
    full_docs_dir = project_root / "data" / "processed_use_cases" / "full_documents"
    output_dir    = project_root / "data" / "processed_use_cases" / "vector_store_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROCESSING USE CASE COURT RULINGS  (one document per case)")
    logger.info("=" * 70)

    pdf_files = sorted(use_cases_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {use_cases_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) in {use_cases_dir}")

    documents = []
    skipped   = []

    for pdf in pdf_files:
        full_doc_path = full_docs_dir / f"{pdf.stem}_full.json"
        if full_doc_path.exists():
            logger.info(f"\n✓ {full_doc_path.name}")
            try:
                documents.append(build_case_document(full_doc_path))
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
        else:
            logger.warning(f"\n⚠ No pre-processed JSON for {pdf.name} — skipping")
            logger.warning(f"  Expected: {full_doc_path}")
            skipped.append(pdf.name)

    if not documents:
        logger.error("No documents produced. Exiting.")
        sys.exit(1)

    output_path = output_dir / "all_documents.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    summary = {
        "processing_date":   datetime.now().isoformat(),
        "total_documents":   len(documents),
        "pdfs_processed":    len(pdf_files) - len(skipped),
        "pdfs_skipped":      skipped,
        "strategy":          "one_complete_document_per_case",
        "master_file":       str(output_path),
    }
    with open(output_dir.parent / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE — {len(documents)} complete case documents")
    if skipped:
        logger.warning(f"Skipped: {', '.join(skipped)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)
    logger.info("\nNext: rebuild the vector store")
    logger.info("  python scripts/build_vectorstore.py")


if __name__ == "__main__":
    main()
