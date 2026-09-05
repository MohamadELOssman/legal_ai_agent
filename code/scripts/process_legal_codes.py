#!/usr/bin/env python3
"""
Process Lebanese Legal Code PDFs (Penal Code, Code of Obligations, etc.)
Uses article-based chunking — NOT the vision AI agent.
Designed for long structured PDFs like legal codes, not scanned court rulings.

Usage:
    python scripts/process_legal_codes.py                  # process all language dirs
    python scripts/process_legal_codes.py --lang ENG       # only English
    python scripts/process_legal_codes.py --lang AR FR     # Arabic and French
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data.preprocess import LegalDocumentProcessor


# Maps folder name → ISO language code and human label
LANG_DIRS = {
    "AR":  {"lang_code": "ar",  "label": "Arabic"},
    "FR":  {"lang_code": "fr",  "label": "French"},
    "ENG": {"lang_code": "en",  "label": "English"},
}

# Maps document type → source_type string stored in metadata
DOC_TYPE_SOURCE = {
    "code_obligations_contracts": "legal_code",
    "penal_code": "legal_code",
    "criminal_procedure": "legal_code",
    "legal_guide": "legal_guide",
    "legal_document": "legal_code",
    "case_law": "court_ruling",
}


def build_enriched_chunks(processor: LegalDocumentProcessor, pdf_path: Path, lang_dir: str) -> list:
    """Extract and chunk a legal code PDF, returning enriched JSONL-ready dicts."""
    logger.info(f"Processing: {pdf_path.name}")

    document = processor.extract_pdf(pdf_path)
    chunks = processor.chunk_document(document, chunk_size=1000, chunk_overlap=200, strategy="article")

    lang_info = LANG_DIRS.get(lang_dir, {"lang_code": document.language, "label": lang_dir})
    source_type = DOC_TYPE_SOURCE.get(document.document_type, "legal_code")
    timestamp = datetime.now().isoformat()

    enriched = []
    for chunk in chunks:
        article_number = chunk.article_number or ""
        # Build a clean article number string without sub-chunk suffix for display
        article_display = article_number.split(".")[0] if article_number else ""

        metadata = {
            # Identity
            "document_id": pdf_path.stem,
            "file_name": pdf_path.name,
            "source": "lebanese_legal_corpus",
            "source_type": source_type,                     # "legal_code" | "court_ruling"
            "document_type": document.document_type,        # "penal_code", "code_obligations_contracts", …
            "document_language": lang_info["lang_code"],
            # Article info (filterable)
            "article_number": article_display,
            "chunk_index": chunk.chunk_index,
            # Retrieval hints
            "chunk_purpose": "article_retrieval",
            "processed_at": timestamp,
        }

        enriched.append({
            "chunk_id": f"{pdf_path.stem}_art{article_display}_idx{chunk.chunk_index}",
            "chunk_type": "legal_article",
            "text": chunk.text,
            "metadata": metadata,
        })

    logger.info(f"  ✓ {pdf_path.name}: {len(enriched)} article chunks ({document.document_type})")
    return enriched


def process_language_dir(lang_dir_name: str, data_root: Path) -> list:
    """Process all PDFs inside data/<LANG_DIR>/."""
    lang_dir = data_root / lang_dir_name
    if not lang_dir.exists():
        logger.warning(f"Directory not found: {lang_dir}  (skipping)")
        return []

    pdf_files = sorted(lang_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {lang_dir}  (skipping)")
        return []

    logger.info(f"\nProcessing {lang_dir_name}: {len(pdf_files)} PDF(s)")

    processor = LegalDocumentProcessor(data_dir=str(data_root))
    all_chunks = []

    for pdf_path in pdf_files:
        try:
            chunks = build_enriched_chunks(processor, pdf_path, lang_dir_name)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"  ✗ Failed to process {pdf_path.name}: {e}")

    return all_chunks


def save_chunks(chunks: list, output_dir: Path, lang_dir_name: str):
    """Save chunks to JSONL in data/processed_<LANG>/vector_store_ready/all_documents.jsonl."""
    vector_dir = output_dir / f"processed_{lang_dir_name}" / "vector_store_ready"
    vector_dir.mkdir(parents=True, exist_ok=True)

    output_path = vector_dir / "all_documents.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "processing_date": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "master_file": str(output_path),
        "document_types": list({c["metadata"]["document_type"] for c in chunks}),
    }
    summary_path = output_dir / f"processed_{lang_dir_name}" / "processing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved {len(chunks)} chunks → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Process Lebanese legal code PDFs by article")
    parser.add_argument(
        "--lang", nargs="+", choices=list(LANG_DIRS.keys()),
        default=list(LANG_DIRS.keys()),
        help="Language directories to process (default: all)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = project_root / "data"

    logger.info("=" * 70)
    logger.info("PROCESSING LEGAL CODE PDFs (article-based chunking)")
    logger.info("=" * 70)
    logger.info(f"Language directories: {', '.join(args.lang)}")

    total_all = 0
    for lang_dir_name in args.lang:
        chunks = process_language_dir(lang_dir_name, data_root)
        if chunks:
            save_chunks(chunks, data_root, lang_dir_name)
            total_all += len(chunks)

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE — {total_all} total article chunks created")
    logger.info("=" * 70)
    logger.info("\nNext step: build the vector store")
    logger.info("  python scripts/build_vectorstore.py")


if __name__ == "__main__":
    main()
