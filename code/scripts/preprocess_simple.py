#!/usr/bin/env python3
"""
Simple preprocessing script - processes all PDFs from data/raw_documents/
Automatically detects language, no need for language-specific folders
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import LegalDocumentProcessor
from loguru import logger


def main():
    """Process all PDFs from data/raw_documents/ (simpler approach)"""

    logger.info("=" * 60)
    logger.info("Simple Legal Document Preprocessing")
    logger.info("=" * 60)

    # Use raw_documents folder (simpler structure)
    raw_docs_dir = project_root / "data" / "raw_documents"

    if not raw_docs_dir.exists():
        logger.error(f"Directory not found: {raw_docs_dir}")
        logger.info("Creating directory...")
        raw_docs_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = list(raw_docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_docs_dir}")
        logger.info("")
        logger.info("📚 How to add documents:")
        logger.info(f"   1. Place your PDFs in: {raw_docs_dir}")
        logger.info("   2. Run this script again")
        logger.info("")
        logger.info("Example:")
        logger.info(f"   cp your_law.pdf {raw_docs_dir}/")
        logger.info("   python scripts/preprocess_simple.py")
        return

    logger.info(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        logger.info(f"  - {pdf.name}")

    logger.info("")
    logger.info("Processing documents...")
    logger.info("")

    # Initialize processor
    processor = LegalDocumentProcessor(data_dir=str(raw_docs_dir.parent))

    all_chunks = []

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            # Extract document
            document = processor.extract_pdf(pdf_file)

            # Chunk document
            chunks = processor.chunk_document(
                document,
                chunk_size=1000,
                chunk_overlap=200,
                strategy="fixed"
            )

            all_chunks.extend(chunks)
            logger.info(f"✓ {pdf_file.name}: {len(chunks)} chunks ({document.language})")

        except Exception as e:
            logger.error(f"✗ Error processing {pdf_file.name}: {e}")

    # Save all chunks
    if all_chunks:
        processor._save_chunks(all_chunks)
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✅ Success! Created {len(all_chunks)} chunks")
        logger.info("=" * 60)

        # Show language distribution
        lang_dist = {}
        for chunk in all_chunks:
            lang_dist[chunk.language] = lang_dist.get(chunk.language, 0) + 1

        logger.info("")
        logger.info("Language distribution:")
        for lang, count in sorted(lang_dist.items()):
            logger.info(f"  {lang}: {count} chunks")

        logger.info("")
        logger.info("📁 Output saved to:")
        logger.info(f"   data_processed/documents/processed_chunks.json")

        logger.info("")
        logger.info("🚀 Next step:")
        logger.info("   python scripts/build_vectorstore.py")

    else:
        logger.warning("No chunks created. Check your PDFs.")


if __name__ == "__main__":
    main()
