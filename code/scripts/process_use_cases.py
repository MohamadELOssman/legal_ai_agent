"""
Process Use Case Documents
Quick script to process the example scanned legal rulings
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.process_legal_documents import DocumentBatchProcessor
from loguru import logger


def main():
    """Process the use case documents."""

    logger.info("=" * 70)
    logger.info("PROCESSING USE CASE DOCUMENTS (Scanned Lebanese Legal Rulings)")
    logger.info("=" * 70)

    # Initialize processor for use_cases directory
    processor = DocumentBatchProcessor(
        raw_documents_dir="data/use_cases",
        output_dir="data/processed_use_cases",
        model="claude-sonnet-4-20250514",  # Best quality for thesis examples
    )

    # Get files
    files = processor.get_document_files()

    if not files:
        logger.error("No documents found in data/use_cases/")
        logger.info("Expected to find scanned PDF files there.")
        return 1

    logger.info(f"\nFound {len(files)} scanned PDF documents to process")
    for f in files:
        logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Confirm processing
    logger.info("\n" + "=" * 70)
    logger.info("These are SCANNED documents - will use Claude Vision for OCR")
    logger.info("Estimated cost: ~$0.05-$0.15 per document")
    logger.info(f"Total estimated cost: ~${len(files) * 0.10:.2f}")
    logger.info("=" * 70)

    proceed = input("\nProceed with processing? [y/N]: ").strip().lower()

    if proceed != 'y':
        logger.info("Cancelled by user")
        return 0

    # Process all documents
    logger.info("\nStarting processing...")
    summary = processor.process_all()

    # Print detailed results
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\nResults:")
    logger.info(f"  Total files: {summary['total_files']}")
    logger.info(f"  Processed successfully: {summary['processed']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Total chunks created: {summary['total_chunks']}")

    logger.info(f"\nOutputs saved to: data/processed_use_cases/")
    logger.info(f"  - Full documents: data/processed_use_cases/full_documents/")
    logger.info(f"  - Summaries: data/processed_use_cases/summaries/")
    logger.info(f"  - Ratios: data/processed_use_cases/ratios/")
    logger.info(f"  - Vector store ready: data/processed_use_cases/vector_store_ready/")

    logger.info(f"\nMaster JSONL file (use this for vector store):")
    logger.info(f"  {summary['master_file']}")

    # Print individual results
    if summary['results']:
        logger.info("\n" + "-" * 70)
        logger.info("Individual Document Results:")
        logger.info("-" * 70)

        for i, result in enumerate(summary['results'], 1):
            file_name = Path(result['file']).name
            if result['success']:
                logger.info(f"\n{i}. ✓ {file_name}")
                logger.info(f"   Document ID: {result['document_id']}")
                logger.info(f"   Chunks created: {result['chunks_created']}")
            else:
                logger.error(f"\n{i}. ✗ {file_name}")
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Next steps:")
    logger.info("1. Review outputs in data/processed_use_cases/")
    logger.info("2. Check OCR quality and confidence scores")
    logger.info("3. Load chunks into vector store:")
    logger.info("   python scripts/build_vectorstore.py --input data/processed_use_cases/vector_store_ready/all_documents.jsonl")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
