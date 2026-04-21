#!/usr/bin/env python3
"""
Build vector store from processed use case documents.
Run this after processing legal documents with process_use_cases.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import build_vectorstore_pipeline


def main():
    """Build vector store from processed use cases."""

    logger.info("=" * 80)
    logger.info("BUILDING VECTOR STORE FROM PROCESSED USE CASES")
    logger.info("=" * 80)

    # Check if chunks file exists
    chunks_file = Path("data/processed_use_cases/vector_store_ready/all_documents.jsonl")

    if not chunks_file.exists():
        logger.error(f"✗ Chunks file not found: {chunks_file}")
        logger.error("Please run preprocessing first:")
        logger.error("  python scripts/process_use_cases.py")
        sys.exit(1)

    # Count chunks
    with open(chunks_file, 'r') as f:
        num_chunks = sum(1 for line in f if line.strip())

    logger.info(f"Found {num_chunks} chunks in {chunks_file}")

    try:
        # Build vector store
        vectorstore = build_vectorstore_pipeline(
            chunks_file=str(chunks_file),
            persist_directory="data/vectorstore_use_cases"
        )

        logger.info("=" * 80)
        logger.info("✓ VECTOR STORE BUILD COMPLETE")
        logger.info("=" * 80)

        # Test search
        logger.info("\nTesting vector store with Arabic query...")
        test_queries = [
            "ما هي عقوبة السرقة في القانون اللبناني؟",
            "الظروف المخففة",
        ]

        for test_query in test_queries:
            logger.info(f"\nTest query: {test_query}")
            results = vectorstore.search(test_query, k=3, strategy="hybrid")

            logger.info(f"Retrieved {len(results)} documents:")

            for i, doc in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    Court: {doc.metadata.get('court', 'N/A')}")
                logger.info(f"    Case: {doc.metadata.get('case_number', 'N/A')}")
                logger.info(f"    Type: {doc.metadata.get('chunk_type', 'N/A')}")
                logger.info(f"    Preview: {doc.page_content[:150]}...")

        logger.info("\n" + "=" * 80)
        logger.info("✓ VECTOR STORE IS READY FOR USE!")
        logger.info("=" * 80)
        logger.info(f"\nVector store location: data/vectorstore_use_cases")
        logger.info(f"Total documents indexed: {num_chunks}")
        logger.info("\nYou can now use this vectorstore in your RAG pipeline!")

    except Exception as e:
        logger.error(f"✗ Vector store build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
