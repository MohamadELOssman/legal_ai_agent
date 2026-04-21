#!/usr/bin/env python3
"""
Script to build the vector store from processed documents.
Run this after preprocessing the legal documents.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import build_vectorstore_pipeline

def main():
    """Build vector store."""
    logger.info("=" * 80)
    logger.info("BUILDING VECTOR STORE")
    logger.info("=" * 80)

    try:
        # Build vector store
        vectorstore = build_vectorstore_pipeline()

        logger.info("=" * 80)
        logger.info("✓ VECTOR STORE BUILD COMPLETE")
        logger.info("=" * 80)

        # Test search
        logger.info("Testing vector store...")
        test_query = "المسؤولية المدنية"
        results = vectorstore.search(test_query, k=3, strategy="hybrid")

        logger.info(f"Test query: {test_query}")
        logger.info(f"Retrieved {len(results)} documents")

        for i, doc in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"  Article: {doc.metadata.get('article_number', 'N/A')}")
            logger.info(f"  Language: {doc.metadata.get('language', 'N/A')}")
            logger.info(f"  Preview: {doc.page_content[:200]}...")

        logger.success("✓ Vector store is ready for use!")

    except Exception as e:
        logger.error(f"✗ Vector store build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
