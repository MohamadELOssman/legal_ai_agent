#!/usr/bin/env python3
"""
Build vector store from ALL processed documents.
Combines use cases, AR documents, FR documents, ENG documents, etc.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vectorstore import build_vectorstore_pipeline


def find_all_chunk_files():
    """Find all available chunk files across different directories."""

    possible_locations = [
        "data/processed_use_cases/vector_store_ready/all_documents.jsonl",
        "data/processed_documents/vector_store_ready/all_documents.jsonl",
        "data/processed_AR/vector_store_ready/all_documents.jsonl",
        "data/processed_FR/vector_store_ready/all_documents.jsonl",
        "data/processed_ENG/vector_store_ready/all_documents.jsonl",
    ]

    found_files = []

    for location in possible_locations:
        path = Path(location)
        if path.exists():
            # Count chunks
            with open(path, 'r') as f:
                num_chunks = sum(1 for line in f if line.strip())

            found_files.append({
                "path": str(path),
                "chunks": num_chunks,
                "name": path.parent.parent.name
            })

    return found_files


def main():
    """Build vector store from all processed documents."""

    logger.info("=" * 80)
    logger.info("BUILDING VECTOR STORE FROM ALL PROCESSED DOCUMENTS")
    logger.info("=" * 80)

    # Find all chunk files
    chunk_files = find_all_chunk_files()

    if not chunk_files:
        logger.error("✗ No processed documents found!")
        logger.error("\nPlease run preprocessing first:")
        logger.error("  python scripts/process_use_cases.py")
        logger.error("  python scripts/process_legal_documents.py --input-dir data/AR")
        logger.error("  python scripts/process_legal_documents.py --input-dir data/FR")
        sys.exit(1)

    # Show what was found
    logger.info("\nFound processed documents:")
    total_chunks = 0
    for file_info in chunk_files:
        logger.info(f"  ✓ {file_info['name']}: {file_info['chunks']} chunks")
        logger.info(f"    → {file_info['path']}")
        total_chunks += file_info['chunks']

    logger.info(f"\n✓ Total: {total_chunks} chunks from {len(chunk_files)} source(s)")

    # Build vector store
    try:
        logger.info("\nBuilding vector store...")

        chunk_paths = [f["path"] for f in chunk_files]

        vectorstore = build_vectorstore_pipeline(
            chunks_files=chunk_paths,
            persist_directory="data/vectorstore"
        )

        logger.info("=" * 80)
        logger.info("✓ VECTOR STORE BUILD COMPLETE")
        logger.info("=" * 80)

        # Test search with multiple languages
        logger.info("\nTesting vector store...")

        test_queries = [
            ("ما هي عقوبة السرقة؟", "Arabic"),
            ("Quelle est la peine pour le vol?", "French"),
            ("What is the penalty for theft?", "English"),
        ]

        for query, lang in test_queries:
            logger.info(f"\nTest query ({lang}): {query}")

            try:
                results = vectorstore.search(query, k=2, strategy="hybrid")
                logger.info(f"  ✓ Retrieved {len(results)} documents")

                if results:
                    doc = results[0]
                    logger.info(f"    Top result:")
                    logger.info(f"      Court: {doc.metadata.get('court', 'N/A')}")
                    logger.info(f"      Case: {doc.metadata.get('case_number', 'N/A')}")
                    logger.info(f"      Type: {doc.metadata.get('chunk_type', 'N/A')}")
            except Exception as e:
                logger.warning(f"  ⚠ Search failed: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("✓ VECTOR STORE IS READY!")
        logger.info("=" * 80)
        logger.info(f"\nLocation: data/vectorstore/")
        logger.info(f"Total documents indexed: {total_chunks}")
        logger.info(f"Sources: {', '.join(f['name'] for f in chunk_files)}")
        logger.info("\nYou can now use this vectorstore in your RAG pipeline!")

        # Show usage example
        logger.info("\n" + "-" * 80)
        logger.info("Usage example:")
        logger.info("-" * 80)
        print("""
from src.rag.vectorstore import LegalVectorStore

# Load vector store
vs = LegalVectorStore(persist_directory="data/vectorstore")
vs.load_vectorstore()

# Search
results = vs.search("your query here", k=5, strategy="hybrid")

for doc in results:
    print(doc.metadata['court'], doc.metadata['case_number'])
        """)

    except Exception as e:
        logger.error(f"✗ Vector store build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
