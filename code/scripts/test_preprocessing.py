"""
Quick test script to verify document preprocessing setup
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.agents import DocumentPreprocessingAgent


def test_preprocessing():
    """Test the document preprocessing agent with sample document."""

    logger.info("=" * 60)
    logger.info("TESTING DOCUMENT PREPROCESSING AGENT")
    logger.info("=" * 60)

    # Initialize agent
    logger.info("\n1. Initializing agent...")
    try:
        agent = DocumentPreprocessingAgent(
            model="claude-sonnet-4.5", temperature=0.1
        )
        logger.info("✓ Agent initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize agent: {e}")
        return False

    # Check if sample document exists
    sample_doc = Path("data/raw_documents/sample_document.txt")
    if not sample_doc.exists():
        logger.error(f"✗ Sample document not found at {sample_doc}")
        logger.info("Please create a sample document first")
        return False

    logger.info(f"✓ Sample document found: {sample_doc}")

    # Process sample document
    logger.info("\n2. Processing sample document...")
    try:
        output = agent.process_file(sample_doc)

        if not output.success:
            logger.error(f"✗ Processing failed: {output.error}")
            return False

        logger.info("✓ Document processed successfully")

        # Validate outputs
        result = output.result
        logger.info("\n3. Validating outputs...")

        # Check Output 1
        if "output_1_full_document" in result:
            output1 = result["output_1_full_document"]
            logger.info("✓ Output 1 (Full Document): Present")
            logger.info(f"  - Sections found: {output1.get('processing_metadata', {}).get('sections_found', [])}")
            logger.info(f"  - Quality flags: {len(output1.get('quality_flags', []))}")
        else:
            logger.warning("✗ Output 1 missing")

        # Check Output 2
        if "output_2_legal_summary" in result:
            output2 = result["output_2_legal_summary"]
            logger.info("✓ Output 2 (Legal Summary): Present")
            logger.info(f"  - Legal issues: {len(output2.get('legal_issues', []))}")
            logger.info(f"  - Rules applied: {len(output2.get('rules_applied', []))}")
        else:
            logger.warning("✗ Output 2 missing")

        # Check Output 3
        if "output_3_ratio_only" in result:
            output3 = result["output_3_ratio_only"]
            logger.info("✓ Output 3 (Ratio Decidendi): Present")
            logger.info(f"  - Length: {len(output3)} characters")
        else:
            logger.warning("✗ Output 3 missing")

        logger.info("\n" + "=" * 60)
        logger.info("TEST PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nYou can now run the full batch processor:")
        logger.info("  python scripts/process_legal_documents.py")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"✗ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    logger.info("\nDocument Preprocessing Agent - Quick Test\n")

    success = test_preprocessing()

    if not success:
        logger.error("\nTest failed. Please check the error messages above.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
