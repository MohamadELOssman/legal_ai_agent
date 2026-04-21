"""
Batch Processing Script for Legal Documents
Processes all raw documents through the Document Preprocessing Agent
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from src.agents.document_preprocessing_agent import DocumentPreprocessingAgent


class DocumentBatchProcessor:
    """Batch process legal documents for vector store ingestion."""

    def __init__(
        self,
        raw_documents_dir: str = "data/raw_documents",
        output_dir: str = "data/processed_documents",
        model: str = "claude-sonnet-4-20250514",
        skip_existing: bool = True,
    ):
        self.raw_documents_dir = Path(raw_documents_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.skip_existing = skip_existing

        # Create output directories
        self.full_docs_dir = self.output_dir / "full_documents"
        self.summaries_dir = self.output_dir / "summaries"
        self.ratios_dir = self.output_dir / "ratios"
        self.vector_store_dir = self.output_dir / "vector_store_ready"

        for directory in [
            self.full_docs_dir,
            self.summaries_dir,
            self.ratios_dir,
            self.vector_store_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Determine if we should use vision (only for use_cases)
        # For legal codes (AR, FR, ENG), use text extraction instead
        use_vision = "use_case" in str(raw_documents_dir).lower()

        # Initialize agent
        self.agent = DocumentPreprocessingAgent(model=model, use_vision=use_vision)

        logger.info(f"Initialized batch processor with model: {model}")
        logger.info(f"Vision processing: {'enabled' if use_vision else 'disabled (text extraction)'}")
        logger.info(f"Input directory: {self.raw_documents_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Skip existing: {skip_existing}")

    def get_document_files(self) -> List[Path]:
        """Get all document files from raw_documents directory."""
        # Support both scanned documents (images/PDFs) and text files
        supported_extensions = [
            ".pdf",   # Scanned PDFs
            ".png",   # Scanned images
            ".jpg",   # Scanned images
            ".jpeg",  # Scanned images
            ".tiff",  # Scanned images
            ".tif",   # Scanned images
            ".txt",   # Plain text (pre-OCR'd)
            ".md",    # Markdown text
        ]
        files = []

        for ext in supported_extensions:
            files.extend(self.raw_documents_dir.glob(f"*{ext}"))

        logger.info(f"Found {len(files)} documents to process")

        # Log breakdown by type
        scanned = sum(1 for f in files if f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"})
        text = len(files) - scanned
        logger.info(f"  - Scanned documents (images/PDFs): {scanned}")
        logger.info(f"  - Text documents: {text}")

        return sorted(files)

    def is_already_processed(self, file_path: Path) -> bool:
        """
        Check if a document has already been processed.

        Args:
            file_path: Path to document file

        Returns:
            True if output files exist, False otherwise
        """
        doc_id = file_path.stem

        # Check if all three output files exist
        full_doc_path = self.full_docs_dir / f"{doc_id}_full.json"
        summary_path = self.summaries_dir / f"{doc_id}_summary.json"
        ratio_path = self.ratios_dir / f"{doc_id}_ratio.txt"
        chunks_path = self.vector_store_dir / f"{doc_id}_chunks.jsonl"

        all_exist = all([
            full_doc_path.exists(),
            summary_path.exists(),
            ratio_path.exists(),
            chunks_path.exists(),
        ])

        return all_exist

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document.

        Args:
            file_path: Path to document file

        Returns:
            Processing result with all outputs
        """
        doc_id = file_path.stem

        # Check if already processed
        if self.skip_existing and self.is_already_processed(file_path):
            logger.info(f"⏭️  Skipping {file_path.name} (already processed)")
            return {
                "file": str(file_path),
                "success": True,
                "skipped": True,
                "document_id": doc_id,
                "chunks_created": 0,
                "message": "Already processed",
                "outputs": {
                    "full_document": str(self.full_docs_dir / f"{doc_id}_full.json"),
                    "summary": str(self.summaries_dir / f"{doc_id}_summary.json"),
                    "ratio": str(self.ratios_dir / f"{doc_id}_ratio.txt"),
                },
            }

        logger.info(f"Processing: {file_path.name}")

        # Process through agent
        output = self.agent.process_file(file_path)

        if not output.success:
            logger.error(f"Failed to process {file_path.name}: {output.error}")
            return {
                "file": str(file_path),
                "success": False,
                "error": output.error,
            }

        # Extract outputs
        result = output.result
        doc_id = file_path.stem

        # Save Output 1: Full processed document
        full_doc_path = self.full_docs_dir / f"{doc_id}_full.json"
        with open(full_doc_path, "w", encoding="utf-8") as f:
            json.dump(result["output_1_full_document"], f, ensure_ascii=False, indent=2)

        # Save Output 2: Legal summary
        summary_path = self.summaries_dir / f"{doc_id}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result["output_2_legal_summary"], f, ensure_ascii=False, indent=2)

        # Save Output 3: Ratio decidendi only
        ratio_path = self.ratios_dir / f"{doc_id}_ratio.txt"
        with open(ratio_path, "w", encoding="utf-8") as f:
            f.write(result["output_3_ratio_only"])

        # Create vector store ready format (JSONL - one chunk per line)
        vector_chunks = self._create_vector_chunks(doc_id, result)

        logger.info(
            f"✓ Processed {file_path.name} - "
            f"{len(vector_chunks)} chunks created for vector store"
        )

        return {
            "file": str(file_path),
            "success": True,
            "document_id": doc_id,
            "chunks_created": len(vector_chunks),
            "outputs": {
                "full_document": str(full_doc_path),
                "summary": str(summary_path),
                "ratio": str(ratio_path),
            },
        }

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata by removing empty lists (Chroma doesn't accept them).

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Cleaned metadata with empty lists removed
        """
        cleaned = {}
        for key, value in metadata.items():
            # Skip empty lists
            if isinstance(value, list) and len(value) == 0:
                continue
            # Keep everything else
            cleaned[key] = value
        return cleaned

    def _create_vector_chunks(
        self, doc_id: str, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create vector store ready chunks from processed document.

        Creates multiple chunks:
        1. Full document chunk (for exact match retrieval)
        2. Summary chunk (for semantic search)
        3. Ratio decidendi chunk (for legal principle search)
        4. Individual section chunks (for granular retrieval)

        Returns:
            List of chunks ready for vector store ingestion
        """
        chunks = []
        timestamp = datetime.now().isoformat()

        output1 = result["output_1_full_document"]
        output2 = result["output_2_legal_summary"]
        output3 = result["output_3_ratio_only"]

        # Common metadata - clean to remove empty lists
        doc_metadata = self._clean_metadata(output1.get("document_metadata", {}))
        base_metadata = {
            "document_id": doc_id,
            "source": "lebanese_legal_corpus",
            "processed_at": timestamp,
            **doc_metadata,
        }

        # Chunk 1: Full cleaned document
        chunk1_metadata = {
            **base_metadata,
            "chunk_purpose": "exact_match_retrieval",
        }
        quality_flags = output1.get("quality_flags", [])
        if quality_flags:  # Only add if not empty
            chunk1_metadata["quality_flags"] = quality_flags

        chunks.append(
            {
                "chunk_id": f"{doc_id}_full",
                "chunk_type": "full_document",
                "text": output1["cleaned_body"],
                "metadata": chunk1_metadata,
            }
        )

        # Chunk 2: RAG-optimized summary
        summary_text = f"""
{output2['case_snapshot']}

الوقائع:
{output2['facts_summary']}

المسائل القانونية:
{chr(10).join(f"{i+1}. {issue}" for i, issue in enumerate(output2['legal_issues']))}

الأساس القانوني:
{chr(10).join(f"- {rule['article']} من {rule['law']}: {rule['how_applied']}" for rule in output2['rules_applied'])}

الحكم القانوني (Ratio Decidendi):
{output2['ratio_decidendi']}
""".strip()

        chunk2_metadata = {
            **base_metadata,
            "chunk_purpose": "semantic_search",
        }
        search_tags = output2.get("search_tags", {})
        if search_tags:  # Only add if not empty
            chunk2_metadata["search_tags"] = search_tags

        chunks.append(
            {
                "chunk_id": f"{doc_id}_summary",
                "chunk_type": "legal_summary",
                "text": summary_text,
                "metadata": chunk2_metadata,
            }
        )

        # Chunk 3: Ratio decidendi only (for legal principle search)
        chunks.append(
            {
                "chunk_id": f"{doc_id}_ratio",
                "chunk_type": "ratio_decidendi",
                "text": output3,
                "metadata": {
                    **base_metadata,
                    "chunk_purpose": "legal_principle_search",
                    "outcome": base_metadata.get("outcome", ""),
                },
            }
        )

        # Chunk 4: Facts section (if substantial)
        if len(output2["facts_summary"]) > 100:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_facts",
                    "chunk_type": "facts",
                    "text": output2["facts_summary"],
                    "metadata": {
                        **base_metadata,
                        "chunk_purpose": "factual_similarity_search",
                    },
                }
            )

        return chunks

    def save_vector_chunks(
        self, doc_id: str, chunks: List[Dict[str, Any]]
    ) -> Path:
        """
        Save chunks to JSONL format for vector store ingestion.

        Args:
            doc_id: Document identifier
            chunks: List of chunks to save

        Returns:
            Path to saved JSONL file
        """
        output_path = self.vector_store_dir / f"{doc_id}_chunks.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        return output_path

    def process_all(self) -> Dict[str, Any]:
        """
        Process all documents in the raw_documents directory.

        Returns:
            Summary of processing results
        """
        files = self.get_document_files()

        if not files:
            logger.warning("No documents found to process!")
            return {
                "total_files": 0,
                "processed": 0,
                "failed": 0,
                "results": [],
            }

        results = []
        all_chunks = []

        # Process each file with progress bar
        for file_path in tqdm(files, desc="Processing documents"):
            result = self.process_document(file_path)
            results.append(result)

            if result["success"]:
                # Get chunks from the processed document
                doc_id = result["document_id"]

                # Re-read the outputs to create chunks
                with open(result["outputs"]["full_document"], "r", encoding="utf-8") as f:
                    output1 = json.load(f)
                with open(result["outputs"]["summary"], "r", encoding="utf-8") as f:
                    output2 = json.load(f)
                with open(result["outputs"]["ratio"], "r", encoding="utf-8") as f:
                    output3 = f.read()

                combined_result = {
                    "output_1_full_document": output1,
                    "output_2_legal_summary": output2,
                    "output_3_ratio_only": output3,
                }

                chunks = self._create_vector_chunks(doc_id, combined_result)
                self.save_vector_chunks(doc_id, chunks)
                all_chunks.extend(chunks)

        # Create master JSONL file with all chunks
        master_file = self.vector_store_dir / "all_documents.jsonl"
        with open(master_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # Create summary report
        successful = sum(1 for r in results if r["success"] and not r.get("skipped", False))
        skipped = sum(1 for r in results if r.get("skipped", False))
        failed = sum(1 for r in results if not r["success"])

        summary = {
            "processing_date": datetime.now().isoformat(),
            "total_files": len(files),
            "processed": successful,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": len(all_chunks),
            "master_file": str(master_file),
            "results": results,
        }

        # Save summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"Total files: {len(files)}")
        logger.info(f"Processed successfully: {successful}")
        logger.info(f"Skipped (already processed): {skipped}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info(f"Master file: {master_file}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("=" * 60)

        return summary


def main():
    """Main entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process Lebanese legal documents"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw_documents",
        help="Directory containing raw documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_documents",
        help="Directory for processed outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for processing (e.g., claude-sonnet-4-20250514, claude-opus-4-20250514)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed documents",
    )

    args = parser.parse_args()

    # Initialize processor
    processor = DocumentBatchProcessor(
        raw_documents_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        skip_existing=not args.force,  # Skip existing unless --force is specified
    )

    # Process all documents
    summary = processor.process_all()

    # Print results
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {summary['total_files']}")
    print(f"Processed: {summary['processed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Master file: {summary['master_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
