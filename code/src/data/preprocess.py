"""
Legal Document Preprocessing Pipeline
Handles PDF extraction, chunking, metadata extraction for Lebanese legal documents
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

import pymupdf  # PyMuPDF
from langdetect import detect, DetectorFactory
from loguru import logger

# Set seed for consistent language detection
DetectorFactory.seed = 0


@dataclass
class LegalDocument:
    """Represents a legal document with metadata."""

    content: str
    language: str
    document_type: str
    source_file: str
    metadata: Dict[str, any]


@dataclass
class LegalChunk:
    """Represents a chunk of legal text."""

    text: str
    article_number: Optional[str]
    language: str
    document_source: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, any]


class LegalDocumentProcessor:
    """Process legal PDFs into structured chunks."""

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./data_processed/documents")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Legal document patterns
        self.article_patterns = {
            "ar": r"(?:المادة|مادة)\s*(\d+)",  # Article in Arabic
            "fr": r"(?:Article|Art\.?)\s*(\d+)",  # Article in French
            "en": r"(?:Article|Art\.?)\s*(\d+)",  # Article in English
        }

    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            # Clean text for better detection
            clean_text = text.strip()[:1000]  # Use first 1000 chars
            lang = detect(clean_text)

            # Map to our supported languages
            lang_map = {
                "ar": "ar",
                "fr": "fr",
                "en": "en",
            }
            return lang_map.get(lang, "unknown")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"

    def extract_pdf(self, pdf_path: Path) -> LegalDocument:
        """Extract text and metadata from PDF."""
        logger.info(f"Processing: {pdf_path.name}")

        doc = pymupdf.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            full_text.append(text)

        content = unicodedata.normalize("NFKC", "\n\n".join(full_text))

        # Detect language
        language = self.detect_language(content)

        # Determine document type
        document_type = self._classify_document(pdf_path.name, content)

        metadata = {
            "page_count": len(doc),
            "file_size": pdf_path.stat().st_size,
            "extracted_date": str(Path(pdf_path).stat().st_mtime),
        }

        doc.close()

        return LegalDocument(
            content=content,
            language=language,
            document_type=document_type,
            source_file=str(pdf_path),
            metadata=metadata,
        )

    def _classify_document(self, filename: str, content: str) -> str:
        """Classify document type based on filename and content."""
        filename_lower = filename.lower()

        if "obligation" in filename_lower or "contract" in filename_lower:
            return "code_obligations_contracts"
        elif "penal" in filename_lower or "عقوبات" in content[:1000]:
            return "penal_code"
        elif "criminal" in filename_lower or "procedure" in filename_lower:
            return "criminal_procedure"
        elif "cassation" in filename_lower or "تمييز" in content[:1000]:
            return "case_law"
        elif "dalil" in filename_lower:
            return "legal_guide"
        else:
            return "legal_document"

    def extract_articles(self, text: str, language: str) -> List[Tuple[str, str]]:
        """Extract article numbers and their content."""
        if language not in self.article_patterns:
            return []

        # Normalize Unicode so Arabic presentation-form characters (from scanned/legacy
        # PDFs) are converted to standard Arabic codepoints before regex matching.
        text = unicodedata.normalize("NFKC", text)

        pattern = self.article_patterns[language]
        articles = []

        # Find all article markers
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))

        for i, match in enumerate(matches):
            article_num = match.group(1)
            start_pos = match.start()

            # Get text until next article or end
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            article_text = text[start_pos:end_pos].strip()
            articles.append((article_num, article_text))

        return articles

    def chunk_document(
        self,
        document: LegalDocument,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "article",
    ) -> List[LegalChunk]:
        """Chunk document into smaller pieces."""

        if strategy == "article":
            return self._chunk_by_article(document, chunk_size)
        else:
            return self._chunk_fixed_size(document, chunk_size, chunk_overlap)

    def _chunk_by_article(
        self, document: LegalDocument, max_chunk_size: int = 1000
    ) -> List[LegalChunk]:
        """Chunk by legal articles."""
        chunks = []
        articles = self.extract_articles(document.content, document.language)

        if not articles:
            # Fallback to fixed size chunking
            logger.warning(
                f"No articles found in {document.source_file}, using fixed chunking"
            )
            return self._chunk_fixed_size(document, max_chunk_size, 200)

        for idx, (article_num, article_text) in enumerate(articles):
            # Each article is stored as one complete chunk — never split.
            # The embedding model truncates long texts internally, but the full
            # text is preserved in the vector store for the agent to read.
            chunks.append(
                LegalChunk(
                    text=article_text,
                    article_number=article_num,
                    language=document.language,
                    document_source=document.source_file,
                    page_number=0,
                    chunk_index=idx,
                    metadata={
                        "document_type": document.document_type,
                        "article_number": article_num,
                    },
                )
            )

        return chunks

    def _chunk_fixed_size(
        self, document: LegalDocument, chunk_size: int, overlap: int
    ) -> List[LegalChunk]:
        """Chunk by fixed size with overlap."""
        text = document.content
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                LegalChunk(
                    text=chunk_text,
                    article_number=None,
                    language=document.language,
                    document_source=document.source_file,
                    page_number=0,
                    chunk_index=len(chunks),
                    metadata={
                        "document_type": document.document_type,
                        "chunking_strategy": "fixed_size",
                    },
                )
            )

            start = end - overlap

        return chunks

    def _split_long_article(self, text: str, max_size: int) -> List[str]:
        """Split long article into smaller pieces at sentence boundaries."""
        sentences = re.split(r"([.!?。]+)", text)
        chunks = []
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_all_documents(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "article",
    ) -> List[LegalChunk]:
        """Process all PDFs in data directory."""

        all_chunks = []

        # Process each language folder
        for lang_folder in ["AR", "FR", "ENG"]:
            lang_dir = self.data_dir / lang_folder
            if not lang_dir.exists():
                logger.warning(f"Directory not found: {lang_dir}")
                continue

            # Process all PDFs in folder
            for pdf_file in lang_dir.glob("*.pdf"):
                try:
                    # Extract document
                    document = self.extract_pdf(pdf_file)

                    # Chunk document
                    chunks = self.chunk_document(
                        document,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        strategy=chunking_strategy,
                    )

                    all_chunks.extend(chunks)
                    logger.info(
                        f"Processed {pdf_file.name}: {len(chunks)} chunks created"
                    )

                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")

        # Save processed chunks
        self._save_chunks(all_chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _save_chunks(self, chunks: List[LegalChunk]):
        """Save processed chunks to JSON."""
        chunks_data = [
            {
                "text": chunk.text,
                "article_number": chunk.article_number,
                "language": chunk.language,
                "document_source": chunk.document_source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        output_file = self.output_dir / "processed_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """Main preprocessing pipeline."""
    logger.info("Starting legal document preprocessing...")

    processor = LegalDocumentProcessor(data_dir="../data")

    # Process all documents
    chunks = processor.process_all_documents(
        chunk_size=1000, chunk_overlap=200, chunking_strategy="article"
    )

    logger.info("Preprocessing complete!")
    logger.info(f"Total chunks: {len(chunks)}")

    # Print statistics
    languages = {}
    doc_types = {}

    for chunk in chunks:
        languages[chunk.language] = languages.get(chunk.language, 0) + 1
        doc_type = chunk.metadata.get("document_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    logger.info(f"Language distribution: {languages}")
    logger.info(f"Document types: {doc_types}")


if __name__ == "__main__":
    main()
