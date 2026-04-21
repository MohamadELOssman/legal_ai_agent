"""
Citation Validation System
Prevents hallucinated article numbers
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Optional
from loguru import logger


class CitationValidator:
    """Validates legal citations against known articles."""

    def __init__(self, articles_db_file: str = "./data_processed/articles_index.json"):
        self.articles_db_file = Path(articles_db_file)
        self.known_articles = self._load_articles_db()

        logger.info(f"Citation validator loaded with {len(self.known_articles)} known articles")

    def _load_articles_db(self) -> Dict[str, Set[str]]:
        """Load database of known articles."""

        if not self.articles_db_file.exists():
            logger.warning("Articles database not found. Creating empty database.")
            return self._create_empty_db()

        try:
            with open(self.articles_db_file, "r") as f:
                data = json.load(f)
                # Convert lists to sets
                return {doc: set(articles) for doc, articles in data.items()}
        except Exception as e:
            logger.error(f"Failed to load articles DB: {e}")
            return self._create_empty_db()

    def _create_empty_db(self) -> Dict[str, Set[str]]:
        """Create empty articles database."""
        return {
            "code_obligations_contracts": set(),
            "penal_code": set(),
            "criminal_procedure": set(),
        }

    def build_articles_index(self, chunks_file: str = "./data_processed/documents/processed_chunks.json"):
        """Build index of all article numbers from processed chunks."""

        logger.info("Building articles index...")

        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        articles_by_doc = {}

        for chunk in chunks:
            article_num = chunk.get("article_number")
            doc_type = chunk.get("metadata", {}).get("document_type", "unknown")

            if article_num and article_num != "None":
                # Clean article number
                article_clean = self._clean_article_number(str(article_num))

                if doc_type not in articles_by_doc:
                    articles_by_doc[doc_type] = set()

                articles_by_doc[doc_type].add(article_clean)

        # Save to file
        self.articles_db_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON
        articles_serializable = {doc: list(articles) for doc, articles in articles_by_doc.items()}

        with open(self.articles_db_file, "w") as f:
            json.dump(articles_serializable, f, indent=2)

        logger.info(f"Articles index saved to {self.articles_db_file}")

        # Reload
        self.known_articles = self._load_articles_db()

        # Print stats
        for doc_type, articles in self.known_articles.items():
            logger.info(f"  {doc_type}: {len(articles)} articles")

    def _clean_article_number(self, article: str) -> str:
        """Extract clean article number."""
        # Extract just the number(s)
        match = re.search(r"\d+", article)
        if match:
            return match.group(0)
        return article

    def validate_citation(self, article_number: str, document_type: str) -> bool:
        """
        Validate if an article number exists.

        Args:
            article_number: Article number to validate
            document_type: Type of legal document

        Returns:
            True if valid, False if hallucinated
        """

        if not self.known_articles:
            # No database - can't validate
            logger.warning("No articles database - cannot validate citations")
            return True  # Assume valid

        # Clean article number
        article_clean = self._clean_article_number(str(article_number))

        # Check if exists
        if document_type in self.known_articles:
            is_valid = article_clean in self.known_articles[document_type]

            if not is_valid:
                logger.warning(
                    f"⚠️ Potential hallucination: Article {article_number} "
                    f"not found in {document_type}"
                )

            return is_valid

        else:
            logger.warning(f"Unknown document type: {document_type}")
            return True  # Can't validate unknown types

    def validate_citations_list(
        self, citations: List[Dict]
    ) -> Dict[str, any]:
        """
        Validate a list of citations.

        Args:
            citations: List of citation dicts with 'article_number' and 'document_type'

        Returns:
            Validation report
        """

        total = len(citations)
        valid = 0
        invalid = 0
        hallucinations = []

        for citation in citations:
            article = citation.get("article_number", "")
            doc_type = citation.get("metadata", {}).get("document_type", "")

            if not doc_type:
                # Try to infer from source
                source = citation.get("source_document", "")
                doc_type = self._infer_doc_type(source)

            is_valid = self.validate_citation(article, doc_type)

            if is_valid:
                valid += 1
            else:
                invalid += 1
                hallucinations.append({
                    "article": article,
                    "document": doc_type,
                    "citation_text": citation.get("citation_text", "")
                })

        report = {
            "total_citations": total,
            "valid_citations": valid,
            "invalid_citations": invalid,
            "hallucination_rate": invalid / total if total > 0 else 0.0,
            "hallucinations": hallucinations,
        }

        return report

    def _infer_doc_type(self, source: str) -> str:
        """Infer document type from source string."""
        source_lower = source.lower()

        if "obligation" in source_lower or "contract" in source_lower:
            return "code_obligations_contracts"
        elif "penal" in source_lower or "عقوبات" in source:
            return "penal_code"
        elif "criminal" in source_lower or "procedure" in source_lower:
            return "criminal_procedure"
        else:
            return "unknown"

    def get_stats(self) -> Dict[str, int]:
        """Get statistics on known articles."""
        return {
            doc_type: len(articles)
            for doc_type, articles in self.known_articles.items()
        }


def main():
    """Build and test citation validator."""

    validator = CitationValidator()

    # Build index from chunks
    validator.build_articles_index()

    # Test validation
    test_citations = [
        {"article_number": "121", "document_type": "code_obligations_contracts"},
        {"article_number": "122", "document_type": "code_obligations_contracts"},
        {"article_number": "9999", "document_type": "code_obligations_contracts"},  # Likely hallucination
    ]

    report = validator.validate_citations_list(test_citations)

    print("\n" + "=" * 60)
    print("CITATION VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Citations: {report['total_citations']}")
    print(f"Valid: {report['valid_citations']}")
    print(f"Invalid (Hallucinations): {report['invalid_citations']}")
    print(f"Hallucination Rate: {report['hallucination_rate']:.1%}")

    if report['hallucinations']:
        print("\nHallucinated Citations:")
        for h in report['hallucinations']:
            print(f"  - Article {h['article']} in {h['document']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
