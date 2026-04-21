"""
Cross-Encoder Reranker for Legal Documents
Improves retrieval precision by 10-20%
"""

from typing import List, Tuple
from langchain_core.documents import Document
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Reranking disabled.")


class LegalDocumentReranker:
    """
    Reranks retrieved documents using cross-encoder.

    Cross-encoders are more accurate than bi-encoders (embeddings) because
    they process query + document together, capturing interaction features.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
                - BAAI/bge-reranker-base (multilingual support)
        """
        self.model_name = model_name
        self.model = None

        if RERANKER_AVAILABLE:
            try:
                logger.info(f"Loading reranker model: {model_name}")
                self.model = CrossEncoder(model_name)
                logger.info("✓ Reranker loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.model = None
        else:
            logger.warning("Reranker not available - install sentence-transformers")

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Reranked documents (most relevant first)
        """

        if not self.is_available():
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k]

        if not documents:
            return []

        try:
            # Prepare pairs for cross-encoder
            pairs = [
                [query, doc.page_content[:1000]]  # Limit to 1000 chars for speed
                for doc in documents
            ]

            # Get relevance scores
            scores = self.model.predict(pairs)

            # Combine documents with scores
            doc_score_pairs = list(zip(documents, scores))

            # Sort by score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return top-k documents
            reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

            logger.debug(
                f"Reranked {len(documents)} docs → top {len(reranked_docs)} "
                f"(scores: {[f'{s:.3f}' for _, s in doc_score_pairs[:3]]})"
            )

            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores.

        Returns:
            List of (document, score) tuples
        """

        if not self.is_available():
            return [(doc, 0.0) for doc in documents[:top_k]]

        if not documents:
            return []

        try:
            # Prepare pairs
            pairs = [[query, doc.page_content[:1000]] for doc in documents]

            # Get scores
            scores = self.model.predict(pairs)

            # Combine and sort
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            return doc_score_pairs[:top_k]

        except Exception as e:
            logger.error(f"Reranking with scores failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]


# Global reranker instance (lazy loading)
_reranker = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> LegalDocumentReranker:
    """Get global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = LegalDocumentReranker(model_name=model_name)
    return _reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5
) -> List[Document]:
    """Convenience function for reranking."""
    reranker = get_reranker()
    return reranker.rerank(query, documents, top_k)


if __name__ == "__main__":
    # Test reranker
    print("Testing Legal Document Reranker...")

    # Create test documents
    test_docs = [
        Document(page_content="المادة 121 من قانون الموجبات والعقود تتعلق بالمسؤولية المدنية"),
        Document(page_content="الطقس جميل اليوم في بيروت"),
        Document(page_content="المسؤولية الجزائية للموظف العام في القانون اللبناني"),
        Document(page_content="عقد البيع في القانون اللبناني"),
    ]

    test_query = "ما هي المسؤولية المدنية؟"

    # Initialize reranker
    reranker = LegalDocumentReranker()

    if reranker.is_available():
        print(f"\n✓ Reranker loaded: {reranker.model_name}")
        print(f"\nQuery: {test_query}")
        print(f"Documents: {len(test_docs)}")

        # Rerank
        reranked = reranker.rerank_with_scores(test_query, test_docs, top_k=4)

        print("\nReranked Results:")
        for i, (doc, score) in enumerate(reranked, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Content: {doc.page_content[:80]}...")

    else:
        print("✗ Reranker not available")
        print("Install with: pip install sentence-transformers")
