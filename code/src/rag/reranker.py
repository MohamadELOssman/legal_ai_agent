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

    # Tried in order. Multilingual first (the corpus is Arabic — the English
    # ms-marco model mis-ranks Arabic and is why reranking was disabled); the
    # English model is kept last as an offline fallback if a download fails.
    DEFAULT_MODELS = [
        "BAAI/bge-reranker-base",                # multilingual cross-encoder (Arabic-capable)
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # English fallback (usually already cached)
    ]

    def __init__(self, model_name: str = None):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model to load. If None, tries DEFAULT_MODELS
                in order (multilingual bge first, English ms-marco as fallback).
                - BAAI/bge-reranker-base / BAAI/bge-reranker-v2-m3 (multilingual)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (English, fast)
        """
        self.model = None
        self.model_name = None

        if not RERANKER_AVAILABLE:
            logger.warning("Reranker not available - install sentence-transformers")
            return

        candidates = [model_name] if model_name else list(self.DEFAULT_MODELS)
        for name in candidates:
            try:
                logger.info(f"Loading reranker model: {name}")
                self.model = CrossEncoder(name)
                self.model_name = name
                logger.info(f"✓ Reranker loaded successfully ({name})")
                break
            except Exception as e:
                logger.warning(f"Could not load reranker '{name}': {e}")
        if self.model is None:
            logger.error("No reranker model could be loaded — reranking disabled")

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


def get_reranker(model_name: str = None) -> LegalDocumentReranker:
    """Get global reranker instance (None model_name → multilingual default chain)."""
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
