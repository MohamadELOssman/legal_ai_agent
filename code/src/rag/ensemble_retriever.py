"""
Simple EnsembleRetriever implementation
Combines multiple retrievers with weighted scoring
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines results from multiple retrievers.

    Args:
        retrievers: List of retrievers to ensemble
        weights: List of weights for each retriever (must sum to 1.0)
    """

    retrievers: List[BaseRetriever]
    weights: List[float]

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float]):
        """Initialize ensemble retriever."""
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")

        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        super().__init__(retrievers=retrievers, weights=weights)

    def invoke(self, query: str) -> List[Document]:
        """
        Invoke the retriever (new API).

        Args:
            query: The query string

        Returns:
            Combined and deduplicated list of documents
        """
        return self._get_relevant_documents(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents from all retrievers and combine them (old API).

        Args:
            query: The query string

        Returns:
            Combined and deduplicated list of documents
        """
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Internal method to get relevant documents.

        Args:
            query: The query string

        Returns:
            Combined and deduplicated list of documents
        """
        # Get documents from each retriever
        all_docs = []
        doc_scores = {}  # Track scores for deduplication

        for retriever, weight in zip(self.retrievers, self.weights):
            # Try new API first, fall back to old
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)

            # Assign scores based on position and weight
            for i, doc in enumerate(docs):
                # Create unique key for document
                doc_key = doc.page_content[:100]  # Use first 100 chars as key

                # Score based on position (higher position = lower score)
                position_score = 1.0 / (i + 1)
                weighted_score = position_score * weight

                if doc_key in doc_scores:
                    # Document already seen, add to score
                    doc_scores[doc_key]["score"] += weighted_score
                else:
                    # New document
                    doc_scores[doc_key] = {
                        "doc": doc,
                        "score": weighted_score
                    }

        # Sort by score and return documents
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_docs]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (not implemented, falls back to sync)."""
        return self._get_relevant_documents(query)
