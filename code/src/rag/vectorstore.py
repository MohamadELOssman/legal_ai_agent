"""
Vector Store for Legal Documents
Implements hybrid search (semantic + BM25) for Lebanese legal corpus
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from loguru import logger

from src.config import get_config
from src.rag.reranker import LegalDocumentReranker
from src.rag.embedding_factory import get_embeddings
from src.rag.ensemble_retriever import EnsembleRetriever


class LegalVectorStore:
    """Vector store with hybrid search for legal documents."""

    def __init__(
        self,
        persist_directory: str = "./data_processed/vectorstore",
        embedding_provider: str = "huggingface",  # "huggingface", "google", "openai"
        embedding_model: str = None,
        use_reranking: bool = True,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_reranking = use_reranking
        self.embedding_provider = embedding_provider

        config = get_config()

        # Initialize embeddings using factory
        logger.info(f"Loading embeddings from provider: {embedding_provider}")
        self.embeddings = get_embeddings(
            provider=embedding_provider,
            model_name=embedding_model
        )

        # Initialize Chroma vector store
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        # Initialize reranker (lazy loading)
        self.reranker = None
        if use_reranking:
            try:
                self.reranker = LegalDocumentReranker()
                if self.reranker.is_available():
                    logger.info("✓ Reranking enabled")
                else:
                    logger.warning("Reranking requested but not available")
                    self.use_reranking = False
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}")
                self.use_reranking = False

        logger.info(f"✓ Initialized LegalVectorStore with {embedding_provider} embeddings")

    def load_documents_from_chunks(
        self, chunks_file: str = "./data_processed/documents/processed_chunks.json"
    ) -> List[Document]:
        """
        Load processed chunks as LangChain Documents.

        Supports both JSON array format and JSONL format.
        """

        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}. Run preprocessing first."
            )

        # Determine format by file extension
        is_jsonl = chunks_path.suffix.lower() == ".jsonl"

        chunks_data = []

        if is_jsonl:
            # Load JSONL format (one JSON object per line)
            logger.info(f"Loading JSONL format from: {chunks_file}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks_data.append(json.loads(line))
        else:
            # Load JSON array format (legacy)
            logger.info(f"Loading JSON format from: {chunks_file}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

        documents = []
        for chunk in chunks_data:
            # Support both old and new format
            # Old format: article_number, language, document_source
            # New format: chunk_id, chunk_type, text, metadata

            if "chunk_id" in chunk:
                # New format from document preprocessing agent
                metadata = chunk.get("metadata", {})
                metadata["chunk_id"] = chunk["chunk_id"]
                metadata["chunk_type"] = chunk["chunk_type"]
                text = chunk["text"]
            else:
                # Old format (legacy)
                metadata = {
                    "article_number": chunk.get("article_number"),
                    "language": chunk.get("language"),
                    "document_source": chunk.get("document_source"),
                    "chunk_index": chunk.get("chunk_index"),
                    **chunk.get("metadata", {}),
                }
                text = chunk["text"]

            # Clean metadata: Chroma only accepts str, int, float, bool, or lists of these primitives
            cleaned_metadata = {}
            for key, value in metadata.items():
                # Skip empty lists
                if isinstance(value, list) and len(value) == 0:
                    continue

                # Handle dictionaries - convert to JSON string
                if isinstance(value, dict):
                    # Check if dict contains only empty lists
                    if all(isinstance(v, list) and len(v) == 0 for v in value.values()):
                        continue  # Skip dicts with only empty lists
                    # Convert non-empty dicts to JSON string
                    cleaned_metadata[key] = json.dumps(value, ensure_ascii=False)
                    continue

                # Handle lists - check if they contain only primitives
                if isinstance(value, list):
                    # Check if list contains objects/dicts
                    if any(isinstance(item, (dict, list)) for item in value):
                        # Convert list of objects to JSON string
                        cleaned_metadata[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        # List of primitives is OK
                        cleaned_metadata[key] = value
                    continue

                # Primitives are OK
                cleaned_metadata[key] = value

            doc = Document(page_content=text, metadata=cleaned_metadata)
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from chunks")
        return documents

    def build_vectorstore(self, documents: List[Document]) -> None:
        """Build Chroma vector store from documents."""

        logger.info(f"Building vector store with {len(documents)} documents...")

        # Create Chroma vectorstore — cosine distance so similarity = 1 - score is correct
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name="lebanese_legal_docs",
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info("Vector store built successfully")

    def build_bm25_retriever(self, documents: List[Document]) -> None:
        """Build BM25 keyword retriever."""

        logger.info("Building BM25 retriever...")

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5  # Return top 5 documents

        logger.info("BM25 retriever built successfully")

    def build_hybrid_retriever(self, documents: List[Document], weights: Tuple[float, float] = (0.5, 0.5)) -> None:
        """Build hybrid retriever combining semantic + BM25."""

        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_vectorstore() first.")

        logger.info("Building hybrid retriever...")

        # Semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10}  # Retrieve more for reranking
        )

        # BM25 retriever
        if self.bm25_retriever is None:
            self.build_bm25_retriever(documents)

        # Ensemble retriever (hybrid)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, self.bm25_retriever],
            weights=weights,  # Equal weight to semantic and keyword
        )

        logger.info("Hybrid retriever built successfully")

    def _rebuild_hybrid_retriever_from_chroma(self) -> None:
        """Rebuild BM25 + hybrid retriever using documents already in Chroma."""
    
        try:
            logger.info("Rebuilding hybrid retriever from Chroma documents...")
        
            # Fetch all documents directly from the Chroma collection
            collection = self.vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])
        
            if not result["documents"]:
                logger.warning("No documents found in Chroma collection. Hybrid retriever not built.")
                return
        
            # Reconstruct LangChain Document objects
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(result["documents"], result["metadatas"])
            ]
        
            logger.info(f"Retrieved {len(documents)} documents from Chroma for BM25 rebuild")
        
            # Rebuild BM25 + hybrid retriever
            self.build_hybrid_retriever(documents, weights=(0.5, 0.5))
            logger.info("✓ Hybrid retriever rebuilt successfully")
        
        except Exception as e:
            logger.warning(f"Could not rebuild hybrid retriever: {e}. Falling back to semantic-only search.")

    def load_vectorstore(self) -> None:
        """Load existing vector store from disk."""

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. Build it first."
            )

        logger.info("Loading vector store from disk...")

        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name="lebanese_legal_docs",
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info("Vector store loaded successfully")
        self._rebuild_hybrid_retriever_from_chroma()

    def semantic_search(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None, score_threshold: float = 0.6
    ) -> List[Document]:
        """Perform semantic similarity search with score threshold.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            score_threshold: Minimum similarity score (0-1). Default 0.6.
                           Only documents with similarity >= threshold are returned.
        """

        if self.vectorstore is None:
            self.load_vectorstore()

        # Get results with scores
        search_kwargs = {"k": k * 2}  # Retrieve more to filter by threshold
        if filter_dict:
            # Chroma's similarity_search_with_score requires the $eq operator form
            search_kwargs["filter"] = {
                k: {"$eq": v} for k, v in filter_dict.items()
            }

        results_with_scores = self.vectorstore.similarity_search_with_score(query, **search_kwargs)

        # Filter by threshold and convert to documents
        filtered_results = []
        for doc, score in results_with_scores:
            # Note: Chroma returns distance, not similarity. Lower is better.
            # For cosine similarity: similarity = 1 - distance
            similarity = 1 - score
            if similarity >= score_threshold:
                filtered_results.append(doc)
                logger.debug(f"Document score: {similarity:.3f} (above threshold {score_threshold})")
            else:
                logger.debug(f"Document score: {similarity:.3f} (below threshold {score_threshold}, filtered out)")

        # Limit to k results
        results = filtered_results[:k]

        logger.info(f"Semantic search returned {len(results)} results (filtered from {len(results_with_scores)} by threshold {score_threshold})")
        return results

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform hybrid search (semantic + BM25)."""

        if self.hybrid_retriever is None:
            raise ValueError(
                "Hybrid retriever not built. Call build_hybrid_retriever() first."
            )

        # Try the new API first, fall back to old API
        try:
            results = self.hybrid_retriever.invoke(query)[:k]
        except AttributeError:
            results = self.hybrid_retriever.get_relevant_documents(query)[:k]

        logger.info(f"Hybrid search returned {len(results)} results")
        return results

    def search(
        self,
        query: str,
        k: int = 5,
        strategy: str = "hybrid",
        filter_dict: Optional[Dict] = None,
        use_reranking: bool = None,
        score_threshold: float = 0.6,
    ) -> List[Document]:
        """
        Search documents using specified strategy.

        Args:
            query: Search query
            k: Number of results to return
            strategy: 'semantic', 'bm25', or 'hybrid'
            filter_dict: Metadata filters (only for semantic search)
            use_reranking: Override default reranking setting
            score_threshold: Minimum similarity score (0-1). Default 0.6.

        Returns:
            List of most relevant documents
        """

        # Determine if we should use reranking
        should_rerank = (
            use_reranking if use_reranking is not None else self.use_reranking
        )

        # Retrieve more documents if reranking (to have candidates)
        retrieve_k = k * 2 if should_rerank else k

        # Initial retrieval
        if strategy == "semantic":
            results = self.semantic_search(query, k=retrieve_k, filter_dict=filter_dict, score_threshold=score_threshold)
        elif strategy == "hybrid":
            if self.hybrid_retriever is not None:
                # BM25 doesn't support Chroma metadata filters; apply filter in Python after retrieval
                results = self.hybrid_search(query, k=retrieve_k * 3 if filter_dict else retrieve_k)
                if filter_dict:
                    results = [
                        d for d in results
                        if all(d.metadata.get(k) == v for k, v in filter_dict.items())
                    ]
            else:
                logger.warning("Hybrid retriever not available, falling back to semantic search.")
                results = self.semantic_search(query, k=retrieve_k, filter_dict=filter_dict, score_threshold=score_threshold)
        elif strategy == "bm25":
            if self.bm25_retriever is None:
                raise ValueError("BM25 retriever not built")
            self.bm25_retriever.k = retrieve_k
            results = self.bm25_retriever.get_relevant_documents(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply reranking if enabled
        if should_rerank and self.reranker and self.reranker.is_available():
            logger.debug(f"Reranking {len(results)} documents → top {k}")
            results = self.reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]

        return results

    def multi_search(
        self,
        queries: List[str],
        k: int = 5,
        strategy: str = "hybrid",
        filter_dict: Optional[Dict] = None,
        use_reranking: bool = None,
        score_threshold: float = 0.0,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Run several query variants and fuse the results with Reciprocal Rank Fusion.

        RRF score(doc) = Σ_variants 1 / (rrf_k + rank). This boosts documents that
        rank well across multiple phrasings/translations of the query — used for
        cross-lingual and multi-query retrieval to lift recall.
        """
        queries = [q for q in queries if q and q.strip()]
        if len(queries) <= 1:
            return self.search(queries[0] if queries else "", k=k, strategy=strategy,
                               filter_dict=filter_dict, use_reranking=use_reranking,
                               score_threshold=score_threshold)

        from collections import defaultdict
        scores: Dict[str, float] = defaultdict(float)
        docmap: Dict[str, Document] = {}
        # Retrieve a deeper pool per variant so fusion has material to work with.
        per_variant_k = max(k, 8)
        for q in queries:
            results = self.search(query=q, k=per_variant_k, strategy=strategy,
                                  filter_dict=filter_dict, use_reranking=use_reranking,
                                  score_threshold=score_threshold)
            for rank, doc in enumerate(results, 1):
                key = "|".join(str(doc.metadata.get(f, "")) for f in
                               ("document_id", "article_number", "document_language")) \
                      or doc.page_content[:80]
                scores[key] += 1.0 / (rrf_k + rank)
                docmap[key] = doc

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docmap[key] for key in ranked[:k]]


def build_vectorstore_pipeline(
    chunks_files: List[str] = None,
    persist_directory: str = "./data/vectorstore"
):
    """
    Complete pipeline to build vector store from multiple sources.

    Args:
        chunks_files: List of paths to chunks files (JSONL or JSON). Auto-detects if None.
        persist_directory: Where to save the vectorstore

    Returns:
        LegalVectorStore instance ready for use
    """

    logger.info("Starting vector store build pipeline...")

    # Auto-detect chunks files if not specified
    if chunks_files is None:
        chunks_files = []

        # Check multiple possible locations
        possible_sources = [
            Path("data/processed_use_cases/vector_store_ready/all_documents.jsonl"),
            Path("data/processed_documents/vector_store_ready/all_documents.jsonl"),
            Path("data/processed_AR/vector_store_ready/all_documents.jsonl"),
            Path("data/processed_FR/vector_store_ready/all_documents.jsonl"),
            Path("data/processed_ENG/vector_store_ready/all_documents.jsonl"),
            Path("./data_processed/documents/processed_chunks.json"),  # Legacy
        ]

        for source in possible_sources:
            if source.exists():
                chunks_files.append(str(source))
                logger.info(f"✓ Found chunks: {source}")

        if not chunks_files:
            raise FileNotFoundError(
                "No chunks files found. Please run preprocessing first:\n"
                "  python scripts/process_use_cases.py\n"
                "  python scripts/process_legal_documents.py\n"
                "Or specify chunks_files parameter."
            )

    # Ensure it's a list
    if isinstance(chunks_files, str):
        chunks_files = [chunks_files]

    # Initialize vector store
    vs = LegalVectorStore(
        persist_directory=persist_directory,
        embedding_provider="huggingface",  # Multilingual for Arabic
        use_reranking=True,
    )

    # Load processed chunks from all sources
    all_documents = []

    for chunks_file in chunks_files:
        logger.info(f"Loading chunks from: {chunks_file}")
        documents = vs.load_documents_from_chunks(chunks_file=chunks_file)
        all_documents.extend(documents)
        logger.info(f"  → Loaded {len(documents)} documents")

    logger.info(f"✓ Total documents loaded: {len(all_documents)}")

    # Build vector store
    vs.build_vectorstore(all_documents)

    # Build hybrid retriever
    vs.build_hybrid_retriever(all_documents, weights=(0.5, 0.5))

    logger.info("✓ Vector store pipeline complete!")

    return vs


if __name__ == "__main__":
    build_vectorstore_pipeline()
