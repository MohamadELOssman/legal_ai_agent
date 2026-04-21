"""
Agent 2: Research Agent
Retrieves relevant Lebanese legal texts from the RAG database
"""

from typing import List, Dict, Any
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.rag.vectorstore import LegalVectorStore
from src.rag.query_expansion import TrilingualQueryExpander


class ResearchAgent(BaseAgent):
    """
    Agent 2: Research Agent

    Responsibility: Retrieve relevant Lebanese legal texts from the database
    Input: Structured query from Agent 1
    Output: Top-k relevant articles from Lebanese Code of Obligations and court decisions
    Technical Approach: RAG (Retrieval-Augmented Generation) using vector database
                       (Chroma) with hybrid retrieval (semantic + BM25)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
        vectorstore: LegalVectorStore = None,
    ):
        super().__init__(
            role=AgentRole.RESEARCH,
            model=model,
            temperature=temperature,
        )

        # Initialize or load vector store
        if vectorstore is None:
            self.vectorstore = LegalVectorStore()
            try:
                self.vectorstore.load_vectorstore()
            except FileNotFoundError:
                logger.warning(
                    "Vector store not found. Build it first using build_vectorstore.py"
                )
        else:
            self.vectorstore = vectorstore

        # Initialize query expander for trilingual search
        self.query_expander = TrilingualQueryExpander()

    def get_system_prompt(self) -> str:
        return """You are a Research Agent for Lebanese Legal Research.

Your task is to formulate effective search queries for retrieving relevant legal documents.

Given a structured legal query, you should:
1. Generate optimal search queries in the appropriate language(s)
2. Consider legal synonyms and related concepts
3. Account for trilingual nature (Arabic, French, English)
4. Focus on Lebanese legal terminology

Lebanese legal sources available:
- Lebanese Code of Obligations and Contracts
- Lebanese Penal Code
- Lebanese Criminal Procedure Code
- Court of Cassation decisions
- Legal commentaries

Return search queries that will retrieve the most relevant legal provisions."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Retrieve relevant legal documents."""

        try:
            # Extract structured query from context
            structured_query = agent_input.context.get("structured_query", {})

            # Get retrieval parameters from metadata (preferred) or context (fallback)
            top_k = agent_input.metadata.get("k") or agent_input.context.get("top_k", 5)
            score_threshold = agent_input.metadata.get("score_threshold", 0.6)

            # Generate search queries
            search_queries = self._generate_search_queries(structured_query)

            # Perform retrieval
            retrieved_documents = self._retrieve_documents(
                search_queries, top_k=top_k, score_threshold=score_threshold
            )

            logger.info(f"Retrieved {len(retrieved_documents)} legal documents (threshold: {score_threshold})")

            output = AgentOutput(
                result={
                    "retrieved_documents": retrieved_documents,  # Consistent key name
                    "search_queries": search_queries,
                    "total_retrieved": len(retrieved_documents),
                },
                metadata={
                    "agent": self.role.value,
                    "retrieval_strategy": "hybrid",
                    "num_queries": len(search_queries),
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Research agent failed: {e}")
            return AgentOutput(
                result={"retrieved_documents": []},  # Consistent key name
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _generate_search_queries(self, structured_query: Dict[str, Any]) -> List[str]:
        """Generate search queries from structured query."""

        # Extract key information
        legal_domain = structured_query.get("legal_domain", "")
        key_entities = structured_query.get("key_entities", [])
        legal_questions = structured_query.get("legal_questions", [])
        language = structured_query.get("language", "ar")

        # Build search queries
        queries = []

        # Query 1: Legal domain + key entities
        if legal_domain and key_entities:
            entities_str = " ".join(key_entities[:3])  # Top 3 entities
            queries.append(f"{legal_domain} {entities_str}")

        # Query 2: Each legal question
        for question in legal_questions[:2]:  # Top 2 questions
            queries.append(question)

        # Query 3: Combine key entities
        if len(key_entities) >= 2:
            queries.append(" ".join(key_entities[:3]))

        # Fallback: Use original query
        if not queries:
            queries.append(structured_query.get("original_query", ""))

        # ENHANCEMENT: Expand queries with trilingual synonyms
        # This improves recall for multilingual legal corpus
        expanded_queries = []
        for query in queries:
            expanded = self.query_expander.expand_query(query, max_expansions=3)
            expanded_queries.append(expanded)

        logger.debug(f"Generated {len(queries)} queries → {len(expanded_queries)} expanded")
        return expanded_queries

    def _retrieve_documents(
        self, search_queries: List[str], top_k: int = 5, score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for each search query with similarity threshold filtering."""

        all_documents = []
        seen_chunks = set()  # Deduplicate

        for query in search_queries:
            try:
                # Perform hybrid search with reranking
                results = self.vectorstore.search(
                    query=query,
                    k=top_k,
                    strategy="hybrid",
                    use_reranking=True,  # Enable reranking for better precision
                    score_threshold=score_threshold  # Filter by similarity threshold
                )

                for doc in results:
                    # Create unique identifier
                    doc_id = f"{doc.metadata.get('article_number', '')}_{doc.metadata.get('chunk_index', '')}"

                    if doc_id not in seen_chunks:
                        all_documents.append(
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "source_query": query,
                            }
                        )
                        seen_chunks.add(doc_id)

            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        # Limit total documents
        return all_documents[:top_k * 2]  # Return up to 2x top_k
