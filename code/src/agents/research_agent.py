"""
Agent 2: Research Agent
Retrieves relevant Lebanese legal texts from the RAG database
"""

from typing import List, Dict, Any
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL
from src.rag.vectorstore import LegalVectorStore


class ResearchAgent(BaseAgent):
    """
    Agent 2: Research Agent

    Responsibility: Retrieve relevant Lebanese legal texts from the database
    Input: Structured query from Agent 1
    Output: Top-k relevant articles from Lebanese Penal Code and court decisions
    Technical Approach: RAG using Chroma vector database with pure semantic search,
                        separate pools for legal articles vs court rulings.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        vectorstore: LegalVectorStore = None,
    ):
        super().__init__(
            role=AgentRole.RESEARCH,
            model=model,
            temperature=temperature,
        )

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

    def get_system_prompt(self) -> str:
        return """You are a Research Agent for Lebanese Legal Research.
Your task is to retrieve relevant Lebanese Penal Code articles and court rulings."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Retrieve relevant legal documents using a single direct semantic search."""

        try:
            structured_query = agent_input.context.get("structured_query", {})

            # Use original_query from structured_query, fall back to raw agent input
            query = (
                structured_query.get("original_query")
                or agent_input.query
                or ""
            )

            top_k = agent_input.metadata.get("k") or agent_input.context.get("top_k", 5)
            score_threshold = agent_input.metadata.get("score_threshold", 0.3)

            orch = agent_input.metadata.get("orchestrator", {})
            research_mode = orch.get("research", {}).get("mode", "articles_and_cases")

            retrieved_documents = self._retrieve_documents(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                research_mode=research_mode,
            )

            logger.info(f"Retrieved {len(retrieved_documents)} legal documents")

            output = AgentOutput(
                result={
                    "retrieved_documents": retrieved_documents,
                    "search_query": query,
                    "total_retrieved": len(retrieved_documents),
                },
                metadata={
                    "agent": self.role.value,
                    "retrieval_strategy": "semantic",
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
                result={"retrieved_documents": []},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        research_mode: str = "articles_and_cases",
    ) -> List[Dict[str, Any]]:
        """
        Single-pass semantic search with separate pools for articles and rulings.

          articles_only       — legal_code pool only
          articles_and_cases  — legal_code pool + court_ruling pool
        """
        article_docs = []
        ruling_docs  = []

        # ── Part 1: Legal code articles ───────────────────────────────────────
        try:
            results = self.vectorstore.search(
                query=query,
                k=top_k,
                strategy="semantic",
                use_reranking=False,
                score_threshold=score_threshold,
                filter_dict={"source_type": "legal_code"},
            )
            article_docs = [
                {
                    "content":     doc.page_content,
                    "metadata":    doc.metadata,
                    "source_query": query,
                    "result_type": "legal_article",
                }
                for doc in results
            ]
        except Exception as e:
            logger.warning(f"Legal-code search failed: {e}")

        # ── Part 2: Court rulings (case_analysis mode only) ───────────────────
        if research_mode == "articles_and_cases":
            try:
                results = self.vectorstore.search(
                    query=query,
                    k=4,
                    strategy="semantic",
                    use_reranking=False,
                    score_threshold=0.3,
                    filter_dict={"source_type": "court_ruling"},
                )
                ruling_docs = [
                    {
                        "content":     doc.page_content,
                        "metadata":    doc.metadata,
                        "source_query": query,
                        "result_type": "court_ruling",
                    }
                    for doc in results
                ]
            except Exception as e:
                logger.warning(f"Court-ruling search failed: {e}")

        logger.info(
            f"Research [{research_mode}]: "
            f"{len(article_docs)} articles + {len(ruling_docs)} rulings"
        )
        return article_docs + ruling_docs
