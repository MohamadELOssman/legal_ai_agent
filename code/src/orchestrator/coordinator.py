"""
Multi-Agent Orchestrator / Coordinator (Agent 7)
Manages workflow between all agents and ensures quality
"""

from typing import Dict, Any, Optional
from loguru import logger

from src.agents.base_agent import AgentInput, AgentOutput
from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.citation_agent import CitationAgent
from src.agents.writing_agent import WritingAgent


class LegalAIOrchestrator:
    """
    Agent 7: Coordinator Agent + Orchestration System

    Responsibility: Manage workflow between all agents and ensure quality
    Input: User query and outputs from all agents
    Output: Orchestrated multi-agent workflow and final validated response
    Technical Approach: Custom Python orchestration logic using Claude/Gemini function calling
    """

    def __init__(
        self,
        primary_model: str = "claude-sonnet-4.5",
        enable_logging: bool = True,
    ):
        self.primary_model = primary_model
        self.enable_logging = enable_logging

        # Initialize all agents
        logger.info("Initializing multi-agent system...")

        self.query_agent = QueryUnderstandingAgent(model=primary_model)
        self.research_agent = ResearchAgent(model=primary_model)
        self.analysis_agent = AnalysisAgent(model=primary_model)
        self.reasoning_agent = ReasoningAgent(model=primary_model)
        self.citation_agent = CitationAgent(model=primary_model)
        self.writing_agent = WritingAgent(model=primary_model)

        logger.info("Multi-agent system initialized successfully")

    def process_query(
        self, user_query: str, max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Process legal query through all agents.

        Workflow:
        1. Query Understanding Agent - Parse query
        2. Research Agent - Retrieve relevant documents
        3. Analysis Agent - Extract provisions
        4. Reasoning Agent - Construct legal argument
        5. Citation Agent - Format citations
        6. Writing Agent - Generate memorandum
        7. Validation - Final quality check
        """

        logger.info(f"Processing query: {user_query[:100]}...")

        execution_trace = []

        try:
            # Step 1: Query Understanding
            logger.info("STEP 1/6: Query Understanding...")
            query_output = self._run_query_understanding(user_query)
            execution_trace.append(("query_understanding", query_output.success))

            if not query_output.success:
                raise Exception(f"Query understanding failed: {query_output.error}")

            structured_query = query_output.result

            # Step 2: Research (RAG)
            logger.info("STEP 2/6: Research (RAG Retrieval)...")
            research_output = self._run_research(user_query, structured_query)
            execution_trace.append(("research", research_output.success))

            if not research_output.success:
                raise Exception(f"Research failed: {research_output.error}")

            documents = research_output.result.get("documents", [])

            # Step 3: Analysis
            logger.info("STEP 3/6: Legal Analysis...")
            analysis_output = self._run_analysis(
                user_query, structured_query, documents
            )
            execution_trace.append(("analysis", analysis_output.success))

            if not analysis_output.success:
                raise Exception(f"Analysis failed: {analysis_output.error}")

            provisions = analysis_output.result.get("provisions", [])

            # Step 4: Reasoning
            logger.info("STEP 4/6: Legal Reasoning...")
            reasoning_output = self._run_reasoning(
                user_query, structured_query, provisions
            )
            execution_trace.append(("reasoning", reasoning_output.success))

            if not reasoning_output.success:
                logger.warning(f"Reasoning failed: {reasoning_output.error}")
                reasoning_text = ""
            else:
                reasoning_text = reasoning_output.result.get("reasoning", "")

            # Step 5: Citation
            logger.info("STEP 5/6: Citation Formatting...")
            citation_output = self._run_citation(structured_query, provisions)
            execution_trace.append(("citation", citation_output.success))

            citations = citation_output.result.get("citations", [])

            # Step 6: Writing
            logger.info("STEP 6/6: Generating Legal Memorandum...")
            writing_output = self._run_writing(
                user_query, structured_query, provisions, reasoning_text, citations
            )
            execution_trace.append(("writing", writing_output.success))

            if not writing_output.success:
                raise Exception(f"Writing failed: {writing_output.error}")

            memorandum = writing_output.result.get("memorandum", "")

            # Final validation
            logger.info("Performing final validation...")
            validation_result = self._validate_output(memorandum, citations)

            # Build final result
            result = {
                "memorandum": memorandum,
                "structured_query": structured_query,
                "provisions": provisions,
                "citations": citations,
                "reasoning": reasoning_text,
                "documents_retrieved": len(documents),
                "validation": validation_result,
                "execution_trace": execution_trace,
                "success": True,
            }

            logger.info("✓ Query processing complete!")
            return result

        except Exception as e:
            logger.error(f"✗ Query processing failed: {e}")
            return {
                "memorandum": "",
                "error": str(e),
                "execution_trace": execution_trace,
                "success": False,
            }

    def _run_query_understanding(self, query: str) -> AgentOutput:
        """Run query understanding agent."""
        agent_input = AgentInput(
            query=query, context={}, metadata={"step": "query_understanding"}
        )
        return self.query_agent.process(agent_input)

    def _run_research(
        self, query: str, structured_query: dict
    ) -> AgentOutput:
        """Run research agent."""
        agent_input = AgentInput(
            query=query,
            context={"structured_query": structured_query, "top_k": 5},
            metadata={"step": "research"},
        )
        return self.research_agent.process(agent_input)

    def _run_analysis(
        self, query: str, structured_query: dict, documents: list
    ) -> AgentOutput:
        """Run analysis agent."""
        agent_input = AgentInput(
            query=query,
            context={
                "structured_query": structured_query,
                "retrieved_documents": documents,
            },
            metadata={"step": "analysis"},
        )
        return self.analysis_agent.process(agent_input)

    def _run_reasoning(
        self, query: str, structured_query: dict, provisions: list
    ) -> AgentOutput:
        """Run reasoning agent."""
        agent_input = AgentInput(
            query=query,
            context={"structured_query": structured_query, "provisions": provisions},
            metadata={"step": "reasoning"},
        )
        return self.reasoning_agent.process(agent_input)

    def _run_citation(
        self, structured_query: dict, provisions: list
    ) -> AgentOutput:
        """Run citation agent."""
        agent_input = AgentInput(
            query="",  # No query needed for citation
            context={"structured_query": structured_query, "provisions": provisions},
            metadata={"step": "citation"},
        )
        return self.citation_agent.process(agent_input)

    def _run_writing(
        self,
        query: str,
        structured_query: dict,
        provisions: list,
        reasoning: str,
        citations: list,
    ) -> AgentOutput:
        """Run writing agent."""
        agent_input = AgentInput(
            query=query,
            context={
                "structured_query": structured_query,
                "provisions": provisions,
                "reasoning": reasoning,
                "citations": citations,
            },
            metadata={"step": "writing"},
        )
        return self.writing_agent.process(agent_input)

    def _validate_output(
        self, memorandum: str, citations: list
    ) -> Dict[str, Any]:
        """Validate final output."""

        validation = {
            "memorandum_length": len(memorandum),
            "has_content": len(memorandum) > 100,
            "num_citations": len(citations),
            "has_citations": len(citations) > 0,
            "is_valid": len(memorandum) > 100 and len(citations) > 0,
        }

        return validation


def main():
    """Test the orchestrator."""
    orchestrator = LegalAIOrchestrator()

    # Test query in Arabic
    test_query = """
    هناك شركة تجارية في بيروت تُدعى "شركة النور للتجارة"، كان المدير المالي مسؤولاً عن إدارة الحسابات والتحويلات المالية خلال سنة 2024.
    قام المدير المالي بالتصرف بالأموال الموجودة في حساب الشركة، حيث قام بتحويل مبلغ 150,000 دولار إلى حسابه الشخصي.
    السؤال: ما هي المسؤولية المدنية والجزائية للمدير المالي؟
    """

    result = orchestrator.process_query(test_query)

    if result["success"]:
        print("\n" + "=" * 80)
        print("LEGAL MEMORANDUM")
        print("=" * 80)
        print(result["memorandum"])
        print("\n" + "=" * 80)
        print(f"Citations: {len(result['citations'])}")
        print(f"Provisions analyzed: {len(result['provisions'])}")
        print(f"Documents retrieved: {result['documents_retrieved']}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
