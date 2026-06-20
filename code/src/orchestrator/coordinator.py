"""
Headless Multi-Agent Pipeline (canonical orchestration)

This is the Streamlit-free equivalent of the pipeline in app.py. It drives the
full 7-agent flow so the system can be run programmatically — for batch
evaluation, baseline comparison, scripting, and tests — without the web UI.

Flow:
  0. Orchestrator       — classify query, emit pipeline_config
  1. Query Understanding — structured query
  2. Research (RAG)     — retrieve articles (+ rulings for case_analysis)
  3. Analysis           — extract & ground provisions
  4. Reasoning          — apply law to facts / explain
  5. Citation           — format & validate citations
  6. Writing            — final memorandum

Both app.py and this module construct identical AgentInput payloads, so behaviour
is consistent across the UI and headless runs.
"""

import time
from typing import Dict, Any, List, Optional

from loguru import logger

from src.config import DEFAULT_MODEL
from src.agents.base_agent import AgentInput, AgentOutput
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.citation_agent import CitationAgent
from src.agents.writing_agent import WritingAgent


class LegalAIPipeline:
    """End-to-end multi-agent legal pipeline, runnable without Streamlit."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_k: int = 5,
        score_threshold: float = 0.3,
        vectorstore: Optional[Any] = None,
        load_vectorstore: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold

        logger.info(f"Initializing LegalAIPipeline (model={model})...")

        # Load the vector store once and share it with the research agent.
        if vectorstore is None and load_vectorstore:
            from src.rag.vectorstore import LegalVectorStore
            vectorstore = LegalVectorStore()
            try:
                vectorstore.load_vectorstore()
            except FileNotFoundError:
                logger.warning("Vector store not found — research will return no documents.")
        self.vectorstore = vectorstore

        # Construct agents once and reuse them across queries (efficient for batch).
        self.orchestrator = OrchestratorAgent(model=model)
        self.query_agent = QueryUnderstandingAgent(model=model, temperature=temperature)
        self.research_agent = ResearchAgent(model=model, temperature=temperature, vectorstore=vectorstore)
        self.analysis_agent = AnalysisAgent(model=model, temperature=temperature)
        self.reasoning_agent = ReasoningAgent(model=model, temperature=temperature)
        self.citation_agent = CitationAgent(model=model, temperature=temperature)
        self.writing_agent = WritingAgent(model=model, temperature=temperature)

        logger.info("LegalAIPipeline initialized successfully")

    # ── public API ────────────────────────────────────────────────────────────

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Run the full pipeline on a single query and return a structured result.

        Never raises on agent failure — failures are captured in the returned
        dict (`success`, `error`, `execution_trace`) so batch runs don't abort.
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        trace: List[Dict[str, Any]] = []
        timings: Dict[str, float] = {}

        # Fresh usage telemetry for this query.
        for agent in self._agents():
            agent.reset_usage()

        def _run(step: str, fn) -> AgentOutput:
            t0 = time.time()
            out = fn()
            timings[step] = round(time.time() - t0, 2)
            trace.append({"step": step, "success": out.success, "error": out.error})
            return out

        try:
            # Step 0 — Orchestrator
            out0 = _run("orchestrator", lambda: self.orchestrator.process(
                AgentInput(query=user_query, context={}, metadata={})
            ))
            routing = out0.result
            query_type = routing.get("query_type", "general_legal_query")
            pipeline_cfg = routing.get("pipeline_config", {})
            orch_meta = {**pipeline_cfg, "extracted_facts": routing.get("extracted_facts", [])}

            # Step 1 — Query Understanding
            out1 = _run("query_understanding", lambda: self.query_agent.process(
                AgentInput(query=user_query, context={}, metadata={"orchestrator": orch_meta})
            ))
            if not out1.success:
                return self._fail("query_understanding", out1.error, trace, timings)
            structured_query = out1.result

            # Step 2 — Research
            out2 = _run("research", lambda: self.research_agent.process(
                AgentInput(
                    query=user_query,
                    context={"structured_query": structured_query},
                    metadata={"k": self.top_k, "score_threshold": self.score_threshold,
                              "orchestrator": orch_meta},
                )
            ))
            if not out2.success:
                return self._fail("research", out2.error, trace, timings)
            documents = out2.result.get("retrieved_documents", [])

            # Step 3 — Analysis
            out3 = _run("analysis", lambda: self.analysis_agent.process(
                AgentInput(
                    query=user_query,
                    context={"structured_query": structured_query, "research_results": out2.result},
                    metadata={"orchestrator": orch_meta},
                )
            ))
            if not out3.success:
                return self._fail("analysis", out3.error, trace, timings)
            provisions = out3.result.get("provisions", [])

            # Step 4 — Reasoning (non-fatal: continue with empty reasoning if it fails)
            out4 = _run("reasoning", lambda: self.reasoning_agent.process(
                AgentInput(
                    query=user_query,
                    context={"structured_query": structured_query,
                             "research_results": out2.result,
                             "analysis_results": out3.result},
                    metadata={"orchestrator": orch_meta},
                )
            ))
            reasoning_text = out4.result.get("reasoning", "") if out4.success else ""

            # Step 5 — Citation
            out5 = _run("citation", lambda: self.citation_agent.process(
                AgentInput(
                    query=user_query,
                    context={"structured_query": structured_query,
                             "research_results": out2.result,
                             "analysis_results": out3.result,
                             "reasoning_results": out4.result},
                    metadata={"orchestrator": orch_meta},
                )
            ))
            citations = out5.result.get("citations", []) if out5.success else []
            validation_report = out5.result.get("validation_report", {}) if out5.success else {}

            # Step 6 — Writing (same safe context construction as app.py)
            writing_ctx = {
                "structured_query": structured_query,
                "provisions": provisions,
                "reasoning": reasoning_text,
                "citations": citations,
                "similar_cases": out3.result.get("similar_cases", []),
                "case_assessment": out3.result.get("case_assessment", {}),
            }
            out6 = _run("writing", lambda: self.writing_agent.process(
                AgentInput(query=user_query, context=writing_ctx,
                           metadata={"orchestrator": orch_meta})
            ))
            if not out6.success:
                return self._fail("writing", out6.error, trace, timings)
            memorandum = out6.result.get("memorandum", "")

            result = {
                "success": True,
                "query": user_query,
                "query_type": query_type,
                "routing": routing,
                "structured_query": structured_query,
                "documents_retrieved": len(documents),
                "provisions": provisions,
                "grounding": out3.metadata.get("grounding", {}),
                "reasoning": reasoning_text,
                "citations": citations,
                "citation_validation": validation_report,
                "memorandum": memorandum,
                "memorandum_format": out6.result.get("format", "legal_explanation"),
                "language": out6.result.get("language", structured_query.get("language", "ar")),
                "validation": self._validate_output(memorandum, citations),
                "execution_trace": trace,
                "timings": timings,
                "total_latency_s": round(sum(timings.values()), 2),
                "usage": self._collect_usage(),
            }
            logger.info(f"✓ Query complete in {result['total_latency_s']}s "
                        f"({query_type}, {len(citations)} citations)")
            return result

        except Exception as e:
            logger.error(f"✗ Pipeline crashed: {e}")
            return self._fail("pipeline", str(e), trace, timings)

    def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Run the pipeline over many queries (for evaluation / baselines)."""
        results = []
        for i, q in enumerate(queries, 1):
            logger.info(f"[batch {i}/{len(queries)}]")
            results.append(self.process_query(q))
        return results

    # ── helpers ───────────────────────────────────────────────────────────────

    def _agents(self):
        return [self.orchestrator, self.query_agent, self.research_agent,
                self.analysis_agent, self.reasoning_agent, self.citation_agent,
                self.writing_agent]

    def _collect_usage(self) -> Dict[str, Any]:
        """Aggregate token/cost/latency telemetry across all agents for this query."""
        per_agent = {}
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        for agent in self._agents():
            summary = agent.usage_summary()
            if summary["calls"] == 0:
                continue
            per_agent[agent.role.value] = summary
            for key in totals:
                totals[key] += summary[key]
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        return {"per_agent": per_agent, "totals": totals}

    @staticmethod
    def _fail(step: str, error: Optional[str], trace, timings) -> Dict[str, Any]:
        return {
            "success": False,
            "failed_step": step,
            "error": error,
            "memorandum": "",
            "execution_trace": trace,
            "timings": timings,
        }

    @staticmethod
    def _validate_output(memorandum: str, citations: list) -> Dict[str, Any]:
        return {
            "memorandum_length": len(memorandum),
            "has_content": len(memorandum) > 100,
            "num_citations": len(citations),
            "num_verified_citations": sum(1 for c in citations if c.get("verified")),
            "has_citations": len(citations) > 0,
            "is_valid": len(memorandum) > 100 and len(citations) > 0,
        }


# Backward-compatible alias for the previous class name.
LegalAIOrchestrator = LegalAIPipeline


def main():
    """Quick manual test of the headless pipeline."""
    pipeline = LegalAIPipeline()
    test_query = "ما هي عقوبة القتل العمد في القانون اللبناني؟"
    result = pipeline.process_query(test_query)

    if result["success"]:
        print("\n" + "=" * 80)
        print(f"QUERY TYPE: {result['query_type']}  |  LATENCY: {result['total_latency_s']}s")
        print(f"Documents: {result['documents_retrieved']}  |  "
              f"Provisions: {len(result['provisions'])}  |  "
              f"Citations: {result['validation']['num_verified_citations']}/"
              f"{result['validation']['num_citations']} verified")
        print("=" * 80)
        print(result["memorandum"])
    else:
        print(f"FAILED at {result.get('failed_step')}: {result.get('error')}")


if __name__ == "__main__":
    main()
