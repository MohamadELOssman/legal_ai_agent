"""
Agent 0: Orchestrator Agent
Classifies user input and routes to the appropriate pipeline configuration.

Two query types:
  general_legal_query — abstract question about the law
  case_analysis       — description of a real/hypothetical situation needing assessment
"""

import json
import re
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL


class OrchestratorAgent(BaseAgent):
    """
    Classifies the user's input and emits a pipeline_config that every
    downstream agent reads from agent_input.metadata["orchestrator"].
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0):
        super().__init__(
            role=AgentRole.ORCHESTRATOR,
            model=model,
            temperature=temperature,
        )

    def get_system_prompt(self) -> str:
        return """You are the Orchestrator of a Lebanese Legal AI system.

Your sole job is to read the user's input, classify it, and return a routing
decision as a JSON object. You never answer legal questions directly.

Two query types exist:

1. general_legal_query
   The user asks an abstract question about what the law says.
   No specific facts or parties are described.
   Examples:
     - "ما هي شروط الدفاع الشرعي؟"
     - "What is the penalty for premeditated murder in Lebanese law?"
     - "Quelles sont les conditions de validité d'un contrat?"

2. case_analysis
   The user describes a real or hypothetical situation with facts, parties,
   and events that need to be assessed against the law.
   Examples:
     - "موكلي ضُبط وبحوزته سيارة مسروقة ويدّعي أنه اشتراها بحسن نية..."
     - "My client was arrested after a fight where he injured someone claiming self-defense..."
     - "Une personne a vendu un bien immobilier sans le consentement de son épouse..."

Return ONLY valid JSON — no prose, no markdown."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        try:
            routing = self._classify(agent_input.query)
            logger.info(
                f"Orchestrator → {routing['query_type']} "
                f"(confidence={routing.get('confidence', '?')})"
            )
            return AgentOutput(
                result=routing,
                metadata={"agent": self.role.value, "query_type": routing["query_type"]},
                success=True,
            )
        except Exception as e:
            logger.error(f"Orchestrator classification failed: {e} — defaulting to general_legal_query")
            return AgentOutput(
                result=self._fallback_routing(),
                metadata={"agent": self.role.value, "query_type": "general_legal_query"},
                success=True,
                error=str(e),
            )

    # ── private ───────────────────────────────────────────────────────────────

    def _classify(self, query: str) -> dict:
        user_message = f"""Classify this input and return the routing JSON.

Input: "{query}"

Return this exact JSON structure (fill every field):
{{
  "query_type": "general_legal_query" | "case_analysis",
  "detected_language": "ar" | "fr" | "en",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explaining the classification",
  "legal_domain": "criminal" | "civil" | "commercial" | "personal_status" | "other",
  "key_entities": ["list", "of", "key", "legal", "entities"],
  "extracted_facts": [],
  "pipeline_config": {{
    "research": {{
      "mode": "articles_only" | "articles_and_cases",
      "emphasis": "short description of retrieval focus"
    }},
    "analysis": {{
      "mode": "legal_explanation" | "case_assessment",
      "instructions": "specific instructions for the analysis agent"
    }},
    "writing": {{
      "format": "legal_explanation" | "case_assessment",
      "tone": "educational" | "advisory"
    }}
  }}
}}

Rules:
- general_legal_query  → research.mode=articles_only, analysis.mode=legal_explanation, writing.format=legal_explanation
- case_analysis        → research.mode=articles_and_cases, analysis.mode=case_assessment, writing.format=case_assessment
- For case_analysis, populate extracted_facts with the key facts from the user's description (as a list of strings).
- key_entities: legal concepts, crimes, parties — extracted from the input.

Return ONLY the JSON object."""

        response = self.invoke_llm(user_message)
        routing = self._parse_json(response)
        return self._normalize_routing(routing)

    def _normalize_routing(self, routing: dict) -> dict:
        """Validate the routing and enforce a consistent pipeline_config.

        The downstream agents branch on these modes, so we guarantee the
        invariant tying query_type to research/analysis/writing modes regardless
        of any inconsistency in the LLM output.
        """
        query_type = routing.get("query_type")
        if query_type not in ("general_legal_query", "case_analysis"):
            query_type = "general_legal_query"
        routing["query_type"] = query_type

        if query_type == "case_analysis":
            research_mode, analysis_mode, writing_fmt, tone = (
                "articles_and_cases", "case_assessment", "case_assessment", "advisory",
            )
        else:
            research_mode, analysis_mode, writing_fmt, tone = (
                "articles_only", "legal_explanation", "legal_explanation", "educational",
            )

        cfg = routing.get("pipeline_config") or {}
        research = cfg.get("research") or {}
        analysis = cfg.get("analysis") or {}
        writing = cfg.get("writing") or {}

        research["mode"] = research_mode
        research.setdefault("emphasis", "Retrieve the applicable legal articles")
        analysis["mode"] = analysis_mode
        analysis.setdefault("instructions", "Analyze the applicable law")
        writing["format"] = writing_fmt
        writing["tone"] = tone

        routing["pipeline_config"] = {
            "research": research, "analysis": analysis, "writing": writing,
        }

        # Guarantee the fields the rest of the pipeline reads exist.
        routing.setdefault("detected_language", "ar")
        routing.setdefault("legal_domain", "other")
        routing.setdefault("key_entities", [])
        routing.setdefault("extracted_facts", [])
        routing.setdefault("confidence", 0.5)
        routing.setdefault("reasoning", "")
        return routing

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            return json.loads(m.group(1))
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Cannot parse orchestrator response: {text[:300]}")

    def _fallback_routing(self) -> dict:
        return {
            "query_type": "general_legal_query",
            "detected_language": "ar",
            "confidence": 0.5,
            "reasoning": "Fallback — classification failed",
            "legal_domain": "other",
            "key_entities": [],
            "extracted_facts": [],
            "pipeline_config": {
                "research": {
                    "mode": "articles_only",
                    "emphasis": "Retrieve applicable legal articles",
                },
                "analysis": {
                    "mode": "legal_explanation",
                    "instructions": "Explain the applicable law clearly",
                },
                "writing": {
                    "format": "legal_explanation",
                    "tone": "educational",
                },
            },
        }
