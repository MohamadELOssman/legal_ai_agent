"""
Agent 0: Orchestrator Agent
Classifies user input and routes to the appropriate pipeline configuration.

It identifies:
  • query_type — general_legal_query | case_analysis
  • user_type  — citizen | lawyer | judge  (who is asking → which OUTPUT SHAPE)

Output shape (writing format) by user type:
  citizen  → plain_answer      (a simple, clear answer)
  lawyer   → legal_explanation (general question) / case_assessment (a client's case)
  judge    → judicial_decision (facts given → the ruling is written)
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL

USER_TYPES = ("citizen", "lawyer", "judge")


# ── Structured-output schema (enforced via tool use; no regex JSON parsing) ──────

class _ResearchCfg(BaseModel):
    mode: str = Field("articles_only", description="articles_only | articles_and_cases")
    emphasis: str = ""


class _AnalysisCfg(BaseModel):
    mode: str = Field("legal_explanation", description="legal_explanation | case_assessment")
    instructions: str = ""


class _WritingCfg(BaseModel):
    format: str = Field("legal_explanation",
                        description="plain_answer | legal_explanation | case_assessment | judicial_decision")
    tone: str = Field("educational", description="plain | educational | advisory | judicial")


class _PipelineConfig(BaseModel):
    research: _ResearchCfg = Field(default_factory=_ResearchCfg)
    analysis: _AnalysisCfg = Field(default_factory=_AnalysisCfg)
    writing: _WritingCfg = Field(default_factory=_WritingCfg)


class RoutingDecision(BaseModel):
    """Routing decision for the legal pipeline."""
    query_type: str = Field(description="general_legal_query | case_analysis")
    user_type: str = Field("citizen", description="citizen | lawyer | judge — who is asking")
    detected_language: str = Field("ar", description="ar | fr | en")
    confidence: float = 0.5
    reasoning: str = Field("", description="one sentence explaining the classification")
    legal_domain: str = Field("other", description="criminal | civil | commercial | personal_status | other")
    key_entities: List[str] = Field(default_factory=list)
    extracted_facts: List[str] = Field(default_factory=list, description="key facts for case_analysis; empty otherwise")
    pipeline_config: _PipelineConfig = Field(default_factory=_PipelineConfig)


class OrchestratorAgent(BaseAgent):
    """
    Classifies the user's input and emits a pipeline_config that every
    downstream agent reads from agent_input.metadata["orchestrator"].
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0):
        super().__init__(role=AgentRole.ORCHESTRATOR, model=model, temperature=temperature)

    def get_system_prompt(self) -> str:
        return """You are the Orchestrator of a Lebanese Legal AI system.

Read the user's input, classify it, and return a routing decision. You never
answer legal questions directly.

Classify TWO things:

A) query_type:
   • general_legal_query — an abstract question about what the law says (no facts/parties).
   • case_analysis       — a real/hypothetical situation with facts to assess.

B) user_type — WHO is asking (this decides the shape of the final answer):
   • citizen — an ordinary person asking a legal question in plain terms.
       e.g. "شو عقوبة السرقة؟", "Can my landlord evict me without notice?"
   • lawyer  — a legal professional; often mentions "my client / my defendant",
       asks for an assessment or a defence strategy.
       e.g. "موكلي ضُبط وبحوزته سيارة مسروقة، كيف أدافع عنه؟"
   • judge   — presents the facts of a case and expects the DECISION/ruling to be written.
       e.g. "المدعى عليه قتل المجني عليه عمداً... أصدر الحكم", "Given these facts, render the verdict."

Return ONLY valid JSON — no prose, no markdown."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        try:
            override = agent_input.metadata.get("user_role")
            if override not in USER_TYPES:
                override = None
            routing = self._classify(agent_input.query, user_role=override)
            logger.info(
                f"Orchestrator → {routing['query_type']} / user={routing['user_type']} "
                f"→ writing={routing['pipeline_config']['writing']['format']} "
                f"(confidence={routing.get('confidence', '?')})"
            )
            return AgentOutput(
                result=routing,
                metadata={"agent": self.role.value, "query_type": routing["query_type"],
                          "user_type": routing["user_type"]},
                success=True,
            )
        except Exception as e:
            logger.error(f"Orchestrator classification failed: {e} — defaulting to citizen/general")
            return AgentOutput(
                result=self._fallback_routing(),
                metadata={"agent": self.role.value, "query_type": "general_legal_query"},
                success=True,
                error=str(e),
            )

    # ── private ───────────────────────────────────────────────────────────────

    def _classify(self, query: str, user_role: Optional[str] = None) -> dict:
        role_line = (f"\nThe user's role is KNOWN to be: {user_role}. Set user_type to '{user_role}'."
                     if user_role else "")
        user_message = f"""Classify this input and produce the routing decision.

Input: "{query}"{role_line}

Set query_type (general_legal_query | case_analysis) and user_type (citizen | lawyer | judge).
For case_analysis, populate extracted_facts with the key facts from the description.
key_entities: legal concepts, crimes, parties extracted from the input."""

        routing = self.invoke_structured(user_message, RoutingDecision).model_dump()
        if user_role:
            routing["user_type"] = user_role
        return self._normalize_routing(routing)

    def _normalize_routing(self, routing: dict) -> dict:
        """Validate and enforce a consistent pipeline_config from (user_type, query_type)."""
        query_type = routing.get("query_type")
        if query_type not in ("general_legal_query", "case_analysis"):
            query_type = "general_legal_query"
        routing["query_type"] = query_type

        user_type = routing.get("user_type")
        if user_type not in USER_TYPES:
            user_type = "citizen"
        routing["user_type"] = user_type

        is_case = query_type == "case_analysis"

        # Research + analysis: a judge and any case need articles + precedent cases.
        if user_type == "judge" or is_case:
            research_mode, analysis_mode = "articles_and_cases", "case_assessment"
        else:
            research_mode, analysis_mode = "articles_only", "legal_explanation"

        # Writing format (the OUTPUT SHAPE) depends on who is asking.
        if user_type == "judge":
            writing_fmt, tone = "judicial_decision", "judicial"
        elif user_type == "citizen":
            writing_fmt, tone = "plain_answer", "plain"
        else:  # lawyer
            writing_fmt, tone = ("case_assessment", "advisory") if is_case else ("legal_explanation", "educational")

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
        routing["pipeline_config"] = {"research": research, "analysis": analysis, "writing": writing}

        routing.setdefault("detected_language", "ar")
        routing.setdefault("legal_domain", "other")
        routing.setdefault("key_entities", [])
        routing.setdefault("extracted_facts", [])
        routing.setdefault("confidence", 0.5)
        routing.setdefault("reasoning", "")
        return routing

    def _fallback_routing(self) -> dict:
        return {
            "query_type": "general_legal_query",
            "user_type": "citizen",
            "detected_language": "ar",
            "confidence": 0.5,
            "reasoning": "Fallback — classification failed",
            "legal_domain": "other",
            "key_entities": [],
            "extracted_facts": [],
            "pipeline_config": {
                "research": {"mode": "articles_only", "emphasis": "Retrieve applicable legal articles"},
                "analysis": {"mode": "legal_explanation", "instructions": "Explain the applicable law clearly"},
                "writing": {"format": "plain_answer", "tone": "plain"},
            },
        }
