"""
Agent 4: Reasoning Agent
Constructs legal arguments by applying law to facts.

Mode-aware (driven by the Orchestrator):
  legal_explanation — explain how the provisions answer a general legal question
  case_assessment   — apply provisions to specific facts and weigh precedent
"""

from typing import List, Dict, Any
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL
from src.utils.prompt_loader import load_agent_prompt


class ReasoningAgent(BaseAgent):
    """
    Agent 4: Reasoning Agent

    Responsibility: Construct legal arguments by applying law to facts
    Input: Structured query + extracted provisions (+ similar cases for case_assessment)
    Output: Legal reasoning / argument construction
    Technical Approach: Chain-of-thought reasoning using Claude Sonnet 4.5
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2,
                 max_tokens: int = 2000):
        super().__init__(
            role=AgentRole.REASONING,
            model=model,
            temperature=temperature,  # Slightly higher for nuanced reasoning
            # Cap output: unbounded chain-of-thought was generating ~4k-token
            # (~15k-char) reasoning that dominated end-to-end latency and inflated
            # the writing agent's input. A focused 2k-token cap keeps the legal
            # reasoning while making batch/statistical runs practical.
            max_tokens=max_tokens,
        )

    def get_system_prompt(self) -> str:
        prompt = load_agent_prompt("reasoning")
        if prompt:
            return prompt

        return """You are a Reasoning Agent for Lebanese Legal Analysis.

Your task is to apply legal provisions to the specific facts of the case and construct sound legal arguments.

Reasoning methodology:
1. Identify the relevant legal provisions
2. Extract the legal requirements/elements from each provision
3. Map the facts to these legal requirements
4. Determine if each requirement is satisfied
5. Draw legal conclusions
6. Consider counterarguments
7. Assess strength of the legal position

Legal reasoning principles (Lebanese civil law):
- Literal interpretation of statutes first
- Consider legislative intent
- Apply established legal principles
- Reference court precedents where applicable
- Consider both civil and criminal liability when relevant
- Account for overlapping legal provisions

Use clear, logical chain-of-thought reasoning. Be thorough and precise.
Never invent article numbers or provisions that are not provided to you."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Construct legal reasoning from the analysed provisions."""

        try:
            orch = agent_input.metadata.get("orchestrator", {})
            mode = orch.get("analysis", {}).get("mode", "legal_explanation")

            structured_query = agent_input.context.get("structured_query", {})

            # Provisions live in the analysis output in the pipeline; fall back to a
            # directly-supplied `provisions` key (used by the individual-agent tester).
            analysis_results = agent_input.context.get("analysis_results", {}) or {}
            provisions = (
                agent_input.context.get("provisions")
                or analysis_results.get("provisions", [])
            )
            similar_cases = (
                agent_input.context.get("similar_cases")
                or analysis_results.get("similar_cases", [])
            )
            case_assessment = (
                agent_input.context.get("case_assessment")
                or analysis_results.get("case_assessment", {})
            )

            # Facts: prefer the orchestrator's extracted facts, then the structured query.
            extracted_facts = orch.get("extracted_facts", []) or structured_query.get("facts", [])

            if not provisions:
                logger.warning("ReasoningAgent: no provisions available to reason over")

            reasoning = self._construct_reasoning(
                structured_query=structured_query,
                provisions=provisions,
                extracted_facts=extracted_facts,
                similar_cases=similar_cases,
                case_assessment=case_assessment,
                mode=mode,
            )

            logger.info(f"ReasoningAgent [{mode}] — {len(provisions)} provisions, {len(reasoning)} chars")

            output = AgentOutput(
                result={"reasoning": reasoning, "mode": mode},
                metadata={
                    "agent": self.role.value,
                    "mode": mode,
                    "provisions_analyzed": len(provisions),
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Reasoning agent failed: {e}")
            return AgentOutput(
                result={"reasoning": ""},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _format_provisions(self, provisions: List[Dict]) -> str:
        if not provisions:
            return "No provisions were extracted."
        parts = []
        for i, prov in enumerate(provisions[:10], 1):  # Top 10 provisions
            parts.append(
                f"Provision {i}:\n"
                f"Article: {prov.get('article_number', 'N/A')}\n"
                f"Text: {prov.get('provision_text', '')[:500]}\n"
                f"Principle: {prov.get('legal_principle', '')}\n"
                f"Relevance: {prov.get('application_reasoning', prov.get('relevance', ''))}"
            )
        return "\n\n".join(parts)

    def _format_cases(self, similar_cases: List[Dict]) -> str:
        if not similar_cases:
            return "No similar court rulings were retrieved."
        parts = []
        for i, c in enumerate(similar_cases[:3], 1):
            parts.append(
                f"Case {c.get('case_id', i)} | {c.get('court', 'N/A')} | {c.get('decision_date', 'N/A')}\n"
                f"Outcome: {c.get('outcome', 'N/A')} | Sentence: {c.get('sentence', 'N/A')}\n"
                f"Similarity: {c.get('similarity_reasoning', '')}"
            )
        return "\n\n".join(parts)

    def _construct_reasoning(
        self,
        structured_query: Dict,
        provisions: List[Dict],
        extracted_facts: List[str],
        similar_cases: List[Dict],
        case_assessment: Dict,
        mode: str,
    ) -> str:
        """Build the reasoning prompt for the requested mode and invoke the LLM."""

        provisions_text = self._format_provisions(provisions)
        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "See query."
        questions = structured_query.get("legal_questions", [])
        questions_text = "\n".join(f"- {q}" for q in questions) if questions else "Answer the user's question."

        if mode == "case_assessment":
            cases_text = self._format_cases(similar_cases)
            assessment = case_assessment or {}
            user_message = f"""Case reasoning task.

LEGAL DOMAIN: {structured_query.get('legal_domain', 'N/A')}

FACTS OF THE CASE:
{facts_text}

LEGAL QUESTIONS:
{questions_text}

APPLICABLE LEGAL PROVISIONS:
{provisions_text}

SIMILAR COURT RULINGS:
{cases_text}

PRELIMINARY ASSESSMENT:
- Strength: {assessment.get('strength_of_case', 'N/A')}
- Likely outcome: {assessment.get('likely_outcome', 'N/A')}

Apply the provisions to the stated facts using chain-of-thought reasoning:
1. For each applicable provision, break it into its legal elements
2. Map the facts to each element and decide whether it is satisfied
3. Weigh the similar rulings — do they support or weaken the position?
4. Address each legal question explicitly
5. State counterarguments and the strength of the case
6. Conclude with the likely legal consequences

Be precise and well-structured but concise — avoid repetition and filler; aim for
a focused legal analysis, not exhaustive prose. Do not introduce any article that is not listed above."""
        else:
            user_message = f"""Legal reasoning task.

LEGAL DOMAIN: {structured_query.get('legal_domain', 'N/A')}

ORIGINAL QUESTION: {structured_query.get('original_query', '')}

LEGAL QUESTIONS:
{questions_text}

APPLICABLE LEGAL PROVISIONS:
{provisions_text}

Using chain-of-thought reasoning:
1. Identify which provisions answer the question and why
2. Break down the requirements, conditions, and exceptions of each provision
3. Explain how they fit together into a coherent rule of law
4. Address each legal question explicitly
5. Conclude with a clear, direct statement of what the law provides

Be precise and well-structured but concise — avoid repetition and filler; aim for
a focused legal analysis, not exhaustive prose. Do not introduce any article that is not listed above."""

        return self.invoke_llm(user_message)
