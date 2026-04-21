"""
Agent 4: Reasoning Agent
Constructs legal arguments by applying law to facts
"""

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput


class ReasoningAgent(BaseAgent):
    """
    Agent 4: Reasoning Agent

    Responsibility: Construct legal arguments by applying law to facts
    Input: User's facts + extracted legal provisions
    Output: Legal reasoning and argument construction
    Technical Approach: Chain-of-thought reasoning using Gemini 3.0 or Claude 3.5 Sonnet
    """

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.2):
        super().__init__(
            role=AgentRole.REASONING,
            model=model,
            temperature=temperature,  # Slightly higher for nuanced reasoning
        )

    def get_system_prompt(self) -> str:
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

Structure your reasoning:
- Main legal argument
- Supporting provisions and facts
- Analysis of each legal element
- Counterarguments (if any)
- Conclusion and legal consequences

Use clear, logical chain-of-thought reasoning. Be thorough and precise."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Construct legal reasoning."""

        try:
            # Get context
            structured_query = agent_input.context.get("structured_query", {})
            provisions = agent_input.context.get("provisions", [])

            if not provisions:
                logger.warning("No provisions available for reasoning")

            # Build reasoning
            reasoning = self._construct_reasoning(structured_query, provisions)

            logger.info("Legal reasoning constructed")

            output = AgentOutput(
                result={"reasoning": reasoning},
                metadata={
                    "agent": self.role.value,
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

    def _construct_reasoning(
        self, structured_query: dict, provisions: list
    ) -> str:
        """Construct legal reasoning."""

        # Format provisions
        provisions_text = "\n\n".join(
            [
                f"Provision {i+1}:\n"
                f"Article: {prov.get('article_number', 'N/A')}\n"
                f"Text: {prov.get('provision_text', '')}\n"
                f"Principle: {prov.get('legal_principle', '')}\n"
                f"Relevance: {prov.get('relevance', '')}"
                for i, prov in enumerate(provisions[:10])  # Top 10 provisions
            ]
        )

        # Format facts
        facts = structured_query.get("facts", [])
        facts_text = "\n".join([f"- {fact}" for fact in facts])

        # Legal questions
        questions = structured_query.get("legal_questions", [])
        questions_text = "\n".join([f"- {q}" for q in questions])

        # Build prompt
        user_message = f"""Case Analysis Request:

LEGAL DOMAIN: {structured_query.get('legal_domain', 'N/A')}

FACTS:
{facts_text}

LEGAL QUESTIONS:
{questions_text}

KEY ENTITIES: {', '.join(structured_query.get('key_entities', []))}

APPLICABLE LEGAL PROVISIONS:
{provisions_text}

---

Apply the above legal provisions to the stated facts and construct a comprehensive legal reasoning that:

1. Identifies which provisions apply to this case
2. Breaks down the legal requirements of each provision
3. Maps the facts to each legal requirement
4. Determines if requirements are satisfied
5. Addresses each legal question
6. Draws clear legal conclusions
7. Discusses legal consequences and remedies

Use chain-of-thought reasoning. Be thorough and systematic."""

        # Invoke LLM with extended thinking
        reasoning = self.invoke_llm(user_message)

        return reasoning
