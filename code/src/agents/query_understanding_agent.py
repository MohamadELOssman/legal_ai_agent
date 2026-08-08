"""
Agent 1: Query Understanding Agent
Parses and understands legal questions in Arabic, French, or English
"""

from pydantic import BaseModel, Field
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL


class StructuredQuery(BaseModel):
    """Structured representation of a legal query."""

    original_query: str = Field(description="Original user query")
    language: str = Field(description="Detected language (ar, fr, en)")
    legal_domain: str = Field(description="Legal domain (contract law, criminal law, etc)")
    key_entities: list[str] = Field(description="Key legal entities mentioned")
    intent: str = Field(description="User intent (legal advice, case analysis, etc)")
    facts: list[str] = Field(description="Relevant facts from the query")
    legal_questions: list[str] = Field(description="Specific legal questions to answer")


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent 1: Query Understanding Agent

    Responsibility: Parse and understand legal questions in any of the three languages
    Input: User's legal question in any language
    Output: Structured query identifying legal domain, key entities, and intent
    Technical Approach: Multilingual NLP using models like multilingual-e5-large
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        super().__init__(
            role=AgentRole.QUERY_UNDERSTANDING,
            model=model,
            temperature=temperature,
        )

    def get_system_prompt(self) -> str:
        # Load from external file
        from src.utils.prompt_loader import load_agent_prompt

        prompt = load_agent_prompt("query_understanding")

        # Fallback to a compact CRAFT prompt if the file is not found.
        if not prompt:
            return """# CONTEXT
You are the entry stage of a Lebanese CRIMINAL-law AI (Penal Code + Code of Criminal Procedure),
a civil-law, trilingual jurisdiction; queries may be in Arabic, French, or English or mix them.
Your structured output drives retrieval, so its precision decides whether the right law is found.
# ROLE
You are the Query Understanding Agent: you parse a raw question into a precise structure; you do
not answer it.
# ACTION
Detect the language; identify the legal domain + specific topic; extract key_entities as the CORE
legal concepts/facts (offence, act, penalty, parties) EXCLUDING boilerplate like "the law",
"the Penal Code", or "Lebanese law"; determine intent; extract the facts; and formulate the
precise legal question(s).
# FORMAT
Return these fields (list fields = SIMPLE STRINGS only): original_query, language (ar/fr/en),
legal_domain, key_entities, intent, facts, legal_questions.
# TARGET
Your consumers are the retrieval and analysis agents — be precise, concept-focused, and faithful
to the query's language."""

        return prompt

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process user query and return structured understanding."""

        try:
            user_message = f"""Analyze this legal query and produce a structured understanding.

Query: {agent_input.query}

Focus on Lebanese CRIMINAL law (Penal Code + Code of Criminal Procedure). Extract
key_entities, facts, and legal_questions as lists of plain strings. For key_entities, capture
the CORE legal concepts and facts (the offence, the act, the penalty, parties) and EXCLUDE
generic boilerplate such as "the law", "legal texts", "articles", "the Penal Code", or
"Lebanese law" — those pollute retrieval."""

            # Schema-validated output (tool use) — no manual JSON parsing needed.
            validated_query = self.invoke_structured(user_message, StructuredQuery)

            logger.info(
                f"Query understood - Language: {validated_query.language}, "
                f"Domain: {validated_query.legal_domain}"
            )

            output = AgentOutput(
                result=validated_query.model_dump(),
                metadata={
                    "agent": self.role.value,
                    "model": self.model_name,
                    "detected_language": validated_query.language,
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            return AgentOutput(
                result={},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )
