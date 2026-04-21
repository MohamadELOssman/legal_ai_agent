"""
Agent 1: Query Understanding Agent
Parses and understands legal questions in Arabic, French, or English
"""

import json
from typing import Dict, Any

from pydantic import BaseModel, Field
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput


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

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.1):
        super().__init__(
            role=AgentRole.QUERY_UNDERSTANDING,
            model=model,
            temperature=temperature,
        )

    def get_system_prompt(self) -> str:
        # Load from external file
        from src.utils.prompt_loader import load_agent_prompt

        prompt = load_agent_prompt("query_understanding")

        # Fallback to hardcoded if file not found
        if not prompt:
            return """You are a Query Understanding Agent for a Lebanese Legal AI System.

Your task is to parse and understand legal questions in Arabic, French, or English (or code-switching between them).

Lebanese legal context:
- Civil law jurisdiction (not common law)
- Trilingual legal system (Arabic is official, French widely used, English for international)
- Primary focus: Lebanese Code of Obligations and Contracts (Contract Law)
- Also covers: Penal Code, Criminal Procedure Code

Your responsibilities:
1. Detect the language(s) used in the query
2. Identify the legal domain (contract law, criminal law, civil liability, etc.)
3. Extract key legal entities (parties, amounts, dates, legal concepts)
4. Determine user intent (legal advice, case analysis, legal research, etc.)
5. Extract relevant facts from the query
6. Formulate specific legal questions to answer

Output a structured JSON with these fields:
- original_query: The user's question as-is
- language: Primary language (ar/fr/en)
- legal_domain: Main legal domain
- key_entities: List of important entities
- intent: What the user wants to know
- facts: Relevant facts extracted from the query
- legal_questions: Specific legal questions derived from the query

Be precise and thorough in your analysis."""

        return prompt

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process user query and return structured understanding."""

        try:
            # Prepare prompt
            user_message = f"""Analyze this legal query:

Query: {agent_input.query}

Provide a structured understanding in JSON format following the schema defined in the system prompt.
Focus on Lebanese legal context (contract law, civil liability, criminal law)."""

            # Invoke LLM
            response = self.invoke_llm(user_message)

            # Parse JSON response
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            structured_query = json.loads(json_str)

            # Clean up list fields if they contain complex objects (convert to simple strings)
            for field in ['key_entities', 'facts', 'legal_questions']:
                if field in structured_query and isinstance(structured_query[field], list):
                    cleaned_list = []
                    for item in structured_query[field]:
                        if isinstance(item, dict):
                            # Extract value from dict (e.g., {'type': 'person', 'value': 'أحمد'} -> 'أحمد')
                            cleaned_list.append(item.get('value', str(item)))
                        else:
                            cleaned_list.append(str(item))
                    structured_query[field] = cleaned_list

            # Validate with Pydantic
            validated_query = StructuredQuery(**structured_query)

            logger.info(
                f"Query understood - Language: {validated_query.language}, "
                f"Domain: {validated_query.legal_domain}"
            )

            output = AgentOutput(
                result=validated_query.dict(),
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

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        else:
            return text
