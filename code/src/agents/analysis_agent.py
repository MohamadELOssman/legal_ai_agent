"""
Agent 3: Analysis Agent
Reads and extracts key provisions from retrieved legal texts
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput


class LegalProvision(BaseModel):
    """Extracted legal provision."""

    article_number: str
    provision_text: str
    legal_principle: str
    relevance: str
    source_document: str


class AnalysisAgent(BaseAgent):
    """
    Agent 3: Analysis Agent

    Responsibility: Read and extract key provisions from retrieved legal texts
    Input: Retrieved legal articles and court decisions
    Output: Extracted relevant provisions, definitions, and legal principles
    Technical Approach: LLM-based extraction with structured output formatting
    """

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.1):
        super().__init__(
            role=AgentRole.ANALYSIS,
            model=model,
            temperature=temperature,
        )

    def get_system_prompt(self) -> str:
        return """You are an Analysis Agent for Lebanese Legal Research.

Your task is to analyze retrieved legal documents and extract key provisions relevant to the legal question.

For each relevant document, extract:
1. Article number (if available)
2. The specific provision text
3. The legal principle it establishes
4. Why it's relevant to the query
5. Source document

Focus on:
- Provisions directly applicable to the facts
- Legal definitions
- Conditions and requirements
- Exceptions and limitations
- Remedies and consequences

Provide structured analysis in JSON format."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Extract key provisions from retrieved documents."""

        try:
            # Get retrieved documents from context
            documents = agent_input.context.get("retrieved_documents", [])
            structured_query = agent_input.context.get("structured_query", {})

            if not documents:
                logger.warning("No documents to analyze")
                return AgentOutput(
                    result={"provisions": []},
                    metadata={"agent": self.role.value},
                    success=True,
                )

            # Extract provisions
            provisions = self._extract_provisions(documents, structured_query)

            logger.info(f"Extracted {len(provisions)} legal provisions")

            output = AgentOutput(
                result={
                    "provisions": provisions,
                    "total_analyzed": len(documents),
                },
                metadata={
                    "agent": self.role.value,
                    "documents_analyzed": len(documents),
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Analysis agent failed: {e}")
            return AgentOutput(
                result={"provisions": []},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _extract_provisions(
        self, documents: List[Dict], structured_query: Dict
    ) -> List[Dict]:
        """Extract legal provisions from documents."""

        # Prepare document summaries
        docs_text = "\n\n---\n\n".join(
            [
                f"Document {i+1}:\nArticle: {doc.get('metadata', {}).get('article_number', 'N/A')}\nContent: {doc.get('content', '')[:1000]}"
                for i, doc in enumerate(documents[:5])  # Analyze top 5 documents
            ]
        )

        # Prepare user message
        user_message = f"""Legal Query Context:
- Domain: {structured_query.get('legal_domain', 'N/A')}
- Key Entities: {', '.join(structured_query.get('key_entities', []))}
- Legal Questions: {', '.join(structured_query.get('legal_questions', []))}

Retrieved Documents:
{docs_text}

Extract all relevant legal provisions from these documents. For each provision, provide:
- article_number: The article number (e.g., "Article 121")
- provision_text: The exact text of the provision (in original language)
- legal_principle: What legal principle this establishes
- relevance: Why this is relevant to the query
- source_document: Source document identifier

Return as a JSON array of provisions."""

        # Invoke LLM
        response = self.invoke_llm(user_message)

        # Parse response
        try:
            json_str = self._extract_json(response)
            provisions_data = json.loads(json_str)

            # Handle both dict with "provisions" key and direct list
            if isinstance(provisions_data, dict):
                provisions_list = provisions_data.get("provisions", [])
            else:
                provisions_list = provisions_data

            # Validate provisions
            validated_provisions = []
            for prov in provisions_list:
                try:
                    validated = LegalProvision(**prov)
                    validated_provisions.append(validated.dict())
                except Exception as e:
                    logger.warning(f"Invalid provision: {e}")

            return validated_provisions

        except Exception as e:
            logger.error(f"Failed to parse provisions: {e}")
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "[" in text or "{" in text:
            # Find first [ or {
            start = min(
                text.find("[") if "[" in text else len(text),
                text.find("{") if "{" in text else len(text),
            )
            # Find last ] or }
            end = max(text.rfind("]"), text.rfind("}")) + 1
            return text[start:end]
        return text
