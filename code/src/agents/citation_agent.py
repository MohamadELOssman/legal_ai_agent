"""
Agent 5: Citation Agent
Formats legal citations according to Lebanese standards
"""

import re
from typing import List, Dict
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput


class CitationAgent(BaseAgent):
    """
    Agent 5: Citation Agent

    Responsibility: Format legal citations according to Lebanese standards
    Input: Referenced legal sources
    Output: Properly formatted citations (e.g., "المادة 121 من قانون الموجبات والعقود")
    Technical Approach: Rule-based citation formatting with validation
    """

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.0):
        super().__init__(
            role=AgentRole.CITATION,
            model=model,
            temperature=temperature,  # Zero temperature for precise formatting
        )

        # Citation templates
        self.templates = {
            "ar": {
                "code_obligations": "المادة {article} من قانون الموجبات والعقود اللبناني",
                "penal_code": "المادة {article} من قانون العقوبات اللبناني",
                "criminal_procedure": "المادة {article} من قانون أصول المحاكمات الجزائية",
            },
            "fr": {
                "code_obligations": "Article {article} du Code des Obligations et des Contrats libanais",
                "penal_code": "Article {article} du Code pénal libanais",
                "criminal_procedure": "Article {article} du Code de procédure pénale",
            },
            "en": {
                "code_obligations": "Article {article} of the Lebanese Code of Obligations and Contracts",
                "penal_code": "Article {article} of the Lebanese Penal Code",
                "criminal_procedure": "Article {article} of the Lebanese Criminal Procedure Code",
            },
        }

    def get_system_prompt(self) -> str:
        return """You are a Citation Agent for Lebanese Legal Documents.

Your task is to format legal citations according to Lebanese legal citation standards.

Lebanese citation format:
- Arabic: "المادة [number] من [law name]"
- French: "Article [number] du/de [law name]"
- English: "Article [number] of the [law name]"

Common Lebanese laws:
- Code of Obligations and Contracts (قانون الموجبات والعقود)
- Penal Code (قانون العقوبات)
- Criminal Procedure Code (قانون أصول المحاكمات الجزائية)
- Court of Cassation decisions

Ensure citations are:
1. Properly formatted for the language
2. Complete with article numbers
3. Include the correct legal source
4. Consistent throughout the document

Validate that cited articles actually exist in the referenced sources."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Format legal citations."""

        try:
            # Get provisions and language
            provisions = agent_input.context.get("provisions", [])
            language = agent_input.context.get("structured_query", {}).get(
                "language", "ar"
            )

            # Format citations
            citations = self._format_citations(provisions, language)

            # Validate citations (basic validation)
            validated_citations = self._validate_citations(citations)

            logger.info(f"Formatted {len(validated_citations)} citations")

            output = AgentOutput(
                result={
                    "citations": validated_citations,
                    "formatted_count": len(validated_citations),
                },
                metadata={
                    "agent": self.role.value,
                    "language": language,
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Citation agent failed: {e}")
            return AgentOutput(
                result={"citations": []},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _format_citations(self, provisions: List[Dict], language: str) -> List[Dict]:
        """Format citations for provisions."""

        formatted_citations = []

        for prov in provisions:
            article = prov.get("article_number", "")
            source = prov.get("source_document", "")

            # Extract article number
            article_num = self._extract_article_number(article)

            if not article_num:
                continue

            # Determine document type
            doc_type = self._classify_document_type(source)

            # Format citation
            citation_text = self._format_single_citation(
                article_num, doc_type, language
            )

            formatted_citations.append(
                {
                    "article_number": article_num,
                    "citation_text": citation_text,
                    "document_type": doc_type,
                    "language": language,
                    "provision_text": prov.get("provision_text", ""),
                }
            )

        return formatted_citations

    def _extract_article_number(self, article_str: str) -> str:
        """Extract article number from string."""
        # Try to find numbers
        match = re.search(r"\d+", str(article_str))
        if match:
            return match.group(0)
        return ""

    def _classify_document_type(self, source: str) -> str:
        """Classify document type from source."""
        source_lower = source.lower()

        if "obligation" in source_lower or "contract" in source_lower:
            return "code_obligations"
        elif "penal" in source_lower or "عقوبات" in source:
            return "penal_code"
        elif "criminal" in source_lower or "procedure" in source_lower:
            return "criminal_procedure"
        else:
            return "code_obligations"  # Default

    def _format_single_citation(
        self, article_num: str, doc_type: str, language: str
    ) -> str:
        """Format a single citation."""

        # Get template
        if language not in self.templates:
            language = "ar"  # Default to Arabic

        if doc_type not in self.templates[language]:
            doc_type = "code_obligations"  # Default

        template = self.templates[language][doc_type]
        return template.format(article=article_num)

    def _validate_citations(self, citations: List[Dict]) -> List[Dict]:
        """Basic validation of citations."""

        validated = []

        for citation in citations:
            # Check that citation has required fields
            if citation.get("citation_text") and citation.get("article_number"):
                validated.append(citation)
            else:
                logger.warning(f"Invalid citation: {citation}")

        return validated
