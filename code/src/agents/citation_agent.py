"""
Agent 5: Citation Agent
Formats legal citations according to Lebanese standards and validates them
against the known corpus to prevent hallucinated article numbers.
"""

import re
from typing import List, Dict
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL

try:
    from src.utils.citation_validator import CitationValidator
    _VALIDATOR_AVAILABLE = True
except Exception:  # pragma: no cover - validator is optional
    _VALIDATOR_AVAILABLE = False


class CitationAgent(BaseAgent):
    """
    Agent 5: Citation Agent

    Responsibility: Format legal citations according to Lebanese standards
    Input: Extracted provisions (from the Analysis agent) + query language
    Output: Properly formatted, corpus-validated citations
            (e.g., "المادة 549 من قانون العقوبات اللبناني")
    Technical Approach: Rule-based formatting + validation against the article index
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0):
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

        # Map our document-type keys to the validator's index keys (the keys in
        # articles_index.json, which are the raw document_type values).
        self._validator_doctype = {
            "penal_code": "penal_code",
            "code_obligations": "code_obligations_contracts",
            "criminal_procedure": "criminal_procedure_code",
        }

        # Lazy-load the corpus validator (article index). Optional — if the index
        # is missing, citations are still produced but marked as unverified.
        self.validator = None
        if _VALIDATOR_AVAILABLE:
            try:
                self.validator = CitationValidator()
            except Exception as e:
                logger.warning(f"CitationValidator unavailable: {e}")

    def get_system_prompt(self) -> str:
        return """# CONTEXT
You format citations for a Lebanese criminal-law system covering the Penal Code and the Code of
Criminal Procedure, whose article numbers OVERLAP — a citation is meaningful only with its code.
# ROLE
You are the Citation Agent: you produce correctly-formatted, verifiable citations.
# ACTION
Format each citation in the answer's language and name the correct code. NEVER cite an article
that is not present in the provided provisions, and never invent a number.
# FORMAT
Lebanese style — ar: "المادة [رقم] من [اسم القانون]"; fr: "Article [n] du/de [loi]";
en: "Article [n] of the [law]". Be consistent.
# TARGET
Lawyers and judges rely on these — correctness and verifiability outweigh coverage."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Format and validate legal citations."""

        try:
            structured_query = agent_input.context.get("structured_query", {}) or {}
            language = structured_query.get("language", "ar")

            # Provisions live in the analysis output in the pipeline; fall back to a
            # directly-supplied `provisions` key (used by the individual-agent tester).
            analysis_results = agent_input.context.get("analysis_results", {}) or {}
            provisions = (
                agent_input.context.get("provisions")
                or analysis_results.get("provisions", [])
            )

            # Default legal source for the active corpus is the Penal Code.
            default_doc_type = self._default_doc_type(structured_query)

            # Cite only DIRECTLY-APPLICABLE, grounded provisions (precision > recall):
            # the earlier benchmark showed over-citation of loosely-related neighbours.
            applicable = self._select_applicable(provisions)
            citations = self._format_citations(applicable, language, default_doc_type)
            validated_citations, report = self._validate_citations(citations)
            # Keep the tight top set so the memorandum does not over-cite.
            validated_citations = validated_citations[:self.MAX_CITATIONS]
            report["kept"] = len(validated_citations)

            logger.info(
                f"CitationAgent — {len(validated_citations)} citations "
                f"({report['verified']} verified, {report['unverified']} unverified; "
                f"{len(provisions)}→{len(applicable)} provisions applicable)"
            )

            output = AgentOutput(
                result={
                    "citations": validated_citations,
                    "formatted_count": len(validated_citations),
                    "validation_report": report,
                },
                metadata={"agent": self.role.value, "language": language},
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

    # Cap the number of citations so the final memorandum stays precise.
    MAX_CITATIONS = 6

    # ── selection ────────────────────────────────────────────────────────────────

    def _select_applicable(self, provisions: List[Dict]) -> List[Dict]:
        """Keep only directly-applicable, corpus-grounded provisions.

        Prefers grounded provisions; among those, prefers ones the analysis marked
        as applying to the case (`applies_to_case`). Falls back gracefully so an
        answer is never left with zero citations when provisions exist.
        """
        grounded = [p for p in provisions if p.get("grounded", True)]
        pool = grounded or provisions
        applicable = [p for p in pool if p.get("applies_to_case", True)]
        return applicable or pool

    # ── formatting ───────────────────────────────────────────────────────────────

    def _default_doc_type(self, structured_query: Dict) -> str:
        """Pick the default legal source. The loaded corpus is the Penal Code, so
        criminal queries (and the default) map to penal_code rather than the old
        obligations-code default."""
        domain = (structured_query.get("legal_domain") or "").lower()
        if "contract" in domain or "obligation" in domain or "civil" in domain:
            return "code_obligations"
        if "procedure" in domain:
            return "criminal_procedure"
        return "penal_code"

    def _format_citations(
        self, provisions: List[Dict], language: str, default_doc_type: str
    ) -> List[Dict]:
        """Format citations for provisions, de-duplicating by (article, doc_type)."""

        formatted_citations = []
        seen = set()

        for prov in provisions:
            article_num = self._extract_article_number(prov.get("article_number", ""))
            if not article_num:
                continue

            doc_type = self._classify_document_type(prov, default_doc_type)

            key = (article_num, doc_type)
            if key in seen:
                continue
            seen.add(key)

            citation_text = self._format_single_citation(article_num, doc_type, language)

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
        match = re.search(r"\d+", str(article_str))
        return match.group(0) if match else ""

    def _classify_document_type(self, prov: Dict, default_doc_type: str) -> str:
        """Classify the legal source for a provision, preferring explicit metadata."""
        # Explicit metadata from the retrieved document, if the analysis kept it.
        meta_type = str(prov.get("document_type", "")).lower()
        if "penal" in meta_type or "عقوبات" in meta_type:
            return "penal_code"
        if "obligation" in meta_type or "contract" in meta_type:
            return "code_obligations"
        if "procedure" in meta_type or "محاكمات" in meta_type:
            return "criminal_procedure"

        source = str(prov.get("source_document", "")).lower()
        if "penal" in source or "عقوبات" in source:
            return "penal_code"
        if "obligation" in source or "contract" in source:
            return "code_obligations"
        if "procedure" in source or "محاكمات" in source:
            return "criminal_procedure"

        return default_doc_type

    def _format_single_citation(self, article_num: str, doc_type: str, language: str) -> str:
        if language not in self.templates:
            language = "ar"
        if doc_type not in self.templates[language]:
            doc_type = "penal_code"
        return self.templates[language][doc_type].format(article=article_num)

    # ── validation ───────────────────────────────────────────────────────────────

    def _validate_citations(self, citations: List[Dict]):
        """Mark each citation as verified against the corpus article index.

        Citations are never dropped — unverifiable ones are flagged so downstream
        agents and the UI can surface potential hallucinations.
        """
        verified = 0
        unverified = 0
        flagged = []

        for citation in citations:
            if not (citation.get("citation_text") and citation.get("article_number")):
                citation["verified"] = False
                unverified += 1
                continue

            is_valid = True
            if self.validator is not None:
                index_key = self._validator_doctype.get(
                    citation["document_type"], citation["document_type"]
                )
                is_valid = self.validator.validate_citation(
                    citation["article_number"], index_key
                )

            citation["verified"] = bool(is_valid)
            if is_valid:
                verified += 1
            else:
                unverified += 1
                flagged.append(citation["citation_text"])
                logger.warning(
                    f"Unverified citation (not in corpus): {citation['citation_text']}"
                )

        report = {
            "total": len(citations),
            "verified": verified,
            "unverified": unverified,
            "validator_available": self.validator is not None,
            "flagged": flagged,
        }
        return citations, report
