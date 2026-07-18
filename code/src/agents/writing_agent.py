"""
Agent 6: Writing Agent
Two output formats driven by the Orchestrator:
  legal_explanation — educational memo explaining what the law says
  case_assessment   — advisory memo assessing a specific situation with precedent comparison
"""

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL
from src.utils.prompt_loader import load_agent_prompt


# Human-readable language names for the strict single-language instruction.
LANG_NAME = {
    "ar": "Arabic (العربية الفصحى)",
    "fr": "French (français)",
    "en": "English",
}

# Section headings per language — the memorandum is written entirely in the query
# language, so headings must match it (no bilingual "الموضوع / SUBJECT" mixing).
SECTIONS_EXPLANATION = {
    "ar": ["الموضوع", "الإطار القانوني", "الشروط والأحكام", "العقوبات والآثار", "الخلاصة"],
    "fr": ["Objet", "Cadre juridique", "Conditions et règles", "Peines et conséquences", "Conclusion"],
    "en": ["Subject", "Legal Framework", "Conditions & Rules", "Penalties & Consequences", "Conclusion"],
}
SECTIONS_ASSESSMENT = {
    "ar": ["الموضوع", "وقائع القضية", "النصوص القانونية المنطبقة",
           "مقارنة بالسوابق القضائية", "التقدير القانوني", "الخلاصة والتوصيات"],
    "fr": ["Objet", "Faits", "Dispositions applicables",
           "Comparaison avec la jurisprudence", "Appréciation juridique", "Conclusion et recommandations"],
    "en": ["Subject", "Facts", "Applicable Law",
           "Precedent Comparison", "Legal Assessment", "Conclusion & Recommendations"],
}

# Judicial decision (for a judge): facts given → the ruling is written.
SECTIONS_DECISION = {
    "ar": ["المحكمة والأطراف", "الوقائع", "النصوص القانونية المطبقة",
           "التعليل القانوني", "الحُكم / القرار"],
    "fr": ["Tribunal et parties", "Faits", "Textes applicables",
           "Motifs", "Décision"],
    "en": ["Court & Parties", "Facts of the Case", "Applicable Law",
           "Legal Reasoning", "Verdict (Decision)"],
}


def _language_directive(language: str) -> str:
    """A strict instruction to write the whole memo in one language only."""
    name = LANG_NAME.get(language, LANG_NAME["ar"])
    return (f"Write the ENTIRE memorandum — including every section heading — in {name} ONLY. "
            f"Do NOT produce bilingual text, and do NOT translate or duplicate headings in another language.")


def _numbered_sections(sections: list) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sections, 1))


class WritingAgent(BaseAgent):

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3,
                 max_tokens: int = 3000):
        # Cap output: a full structured memo fits comfortably in ~3k tokens; the
        # uncapped 4k budget made generation run ~240s and risked timeouts.
        super().__init__(role=AgentRole.WRITING, model=model, temperature=temperature,
                         max_tokens=max_tokens)

    def get_system_prompt(self) -> str:
        prompt = load_agent_prompt("writing")
        if prompt:
            return prompt
        return """You are a Legal Writing Agent specializing in Lebanese law memoranda.
You write in formal Arabic, French, or English depending on the query language.
Your memoranda are precise, well-structured, and legally grounded.
Only cite articles that are supplied to you; never invent legal references."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        try:
            orch         = agent_input.metadata.get("orchestrator", {})
            writing_cfg  = orch.get("writing", {})
            fmt          = writing_cfg.get("format", "legal_explanation")
            tone         = writing_cfg.get("tone", "educational")

            structured_query = agent_input.context.get("structured_query", {})
            provisions       = agent_input.context.get("provisions", [])
            reasoning        = agent_input.context.get("reasoning", "")
            citations        = agent_input.context.get("citations", [])
            similar_cases    = agent_input.context.get("similar_cases", [])
            case_assessment  = agent_input.context.get("case_assessment", {})
            extracted_facts  = orch.get("extracted_facts", [])

            language = structured_query.get("language", "ar")

            if fmt == "judicial_decision":
                memorandum = self._write_judicial_decision(
                    structured_query, provisions, reasoning, citations,
                    similar_cases, case_assessment, extracted_facts, language
                )
            elif fmt == "plain_answer":
                memorandum = self._write_plain_answer(
                    structured_query, provisions, reasoning, citations, language
                )
            elif fmt == "case_assessment":
                memorandum = self._write_case_assessment(
                    structured_query, provisions, reasoning, citations,
                    similar_cases, case_assessment, extracted_facts, language
                )
            else:
                memorandum = self._write_legal_explanation(
                    structured_query, provisions, reasoning, citations, language
                )

            logger.info(f"WritingAgent [{fmt}] — {len(memorandum)} chars")

            return AgentOutput(
                result={"memorandum": memorandum, "language": language, "format": fmt},
                metadata={"agent": self.role.value, "format": fmt,
                          "word_count": len(memorandum.split())},
                success=True,
            )

        except Exception as e:
            logger.error(f"WritingAgent failed: {e}")
            import traceback; logger.error(traceback.format_exc())
            return AgentOutput(
                result={"memorandum": "", "language": "ar", "format": "legal_explanation"},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    # ── format 1: legal explanation ────────────────────────────────────────────

    def _write_legal_explanation(
        self, structured_query, provisions, reasoning, citations, language
    ) -> str:

        lang = language if language in SECTIONS_EXPLANATION else "ar"
        sections = _numbered_sections(SECTIONS_EXPLANATION[lang])

        provisions_text = "\n\n".join(
            f"Article {p.get('article_number', '')}: {p.get('provision_text', '')[:300]}\n"
            f"Principle: {p.get('legal_principle', '')}\n"
            f"Conditions: {', '.join(p.get('conditions', []))}\n"
            f"Consequences: {p.get('penalties_or_consequences', '')}"
            for p in provisions[:6]
        )

        citations_text = "\n".join(
            f"- {c.get('citation_text', '')}" for c in citations[:8]
        )

        user_message = f"""Write a legal explanation memorandum.

{_language_directive(lang)}

QUERY: {structured_query.get('original_query', '')}
LEGAL DOMAIN: {structured_query.get('legal_domain', '')}

APPLICABLE PROVISIONS:
{provisions_text or 'No provisions retrieved.'}

LEGAL REASONING:
{reasoning or 'Not provided.'}

CITATIONS:
{citations_text or 'None.'}

Structure the memorandum with exactly these sections (use these headings verbatim, in order):
{sections}

Tone: educational and clear. Keep it focused and professional — avoid filler and repetition.
Cite every article you reference."""

        return self.invoke_llm(user_message)

    # ── format 2: case assessment ──────────────────────────────────────────────

    def _write_case_assessment(
        self, structured_query, provisions, reasoning, citations,
        similar_cases, case_assessment, extracted_facts, language
    ) -> str:

        lang = language if language in SECTIONS_ASSESSMENT else "ar"
        sections = _numbered_sections(SECTIONS_ASSESSMENT[lang])

        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "See original query."

        applicable = "\n\n".join(
            f"Article {p.get('article_number', '')}: {p.get('provision_text', '')[:250]}\n"
            f"Applies because: {p.get('application_reasoning', p.get('relevance', ''))}"
            for p in provisions[:5] if p.get("applies_to_case", True)
        )

        cases_text = "\n\n".join(
            f"Case {c.get('case_id', i+1)} | {c.get('court', '')} | {c.get('decision_date', '')}\n"
            f"Outcome: {c.get('outcome', '')} | Sentence: {c.get('sentence', '')}\n"
            f"Similarity: {c.get('similarity_reasoning', '')}"
            for i, c in enumerate(similar_cases[:3])
        ) if similar_cases else "No similar court rulings retrieved."

        assessment = case_assessment or {}
        citations_text = "\n".join(f"- {c.get('citation_text', '')}" for c in citations[:8])

        user_message = f"""Write a legal case assessment memorandum for a lawyer.

{_language_directive(lang)}

CASE FACTS:
{facts_text}

ORIGINAL QUERY: {structured_query.get('original_query', '')}

APPLICABLE LEGAL PROVISIONS:
{applicable or 'No directly applicable provisions identified.'}

SIMILAR COURT RULINGS:
{cases_text}

LEGAL REASONING:
{reasoning or 'Not provided.'}

PRELIMINARY ASSESSMENT:
- Strength: {assessment.get('strength_of_case', 'N/A')}
- Likely outcome: {assessment.get('likely_outcome', 'N/A')}
- Key issues: {', '.join(assessment.get('key_legal_issues', []))}
- Mitigating factors: {', '.join(assessment.get('mitigating_factors', []))}
- Aggravating factors: {', '.join(assessment.get('aggravating_factors', []))}

CITATIONS:
{citations_text or 'None.'}

Structure the memorandum with exactly these sections (use these headings verbatim, in order):
{sections}

Tone: advisory — written for a lawyer advising a client.
Keep it focused and professional — avoid filler and repetition.
Cite every article and case you reference."""

        return self.invoke_llm(user_message)

    # ── format 3: plain answer (citizen) ───────────────────────────────────────

    def _write_plain_answer(
        self, structured_query, provisions, reasoning, citations, language
    ) -> str:

        lang = language if language in LANG_NAME else "ar"
        provisions_text = "\n".join(
            f"- Article {p.get('article_number', '')}: {p.get('provision_text', '')[:200]}"
            for p in provisions[:5]
        )
        citations_text = ", ".join(c.get('citation_text', '') for c in citations[:5])

        user_message = f"""Answer this person's legal question in plain, everyday language.

{_language_directive(lang)}

QUESTION: {structured_query.get('original_query', '')}

RELEVANT LAW:
{provisions_text or 'No specific articles retrieved.'}

Write a clear, direct answer that an ordinary person (not a lawyer) can understand:
- Start with a direct answer to their question.
- Explain briefly and simply — avoid legal jargon; if you use a legal term, explain it.
- Keep it short (a few short paragraphs), warm and helpful.
- Mention the relevant article number(s) plainly at the end so they can verify: {citations_text or 'N/A'}.
- Do NOT use formal memorandum headings or numbered sections.
- If appropriate, gently note that this is general information, not a substitute for a lawyer.

Only rely on the law provided above; never invent article numbers."""

        return self.invoke_llm(user_message)

    # ── format 4: judicial decision (judge) ────────────────────────────────────

    def _write_judicial_decision(
        self, structured_query, provisions, reasoning, citations,
        similar_cases, case_assessment, extracted_facts, language
    ) -> str:

        lang = language if language in SECTIONS_DECISION else "ar"
        sections = _numbered_sections(SECTIONS_DECISION[lang])

        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "See the submitted case."
        applicable = "\n\n".join(
            f"Article {p.get('article_number', '')}: {p.get('provision_text', '')[:250]}\n"
            f"Relevance: {p.get('application_reasoning', p.get('relevance', ''))}"
            for p in provisions[:6]
        )
        cases_text = "\n\n".join(
            f"Case {c.get('case_id', i+1)} | {c.get('court', '')} | Outcome: {c.get('outcome', '')} "
            f"| Sentence: {c.get('sentence', '')}"
            for i, c in enumerate(similar_cases[:3])
        ) if similar_cases else "No precedent rulings retrieved."
        assessment = case_assessment or {}
        citations_text = "\n".join(f"- {c.get('citation_text', '')}" for c in citations[:8])

        user_message = f"""You are drafting a court DECISION for a judge who has submitted the case
and expects the ruling to be written. Reach and state a clear decision (verdict and,
where applicable, the sentence), reasoned from the facts, the applicable law, and precedent.

{_language_directive(lang)}

FACTS OF THE CASE:
{facts_text}

SUBMITTED QUERY: {structured_query.get('original_query', '')}

APPLICABLE LEGAL PROVISIONS:
{applicable or 'No directly applicable provisions identified.'}

PRECEDENT RULINGS:
{cases_text}

LEGAL REASONING (supporting analysis):
{reasoning or 'Not provided.'}

ASSESSMENT (guidance): likely outcome — {assessment.get('likely_outcome', 'N/A')};
aggravating — {', '.join(assessment.get('aggravating_factors', []))};
mitigating — {', '.join(assessment.get('mitigating_factors', []))}.

CITATIONS:
{citations_text or 'None.'}

Write the decision as a formal court ruling with exactly these sections
(use these headings verbatim, in order):
{sections}

Requirements:
- The final section MUST state an explicit decision (e.g., conviction/acquittal) and,
  where applicable, the sentence, grounded in the cited articles.
- Formal, impartial judicial tone. Cite every article you rely on.
- Only rely on the law and facts provided; never invent article numbers.
- Note that this is an AI-drafted decision for the judge's review, not a binding judgment."""

        return self.invoke_llm(user_message)
