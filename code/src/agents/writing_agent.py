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


class WritingAgent(BaseAgent):

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3):
        super().__init__(role=AgentRole.WRITING, model=model, temperature=temperature)

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

            if fmt == "case_assessment":
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

        lang_rule = {
            "ar": "Write entirely in formal classical Arabic (الفصحى). Use proper Lebanese legal terminology.",
            "fr": "Write entirely in formal French legal style.",
            "en": "Write entirely in professional legal English.",
        }.get(language, "Write in formal Arabic.")

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

QUERY: {structured_query.get('original_query', '')}
LEGAL DOMAIN: {structured_query.get('legal_domain', '')}
LANGUAGE RULE: {lang_rule}

APPLICABLE PROVISIONS:
{provisions_text or 'No provisions retrieved.'}

LEGAL REASONING:
{reasoning or 'Not provided.'}

CITATIONS:
{citations_text or 'None.'}

Write a structured memorandum with these sections:
1. الموضوع / SUBJECT — one line stating the topic
2. الإطار القانوني / LEGAL FRAMEWORK — list the applicable articles and what they establish
3. الشروط والأحكام / CONDITIONS & RULES — explain the conditions, requirements, and limitations clearly
4. العقوبات والآثار / PENALTIES & CONSEQUENCES — what the law prescribes
5. الخلاصة / CONCLUSION — direct answer to the question

Tone: {structured_query.get('legal_domain', 'educational')} — educational and clear.
Cite every article you reference."""

        return self.invoke_llm(user_message)

    # ── format 2: case assessment ──────────────────────────────────────────────

    def _write_case_assessment(
        self, structured_query, provisions, reasoning, citations,
        similar_cases, case_assessment, extracted_facts, language
    ) -> str:

        lang_rule = {
            "ar": "Write entirely in formal classical Arabic (الفصحى). Use proper Lebanese legal terminology.",
            "fr": "Write entirely in formal French legal style.",
            "en": "Write entirely in professional legal English.",
        }.get(language, "Write in formal Arabic.")

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

CASE FACTS:
{facts_text}

ORIGINAL QUERY: {structured_query.get('original_query', '')}
LANGUAGE RULE: {lang_rule}

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

Write a structured case assessment memorandum with these sections:
1. الموضوع / SUBJECT — one line describing the case situation
2. وقائع القضية / FACTS — concise summary of the relevant facts
3. النصوص القانونية المنطبقة / APPLICABLE LAW — each article with explanation of why it applies
4. مقارنة بالسوابق القضائية / PRECEDENT COMPARISON — compare with similar rulings, note key similarities/differences
5. التقدير القانوني / LEGAL ASSESSMENT — strength of case, likely outcome based on law and precedent
6. الخلاصة والتوصيات / CONCLUSION & RECOMMENDATIONS — concrete advice for the client

Tone: advisory — written for a lawyer advising a client.
Cite every article and case you reference."""

        return self.invoke_llm(user_message)
