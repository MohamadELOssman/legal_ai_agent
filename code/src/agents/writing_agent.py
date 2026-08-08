"""
Agent 6: Writing Agent
Four output formats driven by the Orchestrator, each with a FIXED set of section
headers (see SECTIONS_* below) that the answer must follow verbatim and in order:
  plain_answer      — citizen: التحليل القانوني · الإجابة باختصار
  legal_explanation — case study: الوقائع · الاشكالية · المواد · التطبيق · الحل · المحكمة
  case_assessment   — lawyer defence: الوقائع · القوانين · طريقة الدفاع · الإجابة باختصار
  judicial_decision — judge: المحكمة · الأطراف · الوقائع · القوانين · التطبيق · الحكم
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

# Fixed section headings per format and language. These are MANDATORY templates:
# the answer must use exactly these headers, verbatim and in order. The Arabic
# headers are authoritative (given by the domain expert); FR/EN mirror them so the
# trilingual system keeps an identical structure.

# Case Study (general case analysis) — mapped to the "legal_explanation" format.
SECTIONS_EXPLANATION = {
    "ar": ["الوقائع المنتجة", "الاشكالية القانونية", "المواد والقوانين ذات الصلة",
           "تطبيق القانون على الوقائع", "الحل", "المحكمة المختصة (في حال وجودها)"],
    "fr": ["Faits pertinents", "Problématique juridique", "Textes et lois applicables",
           "Application du droit aux faits", "Solution", "Tribunal compétent (le cas échéant)"],
    "en": ["Relevant Facts", "Legal Issue", "Applicable Laws & Articles",
           "Application of the Law to the Facts", "Solution", "Competent Court (if applicable)"],
}

# Lawyer (defence) — mapped to the "case_assessment" format.
SECTIONS_ASSESSMENT = {
    "ar": ["الوقائع المنتجة", "القوانين والمواد ذات الصلة", "طريقة الدفاع", "الإجابة باختصار"],
    "fr": ["Faits pertinents", "Lois et articles applicables", "Stratégie de défense", "Réponse en bref"],
    "en": ["Relevant Facts", "Applicable Laws & Articles", "Defense Strategy", "Answer in Brief"],
}

# Judge — facts given → the ruling is written. Mapped to "judicial_decision".
SECTIONS_DECISION = {
    "ar": ["المحكمة المختصة", "أطراف الدعوى", "الوقائع المنتجة", "القوانين والمواد ذات الصلة",
           "تطبيق القانون على الوقائع", "الحكم"],
    "fr": ["Tribunal compétent", "Parties au litige", "Faits pertinents", "Lois et articles applicables",
           "Application du droit aux faits", "Jugement"],
    "en": ["Competent Court", "Parties to the Case", "Relevant Facts", "Applicable Laws & Articles",
           "Application of the Law to the Facts", "Judgment"],
}

# Citizen — plain, two-part answer. Mapped to the "plain_answer" format.
SECTIONS_CITIZEN = {
    "ar": ["التحليل القانوني", "الإجابة باختصار"],
    "fr": ["Analyse juridique", "Réponse en bref"],
    "en": ["Legal Analysis", "Answer in Brief"],
}


def _language_directive(language: str) -> str:
    """A strict instruction to write the whole memo in one language only, concisely."""
    name = LANG_NAME.get(language, LANG_NAME["ar"])
    return (f"Write the ENTIRE answer — including every section heading — in {name} ONLY. "
            f"Do NOT produce bilingual text, and do NOT translate or duplicate headings in another language.\n"
            f"BE PRECISE AND TO THE POINT: answer directly, state only what is legally relevant, and avoid "
            f"filler, preamble, and repetition. Finish the answer completely — do not stop mid-sentence.")


def _headers_directive(sections: list) -> str:
    """Force the exact section headers, in order, each a Markdown H3 heading.

    Using an ATX heading ('### ') rather than bold guarantees the header sits on
    its own line (a block element); bold text on its own line collapses back onto
    the following paragraph when rendered as Markdown.
    """
    headers = "\n".join(f"### {s}" for s in sections)
    return ("Use EXACTLY the following section headers, in this exact order. Write each one as a "
            "Markdown level-3 heading — the line must start with '### ' exactly as shown — followed "
            "by its content on the next lines. Do NOT add, remove, rename, reorder, number, or "
            "translate any header, and do NOT use '#' or '##'. Headers:\n\n" + headers)


class WritingAgent(BaseAgent):

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3,
                 max_tokens: int = 4096):
        # Headroom so complete answers are never truncated. Cost scales with the
        # ACTUAL output length, not this cap — the concise prompts keep it short.
        super().__init__(role=AgentRole.WRITING, model=model, temperature=temperature,
                         max_tokens=max_tokens)

    def get_system_prompt(self) -> str:
        prompt = load_agent_prompt("writing")
        if prompt:
            return prompt
        return """# CONTEXT
You write the final answer of a Lebanese criminal-law system; everything you cite comes from
provisions supplied to you.
# ROLE
You are the Writing Agent: you turn the analysis and citations into a clear, correctly-formatted
answer for a citizen, lawyer, or judge.
# ACTION
Write for the given reader, grounded strictly in the supplied provisions; cite article numbers
and name their code; never invent a reference or add an unrequested section.
# FORMAT
Use EXACTLY the section headers given in the task, verbatim and in order, each as a Markdown
level-3 heading ("### "); do not renumber, rename, translate, reorder, add, or drop them. Write
the ENTIRE answer in the user's language.
# TARGET
Plain for a citizen, strategic for a lawyer, formal for a judge. Precise and complete — no
filler. This is legal information, not a substitute for a licensed lawyer."""

    def _citation_constraint(self, citations) -> str:
        """Restrict the memo to the verified article set (precision over recall).

        Passing the allowed article numbers and forbidding others keeps the final
        memorandum from citing loosely-related neighbours it wasn't given.
        """
        allowed = sorted(
            {str(c.get("article_number", "")) for c in citations
             if c.get("article_number") and c.get("verified", True)},
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        if not allowed:
            return ""
        return ("\n\nCITATION CONSTRAINT: You may cite ONLY these article numbers: "
                + ", ".join(allowed) +
                ". Do NOT mention or cite any other article number, even if related.")

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
                    structured_query, provisions, reasoning, citations, extracted_facts, language
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
        self, structured_query, provisions, reasoning, citations, extracted_facts, language
    ) -> str:
        """Case-study format: neutral, objective analysis of a scenario/question."""

        lang = language if language in SECTIONS_EXPLANATION else "ar"

        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "See the query."
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

        user_message = f"""Write an objective legal case study.

{_language_directive(lang)}

QUERY / SCENARIO: {structured_query.get('original_query', '')}
LEGAL DOMAIN: {structured_query.get('legal_domain', '')}

RELEVANT FACTS:
{facts_text}

APPLICABLE PROVISIONS:
{provisions_text or 'No provisions retrieved.'}

LEGAL REASONING:
{reasoning or 'Not provided.'}

CITATIONS:
{citations_text or 'None.'}

{_headers_directive(SECTIONS_EXPLANATION[lang])}

Guidance per section:
- Relevant Facts: state the material facts of the scenario (if the query is a general
  question with no facts, summarise the situation it describes).
- Legal Issue: the precise legal question to resolve.
- Applicable Laws & Articles: the exact articles and what each provides.
- Application of the Law to the Facts: apply the articles to the facts and reason to a result.
- Solution: the concrete legal outcome.
- Competent Court: name it only if it can be determined; otherwise say it depends / is not applicable.

Tone: neutral and analytical (a case study, not advocacy). Cite every article you reference."""

        return self.invoke_llm(user_message + self._citation_constraint(citations))

    # ── format 2: case assessment ──────────────────────────────────────────────

    def _write_case_assessment(
        self, structured_query, provisions, reasoning, citations,
        similar_cases, case_assessment, extracted_facts, language
    ) -> str:

        lang = language if language in SECTIONS_ASSESSMENT else "ar"

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

        user_message = f"""Write a defence-oriented legal analysis for a LAWYER defending a client.

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

{_headers_directive(SECTIONS_ASSESSMENT[lang])}

Guidance per section:
- Relevant Facts: the facts that matter for the defence, as concise bullet points.
- Applicable Laws & Articles: each relevant article and what it provides (incl. any
  defence/mitigation articles).
- Defense Strategy: the actual defence plea — argue for the client, addressed to the court,
  presenting the primary argument and any alternative/subsidiary arguments.
- Answer in Brief: a 2-3 sentence summary of the defence.

Tone: advocacy — you are arguing FOR the client. Cite every article and case you reference."""

        return self.invoke_llm(user_message + self._citation_constraint(citations))

    # ── format 3: plain answer (citizen) ───────────────────────────────────────

    def _write_plain_answer(
        self, structured_query, provisions, reasoning, citations, language
    ) -> str:

        lang = language if language in SECTIONS_CITIZEN else "ar"
        provisions_text = "\n".join(
            f"- Article {p.get('article_number', '')}: {p.get('provision_text', '')[:200]}"
            for p in provisions[:5]
        )
        citations_text = ", ".join(c.get('citation_text', '') for c in citations[:5])

        user_message = f"""Answer this ordinary citizen's legal question.

{_language_directive(lang)}

QUESTION: {structured_query.get('original_query', '')}

RELEVANT LAW:
{provisions_text or 'No specific articles retrieved.'}

CITATIONS: {citations_text or 'N/A'}

{_headers_directive(SECTIONS_CITIZEN[lang])}

Guidance per section:
- Legal Analysis: briefly explain what the relevant article(s) say and how they apply,
  quoting the article numbers (e.g., "Article 636 provides that ..."). Keep it clear and
  accessible for a non-lawyer; explain any legal term you use.
- Answer in Brief: one or two sentences giving the direct, plain answer to the question.

Only rely on the law provided above; never invent article numbers."""

        return self.invoke_llm(user_message + self._citation_constraint(citations))

    # ── format 4: judicial decision (judge) ────────────────────────────────────

    def _write_judicial_decision(
        self, structured_query, provisions, reasoning, citations,
        similar_cases, case_assessment, extracted_facts, language
    ) -> str:

        lang = language if language in SECTIONS_DECISION else "ar"

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

{_headers_directive(SECTIONS_DECISION[lang])}

Guidance per section:
- Competent Court: the court with jurisdiction and why.
- Parties to the Case: the parties (prosecution, accused, civil parties, victim) as applicable.
- Relevant Facts: the established material facts.
- Applicable Laws & Articles: each article relied on and what it provides.
- Application of the Law to the Facts: reason from the facts and articles toward the ruling
  (use "Whereas ..." style considerations where natural).
- Judgment: an explicit ruling (e.g., conviction/acquittal) and, where applicable, the sentence,
  grounded in the cited articles.

Formal, impartial judicial tone. Cite every article you rely on. Only rely on the law and
facts provided; never invent article numbers."""

        return self.invoke_llm(user_message + self._citation_constraint(citations))
