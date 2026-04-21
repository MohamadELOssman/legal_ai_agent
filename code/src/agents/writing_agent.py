"""
Agent 6: Writing Agent
Generates professional legal memorandum
"""

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput


class WritingAgent(BaseAgent):
    """
    Agent 6: Writing Agent

    Responsibility: Generate professional legal memorandum
    Input: Analysis, reasoning, and citations from other agents
    Output: Complete legal opinion in user's preferred language
    Technical Approach: LLM-based text generation with legal writing templates
    """

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.3):
        super().__init__(
            role=AgentRole.WRITING,
            model=model,
            temperature=temperature,  # Higher temperature for natural writing
        )

    def get_system_prompt(self) -> str:
        return """You are a Writing Agent for Lebanese Legal Memoranda.

Your task is to generate professional legal memoranda in Arabic, French, or English.

Structure of legal memorandum:
1. **Introduction**
   - Brief summary of the matter
   - Key legal questions

2. **Facts**
   - Relevant facts presented clearly

3. **Legal Analysis**
   - Applicable legal provisions (with citations)
   - Application of law to facts
   - Legal reasoning
   - Discussion of relevant precedents

4. **Conclusion**
   - Clear answer to legal questions
   - Legal consequences and recommendations

Writing style:
- Professional and formal tone
- Clear and precise language
- Structured with headings and subheadings
- Proper use of legal terminology
- Citations in Lebanese format
- Logical flow of arguments

Lebanese legal writing conventions:
- Arabic: Formal classical Arabic (not colloquial)
- French: Formal French legal style
- English: Professional legal English

Ensure the memorandum is:
- Comprehensive yet concise
- Well-organized and easy to follow
- Properly cited
- Actionable for the client"""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Generate legal memorandum."""

        try:
            # Get context from previous agents
            structured_query = agent_input.context.get("structured_query", {})
            provisions = agent_input.context.get("provisions", [])
            reasoning = agent_input.context.get("reasoning", "")
            citations = agent_input.context.get("citations", [])

            # Get language preference
            language = structured_query.get("language", "ar")

            # Generate memorandum
            memorandum = self._generate_memorandum(
                structured_query, provisions, reasoning, citations, language
            )

            logger.info(f"Generated legal memorandum ({len(memorandum)} characters)")

            output = AgentOutput(
                result={"memorandum": memorandum, "language": language},
                metadata={
                    "agent": self.role.value,
                    "language": language,
                    "word_count": len(memorandum.split()),
                },
                success=True,
            )

            # Skip logging to avoid serialization issues
            # self.log_input_output(agent_input, output)
            logger.info(f"[{self.role.value}] Memorandum generated successfully")

            return output

        except Exception as e:
            logger.error(f"Writing agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return AgentOutput(
                result={"memorandum": ""},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _generate_memorandum(
        self,
        structured_query: dict,
        provisions: list,
        reasoning: str,
        citations: list,
        language: str,
    ) -> str:
        """Generate complete legal memorandum."""

        # Format citations
        citations_text = "\n".join(
            [
                f"- {cit.get('citation_text', '')}"
                for cit in citations[:10]  # Top 10 citations
            ]
        )

        # Format provisions
        provisions_text = "\n\n".join(
            [
                f"{i+1}. {prov.get('article_number', '')}: {prov.get('provision_text', '')[:200]}..."
                for i, prov in enumerate(provisions[:5])
            ]
        )

        # Language-specific instructions
        lang_instructions = {
            "ar": "Write in formal classical Arabic (الفصحى). Use proper legal terminology.",
            "fr": "Write in formal French legal style. Use proper legal terminology.",
            "en": "Write in professional legal English. Use clear and precise language.",
        }

        lang_instruction = lang_instructions.get(language, lang_instructions["ar"])

        # Build prompt
        user_message = f"""Generate a complete legal memorandum based on the following:

QUERY INFORMATION:
- Original Query: {structured_query.get('original_query', '')}
- Legal Domain: {structured_query.get('legal_domain', '')}
- Legal Questions: {', '.join(structured_query.get('legal_questions', []))}
- Facts: {', '.join(structured_query.get('facts', []))}

APPLICABLE LEGAL PROVISIONS:
{provisions_text}

LEGAL REASONING:
{reasoning}

CITATIONS:
{citations_text}

---

INSTRUCTIONS:
{lang_instruction}

Generate a professional legal memorandum with the following structure:

1. **مذكرة قانونية / NOTE JURIDIQUE / LEGAL MEMORANDUM**

2. **الموضوع / OBJET / SUBJECT**
   - Brief description of the matter

3. **الوقائع / FAITS / FACTS**
   - Summary of relevant facts

4. **التحليل القانوني / ANALYSE JURIDIQUE / LEGAL ANALYSIS**
   - Applicable legal provisions with citations
   - Application of law to facts
   - Legal reasoning and conclusions

5. **الخلاصة والتوصيات / CONCLUSION ET RECOMMANDATIONS / CONCLUSION**
   - Clear answers to legal questions
   - Legal consequences
   - Recommendations

Make it professional, well-structured, and actionable."""

        # Generate memorandum
        memorandum = self.invoke_llm(user_message)

        return memorandum
