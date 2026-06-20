"""
Agent 3: Analysis Agent
Two modes driven by the Orchestrator:
  legal_explanation — extract and explain provisions for a general legal query
  case_assessment   — apply provisions to specific facts and compare with similar rulings
"""

import json
import re
from typing import List, Dict, Any

from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL
from src.utils.prompt_loader import load_agent_prompt


class AnalysisAgent(BaseAgent):

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        super().__init__(role=AgentRole.ANALYSIS, model=model, temperature=temperature)

    def get_system_prompt(self) -> str:
        prompt = load_agent_prompt("analysis")
        if prompt:
            return prompt
        return """You are a Legal Analysis Agent specializing in Lebanese law.
You receive retrieved legal documents and analyze them precisely.
Always return structured JSON. Never invent articles or invent facts."""

    # ── entry point ────────────────────────────────────────────────────────────

    def process(self, agent_input: AgentInput) -> AgentOutput:
        try:
            orch          = agent_input.metadata.get("orchestrator", {})
            analysis_cfg  = orch.get("analysis", {})
            mode          = analysis_cfg.get("mode", "legal_explanation")
            instructions  = analysis_cfg.get("instructions", "")

            research_results   = agent_input.context.get("research_results", {})
            documents          = research_results.get("retrieved_documents", [])
            structured_query   = agent_input.context.get("structured_query", {})
            extracted_facts    = orch.get("extracted_facts", [])

            if not documents:
                logger.warning("AnalysisAgent: no documents to analyze")
                return AgentOutput(
                    result={"provisions": [], "mode": mode},
                    metadata={"agent": self.role.value, "mode": mode},
                    success=True,
                )

            if mode == "case_assessment":
                result = self._case_assessment(
                    documents, structured_query, extracted_facts, instructions
                )
            else:
                result = self._legal_explanation(
                    documents, structured_query, instructions
                )

            result["mode"] = mode

            # Ground extracted provisions against the retrieved documents so that
            # downstream agents (reasoning, citation, writing) never build on
            # hallucinated article numbers.
            grounding = self._ground_provisions(result.get("provisions", []), documents)

            logger.info(
                f"AnalysisAgent [{mode}] complete — {len(result.get('provisions', []))} provisions "
                f"({grounding['grounded']} grounded, {grounding['ungrounded']} ungrounded)"
            )

            return AgentOutput(
                result=result,
                metadata={"agent": self.role.value, "mode": mode,
                          "documents_analyzed": len(documents),
                          "grounding": grounding},
                success=True,
            )

        except Exception as e:
            logger.error(f"AnalysisAgent failed: {e}")
            return AgentOutput(
                result={"provisions": [], "mode": "legal_explanation"},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    # ── mode 1: legal explanation ──────────────────────────────────────────────

    def _legal_explanation(
        self,
        documents: List[Dict],
        structured_query: Dict,
        extra_instructions: str,
    ) -> Dict:
        """Extract and explain provisions for a general legal question."""

        articles = [d for d in documents if d.get("result_type") == "legal_article"]
        docs_text = self._format_articles(articles[:8])

        user_message = f"""Legal question context:
- Domain: {structured_query.get('legal_domain', 'N/A')}
- Key entities: {', '.join(structured_query.get('key_entities', []))}
- Questions: {', '.join(structured_query.get('legal_questions', []))}
{f'- Extra instructions: {extra_instructions}' if extra_instructions else ''}

Retrieved legal articles:
{docs_text}

Extract all provisions relevant to the query. Return JSON:
{{
  "provisions": [
    {{
      "article_number": "549",
      "provision_text": "exact text of the article",
      "legal_principle": "the rule of law this article establishes",
      "conditions": ["condition 1", "condition 2"],
      "penalties_or_consequences": "what the law prescribes",
      "relevance": "why this is relevant to the query",
      "source_document": "document identifier"
    }}
  ],
  "legal_summary": "2-3 sentence summary of what the law says on this topic"
}}"""

        response = self.invoke_llm(user_message)
        return self._parse_json(response, fallback={"provisions": [], "legal_summary": ""})

    # ── mode 2: case assessment ────────────────────────────────────────────────

    def _case_assessment(
        self,
        documents: List[Dict],
        structured_query: Dict,
        extracted_facts: List[str],
        extra_instructions: str,
    ) -> Dict:
        """Apply provisions to the specific facts and compare with similar rulings."""

        articles = [d for d in documents if d.get("result_type") == "legal_article"]
        rulings  = [d for d in documents if d.get("result_type") == "court_ruling"]

        articles_text = self._format_articles(articles[:6])
        rulings_text  = self._format_rulings(rulings[:3])

        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "See query."

        user_message = f"""Case analysis task.

FACTS OF THE CASE:
{facts_text}

ORIGINAL QUERY:
{structured_query.get('original_query', '')}
{f'Extra instructions: {extra_instructions}' if extra_instructions else ''}

RETRIEVED LEGAL ARTICLES:
{articles_text}

SIMILAR COURT RULINGS:
{rulings_text}

Analyze this case. Return JSON:
{{
  "provisions": [
    {{
      "article_number": "549",
      "provision_text": "exact article text",
      "legal_principle": "what rule this establishes",
      "applies_to_case": true | false,
      "application_reasoning": "why/how this article applies to the specific facts above",
      "source_document": "identifier"
    }}
  ],
  "similar_cases": [
    {{
      "case_id": "ruling document id",
      "court": "court name",
      "decision_date": "date",
      "outcome": "conviction / acquittal / etc.",
      "similarity_reasoning": "how this case compares to the current facts",
      "applicable_articles": "articles cited in that ruling",
      "sentence": "sentence imposed if any"
    }}
  ],
  "case_assessment": {{
    "applicable_articles": ["list of article numbers that clearly apply"],
    "strength_of_case": "strong / moderate / weak",
    "likely_outcome": "brief prediction based on law and precedent",
    "key_legal_issues": ["issue 1", "issue 2"],
    "mitigating_factors": ["factor 1"],
    "aggravating_factors": ["factor 1"]
  }}
}}"""

        response = self.invoke_llm(user_message)
        return self._parse_json(response, fallback={
            "provisions": [], "similar_cases": [],
            "case_assessment": {"applicable_articles": [], "likely_outcome": ""}
        })

    # ── grounding ────────────────────────────────────────────────────────────────

    def _ground_provisions(self, provisions: List[Dict], documents: List[Dict]) -> Dict:
        """Tag each provision with whether its article number appears in the
        retrieved documents, and attach the source document_type for citation.

        Provisions are not dropped — ungrounded ones are flagged (`grounded=False`)
        so the UI and downstream agents can surface possible hallucinations.
        """
        # Map article_number -> document_type from the retrieved legal articles.
        retrieved = {}
        for doc in documents:
            if doc.get("result_type") != "legal_article":
                continue
            meta = doc.get("metadata", {})
            num = self._digits(meta.get("article_number", ""))
            if num:
                retrieved[num] = meta.get("document_type", "penal_code")

        grounded = ungrounded = 0
        for prov in provisions:
            num = self._digits(prov.get("article_number", ""))
            if num and num in retrieved:
                prov["grounded"] = True
                prov.setdefault("document_type", retrieved[num])
                grounded += 1
            else:
                prov["grounded"] = False
                ungrounded += 1
                logger.warning(
                    f"AnalysisAgent: provision article '{prov.get('article_number')}' "
                    f"not found in retrieved documents (possible hallucination)"
                )

        return {
            "grounded": grounded,
            "ungrounded": ungrounded,
            "retrieved_article_count": len(retrieved),
        }

    @staticmethod
    def _digits(value) -> str:
        m = re.search(r"\d+", str(value))
        return m.group(0) if m else ""

    # ── helpers ────────────────────────────────────────────────────────────────

    def _format_articles(self, articles: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(articles, 1):
            meta = doc.get("metadata", {})
            parts.append(
                f"[Article {i}]\n"
                f"Number: {meta.get('article_number', 'N/A')} | "
                f"Language: {meta.get('document_language', 'N/A')} | "
                f"Type: {meta.get('document_type', 'N/A')}\n"
                f"Text: {doc.get('content', '')[:800]}"
            )
        return "\n\n---\n\n".join(parts) if parts else "No articles retrieved."

    def _format_rulings(self, rulings: List[Dict]) -> str:
        if not rulings:
            return "No similar court rulings retrieved."
        parts = []
        for i, doc in enumerate(rulings, 1):
            meta = doc.get("metadata", {})
            parts.append(
                f"[Ruling {i}]\n"
                f"ID: {meta.get('document_id', 'N/A')} | "
                f"Court: {meta.get('court', 'N/A')} | "
                f"Date: {meta.get('decision_date', 'N/A')} | "
                f"Outcome: {meta.get('outcome', 'N/A')}\n"
                f"Articles: {meta.get('applicable_articles', 'N/A')}\n"
                f"Sentence: {meta.get('sentence_final', 'N/A')}\n"
                f"Content: {doc.get('content', '')[:600]}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse_json(self, text: str, fallback: Dict) -> Dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.error(f"AnalysisAgent: cannot parse JSON response")
        return fallback
