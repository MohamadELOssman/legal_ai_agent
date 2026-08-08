"""
Agent 2: Research Agent
Retrieves relevant Lebanese legal texts from the RAG database
"""

from typing import List, Dict, Any
from loguru import logger

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL
from src.rag.vectorstore import LegalVectorStore


# Generic / boilerplate "entities" that describe the corpus rather than the legal
# topic. If they leak into the focused retrieval query they match jurisdiction and
# meta articles ("application of Lebanese law", "the Penal Code") instead of the
# subject — so they are stripped before building the concept query.
_GENERIC_ENTITY_SUBSTR = (
    "قانون العقوبات", "القانون اللبناني", "الشريعة اللبنانية", "القانون الجزائي",
    "قانون أصول المحاكمات", "النصوص القانونية", "المواد القانونية", "نص قانوني",
    "penal code", "lebanese law", "code pénal", "criminal procedure",
)
_GENERIC_ENTITY_EXACT = {
    "القانون", "القوانين", "لبنان", "المواد", "النصوص", "العقوبات", "المادة",
    "law", "laws", "article", "articles", "code", "loi",
}


def _focus_terms(entities) -> list:
    """Keep only topic-bearing entities (drop corpus/boilerplate terms)."""
    out = []
    for e in entities or []:
        s = str(e).strip()
        if not s:
            continue
        low = s.lower()
        if s in _GENERIC_ENTITY_EXACT or low in _GENERIC_ENTITY_EXACT:
            continue
        if any(sub in low for sub in _GENERIC_ENTITY_SUBSTR):
            continue
        out.append(s)
    return out


class ResearchAgent(BaseAgent):
    """
    Agent 2: Research Agent

    Responsibility: Retrieve relevant Lebanese legal texts from the database
    Input: Structured query from Agent 1
    Output: Top-k relevant articles from Lebanese Penal Code and court decisions
    Technical Approach: RAG using Chroma vector database with pure semantic search,
                        separate pools for legal articles vs court rulings.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        vectorstore: LegalVectorStore = None,
    ):
        super().__init__(
            role=AgentRole.RESEARCH,
            model=model,
            temperature=temperature,
        )

        if vectorstore is None:
            self.vectorstore = LegalVectorStore()
            try:
                self.vectorstore.load_vectorstore()
            except FileNotFoundError:
                logger.warning(
                    "Vector store not found. Build it first using build_vectorstore.py"
                )
        else:
            self.vectorstore = vectorstore

    def get_system_prompt(self) -> str:
        return """# CONTEXT
You are the Research (retrieval) sub-agent of a multi-agent Lebanese criminal-law assistant.
The searchable corpus contains the Lebanese Penal Code (قانون العقوبات) and the Code of
Criminal Procedure (قانون أصول المحاكمات الجزائية), together with Court of Cassation rulings,
indexed in Arabic (primary) and English. Retrieval is HYBRID — dense semantic embeddings fused
with BM25 keyword matching — and the two codes share article numbers, so the code each article
belongs to is always tracked. Your output is the factual foundation for every later stage: if
the governing provision is not surfaced here, it can never be analysed, cited, or explained
downstream. Recall of the controlling law is therefore the single most important outcome.

# ROLE
You are a specialised legal-information-retrieval agent. You never interpret, argue, or answer
the legal question; you locate and return the primary sources (articles and rulings) most
relevant to it.

# ACTION
- Read the query as a legal CONCEPT rather than a string of words: focus on the offence, the
  act, the penalty, and the surrounding circumstances, and ignore surface boilerplate
  ("which articles", "legal texts", "in Lebanese law").
- Retrieve the articles and rulings that most directly govern that concept, across both codes.
- For a general legal question, prioritise the substantive articles that define the offence and
  fix its penalty; for a fact-based case, also surface aggravating/mitigating provisions and the
  most similar precedents.
- When the query is in French or English, remember the corpus is Arabic-primary — match on the
  underlying legal concept (and, when enabled, an Arabic rendering of it) to preserve recall.
- Favour precision at the top of the list, but never discard a plausibly controlling provision.

# FORMAT
Return the retrieved documents (article number + text + code, or ruling metadata) exactly as the
system provides them, for the Analysis agent to consume. Do not summarise, editorialise the
ranking, or add commentary.

# TARGET
Your consumer is the Analysis agent, not a human reader. Optimise for topical precision and for
recall of the provisions that actually govern the question."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Retrieve relevant legal documents using a single direct semantic search."""

        try:
            structured_query = agent_input.context.get("structured_query", {})

            # The raw question, plus a FOCUSED query built from the extracted legal
            # concepts (key entities). A verbose phrasing such as "which articles /
            # legal texts talk about X in Lebanese law" embeds boilerplate ("legal
            # texts", "Lebanese law") that pulls in jurisdiction/meta articles instead
            # of the actual topic — so for a general question we retrieve on the
            # concept the user is really asking about, not the whole sentence.
            original = structured_query.get("original_query") or agent_input.query or ""
            key_entities = structured_query.get("key_entities") or []
            focused = " ".join(_focus_terms(key_entities)).strip()

            top_k = agent_input.metadata.get("k") or agent_input.context.get("top_k", 5)
            score_threshold = agent_input.metadata.get("score_threshold", 0.3)

            # Retrieval strategy is configurable so the evaluation harness can A/B
            # semantic vs bm25 vs hybrid and rerank on/off. Defaults are EVIDENCE-BASED:
            # eval_retrieval.py on the 196-question benchmark shows HYBRID (BM25+dense)
            # wins (recall@5 0.50 / nDCG 0.41) over semantic (0.43 / 0.36); the
            # English-only cross-encoder reranker degrades this corpus, so it stays off.
            strategy = agent_input.metadata.get("retrieval_strategy", "hybrid")
            use_reranking = agent_input.metadata.get("use_reranking", False)

            orch = agent_input.metadata.get("orchestrator", {})
            # Default to articles_only when no orchestrator decision is supplied
            # (standalone / benchmark calls), so the retrieved count is predictable
            # (== top_k). The end-to-end pipeline always sets the mode explicitly.
            research_mode = orch.get("research", {}).get("mode", "articles_only")

            # Build retrieval query variants (fused via RRF in multi_search):
            #  • general legal question -> search the FOCUSED concept (boilerplate hurts);
            #  • case analysis -> keep the full scenario (facts matter for case matching),
            #    plus the focused concept to sharpen the article pool.
            if research_mode == "articles_and_cases":
                queries = [original] + ([focused] if focused and focused != original else [])
            else:
                queries = [focused] if len(focused) >= 3 else [original]
            query = queries[0] if queries else original

            # Cross-lingual retrieval (opt-in): for EN/FR queries, also search with an
            # Arabic translation and fuse (RRF). OFF by default — it adds an LLM call
            # per query and its benefit is not yet validated. Enable via
            # metadata["cross_lingual"] = True.
            cross_lingual = agent_input.metadata.get("cross_lingual", False)
            lang = (structured_query.get("language") or "").lower()
            if cross_lingual and lang and lang != "ar":
                ar = self._translate_to_arabic(original)
                if ar and ar not in queries:
                    queries.append(ar)

            retrieved_documents = self._retrieve_documents(
                queries=queries,
                top_k=top_k,
                score_threshold=score_threshold,
                research_mode=research_mode,
                strategy=strategy,
                use_reranking=use_reranking,
            )

            logger.info(f"Retrieved {len(retrieved_documents)} legal documents")

            output = AgentOutput(
                result={
                    "retrieved_documents": retrieved_documents,
                    "search_query": query,
                    "total_retrieved": len(retrieved_documents),
                },
                metadata={
                    "agent": self.role.value,
                    "retrieval_strategy": strategy,
                    "use_reranking": use_reranking,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Research agent failed: {e}")
            return AgentOutput(
                result={"retrieved_documents": []},
                metadata={"agent": self.role.value},
                success=False,
                error=str(e),
            )

    def _translate_to_arabic(self, query: str) -> str:
        """Translate a non-Arabic query to Modern Standard Arabic for cross-lingual search."""
        try:
            sp = ("You are a legal translator. Translate the user's legal query into Modern "
                  "Standard Arabic using correct Lebanese legal terminology. "
                  "Output ONLY the translation, with no explanation or quotes.")
            return self.invoke_llm(query, system_prompt=sp).strip()
        except Exception as e:
            logger.warning(f"Cross-lingual translation failed: {e}")
            return ""

    def _search_pool(self, queries, k, strategy, use_reranking, score_threshold, source_type):
        """Search one source pool with RRF fusion across query variants."""
        return self.vectorstore.multi_search(
            queries=queries, k=k, strategy=strategy, use_reranking=use_reranking,
            score_threshold=score_threshold, filter_dict={"source_type": source_type},
        )

    def _retrieve_documents(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: float = 0.3,
        research_mode: str = "articles_and_cases",
        strategy: str = "hybrid",
        use_reranking: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with separate pools for articles and rulings, fusing results
        across the supplied query variants (cross-lingual / multi-query) via RRF.

          articles_only       — legal_code pool only
          articles_and_cases  — legal_code pool + court_ruling pool

        strategy: "semantic" | "bm25" | "hybrid" (BM25 + dense).
        use_reranking: apply cross-encoder reranking to the candidate set.
        """
        article_docs = []
        ruling_docs  = []
        primary = queries[0] if queries else ""

        # ── Part 1: Legal code articles ───────────────────────────────────────
        try:
            results = self._search_pool(queries, top_k, strategy, use_reranking,
                                        score_threshold, "legal_code")
            article_docs = [
                {"content": doc.page_content, "metadata": doc.metadata,
                 "source_query": primary, "result_type": "legal_article"}
                for doc in results
            ]
        except Exception as e:
            logger.warning(f"Legal-code search failed: {e}")

        # ── Part 2: Court rulings (case_analysis mode only) ───────────────────
        if research_mode == "articles_and_cases":
            try:
                results = self._search_pool(queries, top_k, strategy, use_reranking,
                                            score_threshold, "court_ruling")
                ruling_docs = [
                    {"content": doc.page_content, "metadata": doc.metadata,
                     "source_query": primary, "result_type": "court_ruling"}
                    for doc in results
                ]
            except Exception as e:
                logger.warning(f"Court-ruling search failed: {e}")

        logger.info(
            f"Research [{research_mode}, {len(queries)} variant(s)]: "
            f"{len(article_docs)} articles + {len(ruling_docs)} rulings"
        )
        return article_docs + ruling_docs
