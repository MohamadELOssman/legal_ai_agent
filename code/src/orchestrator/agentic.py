"""
Agentic legal assistant (ADK-style tool-calling orchestrator).

Instead of a fixed 7-step pipeline that always runs every agent, this is a single
orchestrator LLM that is given the sub-capabilities as TOOLS and decides which to
call, how many times, and when it has enough to answer. A simple citizen question
may need one search (or none); a complex case may search several times. This saves
time and tokens and behaves more intelligently. It also supports multi-turn chat.

The orchestrator keeps the project's named sub-agents and calls only the ones a
given question needs. They are exposed to the orchestrator as tools:
  • Research Agent  — retrieve relevant Penal Code articles + court rulings
  • Analysis Agent  — extract and explain the applicable provisions (grounded)
  • Citation Agent  — verify article numbers against the corpus before citing

Grounding is preserved by (a) instructing the model to cite only retrieved
articles and (b) verifying the cited article numbers against the corpus index.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.config import get_config, DEFAULT_MODEL
from src.utils.llm import make_chat

SYSTEM_PROMPT = """═══════════════════════════════════════════════════════════════
 LEGAL CHAT ORCHESTRATOR — v2   ·   Framework: CRAFT
═══════════════════════════════════════════════════════════════

## Context
- You run a retrieval-augmented Lebanese criminal-law assistant. Specialised sub-agents fetch
  and analyse the REAL law: Penal Code (قانون العقوبات) + Code of Criminal Procedure
  (قانون أصول المحاكمات الجزائية) + Court of Cassation rulings.
- That corpus is the ONLY source of truth; your own memory is not, and an invented article
  number is the single worst failure.
- The two codes share article numbers, so a citation means something only when the code is named.

## Role
- You are the orchestrator and the voice of the assistant, serving citizens, lawyers, and judges.
- You decide which sub-agents to consult, then write the final answer yourself.
- You are decisive and efficient — never exhaustive for its own sake.

## Action
1. Read the question: its language, who is asking, and whether it is a general question or a
   concrete case with facts.
2. Gather only what you need, using your sub-agents (tools):
   • research_agent(query)         → relevant articles (both codes) + rulings, each labelled with
     its code; use it for exact article text and numbers.
   • analysis_agent(question)      → retrieves the law and extracts the applicable provisions with
     a grounded explanation; use it for a thorough analysis (lawyer/judge cases), not a lookup.
   • citation_agent(article_numbers) → verifies article numbers exist in the corpus; use before
     citing when unsure a number is real.
3. BE DECISIVE — do not loop: use AT MOST 2-3 calls total, NEVER repeat a call with the same or a
   near-identical query, and once you have the relevant provisions, STOP and write the answer.
4. Ground everything in what the sub-agents returned: cite ONLY numbers they returned (NEVER
   invent one; if the law is not found, say so), and ALWAYS name the code for each citation
   (e.g. "المادة 24 من قانون العقوبات" vs "المادة 24 من قانون أصول المحاكمات الجزائية").

## Format  (MANDATORY for substantive legal answers)
- Pick the template that matches who is asking and use its section headers EXACTLY, in order.
- Write each header as a Markdown level-3 heading — the line MUST start with "### " (never
  "#"/"##", never bold) — followed by its content.
- Do not add, drop, rename, number, or reorder headers. The Arabic headers below are canonical;
  in French or English, translate them faithfully and keep the same structure and order.
- For greetings, clarifications, or brief follow-ups, reply naturally without a template.

• Ordinary citizen — a plain question:
  ### التحليل القانوني
  ### الإجابة باختصار

• Lawyer defending a client — facts of a client's case are given:
  ### الوقائع المنتجة
  ### القوانين والمواد ذات الصلة
  ### طريقة الدفاع
  ### الإجابة باختصار

• Judge — facts are given and a ruling is expected:
  ### المحكمة المختصة
  ### أطراف الدعوى
  ### الوقائع المنتجة
  ### القوانين والمواد ذات الصلة
  ### تطبيق القانون على الوقائع
  ### الحكم

• General legal question or neutral case analysis (no specific role — a "case study"):
  ### الوقائع المنتجة
  ### الاشكالية القانونية
  ### المواد والقوانين ذات الصلة
  ### تطبيق القانون على الوقائع
  ### الحل
  ### المحكمة المختصة (في حال وجودها)

## Target
- Answer in the SAME language as the user's question.
- Match the reader: plain and reassuring for a citizen; precise and strategic for a lawyer;
  formal and impartial for a judge.
- Be precise and to the point — no filler, no repetition — and finish completely.
- This is general legal information, not a substitute for a licensed lawyer."""


# Readable code names by document_type (for labelling retrieved articles).
_CODE_LABEL = {
    "penal_code": "Penal Code",
    "criminal_procedure_code": "Code of Criminal Procedure",
    "code_obligations_contracts": "Code of Obligations & Contracts",
}


def _code_of(meta) -> str:
    return _CODE_LABEL.get(meta.get("document_type", ""), "Penal Code")


class _SearchArgs(BaseModel):
    query: str = Field(description="Search query, in the user's language or in Arabic.")


class _QuestionArgs(BaseModel):
    question: str = Field(description="The legal question or situation to analyse.")


class _ArticlesArgs(BaseModel):
    article_numbers: str = Field(description="Article number(s) to verify, e.g. '547, 549'.")


class AgenticLegalAssistant:
    """A tool-calling orchestrator over the legal corpus, with multi-turn chat."""

    def __init__(self, model: str = DEFAULT_MODEL, vectorstore: Optional[Any] = None,
                 top_k: int = 5, max_iters: int = 8, max_tool_calls: int = 6,
                 disabled_tools: Optional[Any] = None):
        cfg = get_config()
        self.top_k = top_k
        self.max_iters = max_iters
        self.max_tool_calls = max_tool_calls   # after this many retrievals, force an answer
        # Sub-agents to disable for this run (ablation studies): a set of tool names.
        self.disabled_tools = set(disabled_tools or ())

        if vectorstore is None:
            from src.rag.vectorstore import LegalVectorStore
            vectorstore = LegalVectorStore()
            vectorstore.load_vectorstore()
        self.vs = vectorstore

        self.model = model
        self._analysis = None  # lazy — the Analysis sub-agent (built on first use)
        self._sources = []     # documents retrieved during the current turn
        # Build all sub-agent tools, then drop any that are disabled for this run.
        self.tools = [t for t in self._build_tools() if t.name not in self.disabled_tools]
        self._tool_map = {t.name: t for t in self.tools}

        # Per-instance system prompt: note any unavailable sub-agents so the
        # orchestrator does not try to rely on them.
        self.system_prompt = SYSTEM_PROMPT
        if self.disabled_tools:
            self.system_prompt += (
                "\n\n## Constraint (ablation)\n- The following sub-agent(s) are UNAVAILABLE for "
                "this run: " + ", ".join(sorted(self.disabled_tools)) + ". Do not rely on them; "
                "answer using only the sub-agents you have.")

        llm = make_chat(model=model, api_key=cfg.anthropic_api_key,
                        max_tokens=4096, timeout=300, max_retries=2)
        self._llm_plain = llm               # no tools — used to force a final answer
        self.llm = llm.bind_tools(self.tools) if self.tools else llm

        # Corpus index for citation verification — union across every code in the
        # index (penal_code, criminal_procedure_code, ...), so citations to any
        # ingested code are recognised.
        try:
            _idx = json.load(open("data_processed/articles_index.json", encoding="utf-8"))
            self._known = set().union(*(set(v) for v in _idx.values())) if _idx else set()
        except Exception:
            self._known = set()

    # ── the named sub-agents, exposed to the orchestrator as tools ─────────────
    def _retrieve(self, query: str, source_type: str, k: int):
        return self.vs.search(query=query, k=k, strategy="hybrid",
                              use_reranking=False, score_threshold=0.0,
                              filter_dict={"source_type": source_type})

    def _record_article(self, d):
        self._sources.append({
            "kind": "article", "code": _code_of(d.metadata),
            "number": str(d.metadata.get("article_number", "?")),
            "lang": d.metadata.get("document_language", ""),
            "text": (d.page_content or "")[:500]})

    def _record_ruling(self, d):
        self._sources.append({
            "kind": "ruling", "id": str(d.metadata.get("document_id", "?")),
            "court": d.metadata.get("court", ""),
            "outcome": d.metadata.get("outcome", ""),
            "articles": d.metadata.get("applicable_articles", ""),
            "text": (d.page_content or "")[:300]})

    def _get_analysis_agent(self):
        """Build the Analysis sub-agent lazily (first time it is actually used)."""
        if self._analysis is None:
            from src.agents.analysis_agent import AnalysisAgent
            self._analysis = AnalysisAgent(model=self.model)
        return self._analysis

    def _build_tools(self):
        def research_agent(query: str) -> str:
            """Research sub-agent: retrieve relevant articles and rulings."""
            arts = self._retrieve(query, "legal_code", self.top_k)
            rulings = self._retrieve(query, "court_ruling", 3)
            for d in arts:
                self._record_article(d)
            for d in rulings:
                self._record_ruling(d)
            parts = [
                f"{_code_of(d.metadata)} — Article {d.metadata.get('article_number', '?')} "
                f"[{d.metadata.get('document_language', '')}]: {d.page_content[:600]}"
                for d in arts]
            if rulings:
                parts.append("--- Court rulings ---")
                parts += [
                    f"Ruling {d.metadata.get('document_id', '?')} | court: "
                    f"{d.metadata.get('court', '')} | outcome: {d.metadata.get('outcome', '')} "
                    f"| articles: {d.metadata.get('applicable_articles', '')}\n{d.page_content[:350]}"
                    for d in rulings]
            return "\n\n".join(parts) if parts else "No matching law found."

        def analysis_agent(question: str) -> str:
            """Analysis sub-agent: retrieve + extract the applicable provisions (grounded)."""
            from src.agents.base_agent import AgentInput
            arts = self._retrieve(question, "legal_code", self.top_k)
            if not arts:
                return "No applicable provisions found (no law retrieved)."
            for d in arts:
                self._record_article(d)
            docs = [{"content": d.page_content, "metadata": d.metadata,
                     "result_type": "legal_article"} for d in arts]
            sq = {"original_query": question, "legal_domain": "criminal",
                  "key_entities": [], "legal_questions": [question]}
            out = self._get_analysis_agent().process(AgentInput(
                query=question,
                context={"structured_query": sq,
                         "research_results": {"retrieved_documents": docs}},
                metadata={"orchestrator": {"analysis": {"mode": "legal_explanation"}}}))
            provs = (out.result or {}).get("provisions", [])
            if not provs:
                return "No applicable provisions extracted."
            lines = []
            for p in provs[:6]:
                flag = "" if p.get("grounded", True) else " [UNVERIFIED — do not cite]"
                lines.append(
                    f"{_code_of(p)} — Article {p.get('article_number', '?')}{flag}: "
                    f"{p.get('legal_principle', '')} — "
                    f"{p.get('penalties_or_consequences', p.get('relevance', ''))}")
            summary = (out.result or {}).get("legal_summary", "")
            if summary:
                lines.append(f"Summary: {summary}")
            return "\n".join(lines)

        def citation_agent(article_numbers: str) -> str:
            """Citation sub-agent: verify article numbers against the corpus."""
            import re
            nums = re.findall(r"\d+", article_numbers or "")
            if not nums:
                return "No article numbers provided to verify."
            out = []
            for n in nums:
                ok = n in self._known
                out.append(f"Article {n}: {'verified — safe to cite' if ok else 'NOT in corpus — do not cite'}")
            return "\n".join(out)

        return [
            StructuredTool.from_function(
                func=research_agent, name="research_agent", args_schema=_SearchArgs,
                description="Research sub-agent. Retrieves Lebanese Penal Code articles and "
                            "court rulings relevant to a query. Use for a quick look-up of the "
                            "exact article text and numbers."),
            StructuredTool.from_function(
                func=analysis_agent, name="analysis_agent", args_schema=_QuestionArgs,
                description="Analysis sub-agent. Retrieves the law and extracts the applicable "
                            "provisions with a grounded explanation. Use for a thorough legal "
                            "analysis of a lawyer/judge question, not a simple look-up."),
            StructuredTool.from_function(
                func=citation_agent, name="citation_agent", args_schema=_ArticlesArgs,
                description="Citation sub-agent. Verifies that article numbers exist in the "
                            "corpus and flags any that do not. Use before citing if unsure."),
        ]

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") in (None, "text") and "text" in b)
        return str(content)

    def _verify_citations(self, answer: str) -> Dict[str, Any]:
        from src.evaluation.comparison import extract_cited_articles
        cited = extract_cited_articles(answer)
        verified = sorted((a for a in cited if a in self._known), key=lambda x: int(x))
        unverified = sorted((a for a in cited if a not in self._known), key=lambda x: int(x))
        return {"cited": sorted(cited, key=lambda x: int(x) if x.isdigit() else 0),
                "verified": verified, "unverified": unverified}

    # ── chat ──────────────────────────────────────────────────────────────────
    def chat(self, history: List[Dict[str, str]], user_message: str,
             on_event=None) -> Dict[str, Any]:
        """Run one assistant turn. `history` is a list of {role, content} (prior turns).

        `on_event(event)` (optional) is called live as the agent works, so a UI can
        show ADK-style step-by-step progress. Event types: thinking, tool_call,
        tool_result, answering.
        """
        def emit(ev):
            if on_event:
                try:
                    on_event(ev)
                except Exception:
                    pass

        self._sources = []  # reset per-turn source collector

        msgs: List[Any] = [SystemMessage(content=self.system_prompt)]
        for h in history:
            if h.get("role") == "user":
                msgs.append(HumanMessage(content=h["content"]))
            elif h.get("role") == "assistant":
                msgs.append(AIMessage(content=h["content"]))
        msgs.append(HumanMessage(content=user_message))

        trace: List[Dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        answer = ""
        t0 = time.time()

        seen_queries: Dict[tuple, str] = {}   # (tool, normalized query) -> cached result
        tool_calls_total = 0

        def _acc_usage(ai):
            um = getattr(ai, "usage_metadata", None) or {}
            usage["input_tokens"] += int(um.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(um.get("output_tokens", 0) or 0)

        for it in range(self.max_iters):
            emit({"type": "thinking"})

            # Force a final answer once the retrieval budget is spent (or on the last
            # iteration): invoke WITHOUT tools so the model MUST write the answer from
            # what it already gathered — this stops the endless re-search loop.
            if tool_calls_total >= self.max_tool_calls or it == self.max_iters - 1:
                msgs.append(HumanMessage(content=(
                    "You now have enough information. Write the COMPLETE final answer for the "
                    "user now, in the required format, using the sub-agent results already "
                    "gathered above. Do NOT call any more tools.")))
                emit({"type": "answering"})
                ai = self._llm_plain.invoke(msgs)
                _acc_usage(ai); msgs.append(ai)
                answer = self._to_text(ai.content)
                break

            ai = self.llm.invoke(msgs)
            _acc_usage(ai)
            msgs.append(ai)  # keep the full message (incl. thinking/tool blocks)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                emit({"type": "answering"})
                answer = self._to_text(ai.content)
                break

            for tc in tool_calls:
                name = tc.get("name"); args = tc.get("args", {}) or {}
                # Sub-agents use different arg names (query / question / article_numbers).
                query = (args.get("query") or args.get("question")
                         or args.get("article_numbers") or "")
                trace.append({"tool": name, "query": query})
                emit({"type": "tool_call", "tool": name, "query": query})

                # Short-circuit an exact repeat: return the cached result plus a nudge
                # instead of re-retrieving the same chunks.
                nkey = (name, " ".join(str(query).lower().split()))
                if nkey in seen_queries:
                    result = (seen_queries[nkey] +
                              "\n\n[Note: you already retrieved this — do NOT search again; "
                              "use these results and write the answer.]")
                else:
                    try:
                        result = str(self._tool_map[name].invoke(args))
                    except Exception as e:
                        result = f"tool error: {e}"
                        logger.warning(f"tool {name} failed: {e}")
                    seen_queries[nkey] = result
                tool_calls_total += 1

                # A short preview (e.g., the article numbers found) for the UI.
                _n_hits = result.count("Article ") + result.count("Ruling ")
                emit({"type": "tool_result", "tool": name, "hits": _n_hits,
                      "preview": result[:160].replace("\n", " ")})
                msgs.append(ToolMessage(content=result, tool_call_id=tc.get("id")))

        if not answer:
            answer = self._to_text(getattr(ai, "content", "")) or \
                "I could not complete the answer within the step limit."

        try:
            from src.utils.cost_tracker import CostTracker
            price = CostTracker.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        except Exception:
            price = {"input": 3.0, "output": 15.0}
        cost = round(usage["input_tokens"] / 1e6 * price["input"]
                     + usage["output_tokens"] / 1e6 * price["output"], 6)

        # Dedupe the retrieved sources (articles by code+number, rulings by id).
        seen, sources = set(), []
        for s in self._sources:
            key = (s["kind"], s.get("code", ""), s.get("number") or s.get("id"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(s)

        return {
            "answer": answer,
            "trace": trace,
            "tools_used": len(trace),
            "citations": self._verify_citations(answer),
            "sources": sources,
            "usage": {**usage,
                      "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                      "cost_usd": cost},
            "latency_s": round(time.time() - t0, 1),
            "model": self.model,
        }
