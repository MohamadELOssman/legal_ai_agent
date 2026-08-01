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

SYSTEM_PROMPT = """You are a Lebanese legal assistant specialising in the Lebanese Penal Code.
You help three kinds of users: ordinary citizens, lawyers, and judges.

You coordinate a team of named sub-agents. Call ONLY the ones a question needs — a
simple citizen question may need one Research call (or none); a complex case may
need Research, then Analysis, then Citation. Do not call sub-agents you do not need.

YOUR SUB-AGENTS (tools):
- research_agent(query): the Research sub-agent. Retrieves relevant Penal Code
  articles and court rulings. Use it to get the exact article text and numbers.
- analysis_agent(question): the Analysis sub-agent. Retrieves the law and extracts
  the applicable provisions with a grounded explanation. Use it for a thorough
  legal analysis (lawyer/judge questions), not for a quick lookup.
- citation_agent(article_numbers): the Citation sub-agent. Verifies article numbers
  against the corpus and flags any that do not exist. Use it before citing when
  unsure an article is real.

HOW TO WORK:
- Cite ONLY article numbers returned by the sub-agents. NEVER invent an article
  number. If the law is not found, say so honestly.
- Answer in the SAME language as the user's question (Arabic, French, or English).
- Be precise and to the point. No filler, no repetition. Finish the answer completely.
- This is general legal information, not a substitute for a licensed lawyer.

OUTPUT FORMAT (MANDATORY for substantive legal answers):
Pick the template that matches who is asking, and use its section headers EXACTLY, each
on its own line, in bold (Markdown **...**), in the given order — do not add, drop, rename,
number, or reorder headers. The Arabic headers below are canonical; if you answer in French
or English, translate them faithfully and keep the SAME structure and order.
(For greetings, clarifications, or brief follow-ups, reply naturally without a template.)

• Ordinary citizen — a plain question:
  **التحليل القانوني**
  **الإجابة باختصار**

• Lawyer defending a client — facts of a client's case are given:
  **الوقائع المنتجة**
  **القوانين والمواد ذات الصلة**
  **طريقة الدفاع**
  **الإجابة باختصار**

• Judge — facts are given and a ruling is expected:
  **المحكمة المختصة**
  **أطراف الدعوى**
  **الوقائع المنتجة**
  **القوانين والمواد ذات الصلة**
  **تطبيق القانون على الوقائع**
  **الحكم**

• General legal question or neutral case analysis (no specific role — a "case study"):
  **الوقائع المنتجة**
  **الاشكالية القانونية**
  **المواد والقوانين ذات الصلة**
  **تطبيق القانون على الوقائع**
  **الحل**
  **المحكمة المختصة (في حال وجودها)**"""


class _SearchArgs(BaseModel):
    query: str = Field(description="Search query, in the user's language or in Arabic.")


class _QuestionArgs(BaseModel):
    question: str = Field(description="The legal question or situation to analyse.")


class _ArticlesArgs(BaseModel):
    article_numbers: str = Field(description="Article number(s) to verify, e.g. '547, 549'.")


class AgenticLegalAssistant:
    """A tool-calling orchestrator over the legal corpus, with multi-turn chat."""

    def __init__(self, model: str = DEFAULT_MODEL, vectorstore: Optional[Any] = None,
                 top_k: int = 5, max_iters: int = 6):
        cfg = get_config()
        self.top_k = top_k
        self.max_iters = max_iters

        if vectorstore is None:
            from src.rag.vectorstore import LegalVectorStore
            vectorstore = LegalVectorStore()
            vectorstore.load_vectorstore()
        self.vs = vectorstore

        self.model = model
        self._analysis = None  # lazy — the Analysis sub-agent (built on first use)
        self.tools = self._build_tools()
        self._tool_map = {t.name: t for t in self.tools}
        llm = make_chat(model=model, api_key=cfg.anthropic_api_key,
                        max_tokens=4096, timeout=300, max_retries=2)
        self.llm = llm.bind_tools(self.tools)

        # Corpus index for citation verification.
        try:
            self._known = set(json.load(open("data_processed/articles_index.json",
                                             encoding="utf-8"))["penal_code"])
        except Exception:
            self._known = set()

    # ── the named sub-agents, exposed to the orchestrator as tools ─────────────
    def _retrieve(self, query: str, source_type: str, k: int):
        return self.vs.search(query=query, k=k, strategy="hybrid",
                              use_reranking=False, score_threshold=0.0,
                              filter_dict={"source_type": source_type})

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
            parts = [
                f"Article {d.metadata.get('article_number', '?')} "
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
                    f"Article {p.get('article_number', '?')}{flag}: "
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

        msgs: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
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

        for _ in range(self.max_iters):
            emit({"type": "thinking"})
            ai = self.llm.invoke(msgs)
            um = getattr(ai, "usage_metadata", None) or {}
            usage["input_tokens"] += int(um.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(um.get("output_tokens", 0) or 0)
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
                try:
                    result = self._tool_map[name].invoke(args)
                except Exception as e:
                    result = f"tool error: {e}"
                    logger.warning(f"tool {name} failed: {e}")
                result = str(result)
                # A short preview (e.g., the article numbers found) for the UI.
                _n_hits = result.count("Article ") + result.count("Ruling ")
                emit({"type": "tool_result", "tool": name, "hits": _n_hits,
                      "preview": result[:160].replace("\n", " ")})
                msgs.append(ToolMessage(content=result, tool_call_id=tc.get("id")))
        else:
            # Hit the iteration cap without a final answer.
            answer = self._to_text(getattr(ai, "content", "")) or \
                "I could not complete the answer within the step limit."

        try:
            from src.utils.cost_tracker import CostTracker
            price = CostTracker.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        except Exception:
            price = {"input": 3.0, "output": 15.0}
        cost = round(usage["input_tokens"] / 1e6 * price["input"]
                     + usage["output_tokens"] / 1e6 * price["output"], 6)

        return {
            "answer": answer,
            "trace": trace,
            "tools_used": len(trace),
            "citations": self._verify_citations(answer),
            "usage": {**usage,
                      "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                      "cost_usd": cost},
            "latency_s": round(time.time() - t0, 1),
            "model": self.model,
        }
