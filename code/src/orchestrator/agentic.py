"""
Agentic legal assistant (ADK-style tool-calling orchestrator).

Instead of a fixed 7-step pipeline that always runs every agent, this is a single
orchestrator LLM that is given the sub-capabilities as TOOLS and decides which to
call, how many times, and when it has enough to answer. A simple citizen question
may need one search (or none); a complex case may search several times. This saves
time and tokens and behaves more intelligently. It also supports multi-turn chat.

Tools exposed to the orchestrator (the "subagents"):
  • search_penal_code(query)  — retrieve relevant Penal Code articles
  • search_court_rulings(query) — retrieve relevant court rulings / precedents

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

TOOLS (use them to ground your answer in the real law):
- search_penal_code(query): returns relevant Penal Code articles (number + text).
- search_court_rulings(query): returns relevant court rulings / precedents.

HOW TO WORK:
- Search whenever you need the exact article text or number. Decide how many searches
  you need: a simple question may need ONE search (or none if you are certain); a
  complex case may need several. Do not over-search.
- Cite ONLY article numbers returned by the tools. NEVER invent an article number.
  If the law is not found, say so honestly.
- Answer in the SAME language as the user's question (Arabic, French, or English).
- Adapt the format to the user:
  * ordinary citizen -> a short, clear, jargon-free answer.
  * lawyer (mentions a client / a defence) -> a structured advisory analysis.
  * judge (gives the facts and asks for the ruling) -> a formal decision ending in a verdict.
- Be precise and to the point. No filler, no repetition. Finish the answer completely.
- This is general legal information, not a substitute for a licensed lawyer."""


class _SearchArgs(BaseModel):
    query: str = Field(description="Search query, in the user's language or in Arabic.")


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

    # ── tools (the subagents) ─────────────────────────────────────────────────
    def _build_tools(self):
        def search_penal_code(query: str) -> str:
            docs = self.vs.search(query=query, k=self.top_k, strategy="hybrid",
                                  use_reranking=False, score_threshold=0.0,
                                  filter_dict={"source_type": "legal_code"})
            if not docs:
                return "No matching Penal Code articles found."
            return "\n\n".join(
                f"Article {d.metadata.get('article_number', '?')} "
                f"[{d.metadata.get('document_language', '')}]: {d.page_content[:600]}"
                for d in docs)

        def search_court_rulings(query: str) -> str:
            docs = self.vs.search(query=query, k=4, strategy="hybrid",
                                  use_reranking=False, score_threshold=0.0,
                                  filter_dict={"source_type": "court_ruling"})
            if not docs:
                return "No matching court rulings found."
            return "\n\n".join(
                f"Ruling {d.metadata.get('document_id', '?')} | court: {d.metadata.get('court', '')} "
                f"| outcome: {d.metadata.get('outcome', '')} | articles: "
                f"{d.metadata.get('applicable_articles', '')}\n{d.page_content[:400]}"
                for d in docs)

        return [
            StructuredTool.from_function(
                func=search_penal_code, name="search_penal_code", args_schema=_SearchArgs,
                description="Search the Lebanese Penal Code for articles relevant to a query. "
                            "Returns article numbers and text."),
            StructuredTool.from_function(
                func=search_court_rulings, name="search_court_rulings", args_schema=_SearchArgs,
                description="Search Lebanese court rulings (precedents) relevant to a case. "
                            "Use for case analysis and judicial decisions."),
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
    def chat(self, history: List[Dict[str, str]], user_message: str) -> Dict[str, Any]:
        """Run one assistant turn. `history` is a list of {role, content} (prior turns)."""
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
            ai = self.llm.invoke(msgs)
            um = getattr(ai, "usage_metadata", None) or {}
            usage["input_tokens"] += int(um.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(um.get("output_tokens", 0) or 0)
            msgs.append(ai)  # keep the full message (incl. thinking/tool blocks)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                answer = self._to_text(ai.content)
                break

            for tc in tool_calls:
                name = tc.get("name"); args = tc.get("args", {}) or {}
                trace.append({"tool": name, "query": args.get("query", "")})
                try:
                    result = self._tool_map[name].invoke(args)
                except Exception as e:
                    result = f"tool error: {e}"
                    logger.warning(f"tool {name} failed: {e}")
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc.get("id")))
        else:
            # Hit the iteration cap without a final answer.
            answer = self._to_text(getattr(ai, "content", "")) or \
                "I could not complete the answer within the step limit."

        return {
            "answer": answer,
            "trace": trace,
            "tools_used": len(trace),
            "citations": self._verify_citations(answer),
            "usage": {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]},
            "latency_s": round(time.time() - t0, 1),
        }
