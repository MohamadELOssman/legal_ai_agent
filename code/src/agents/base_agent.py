"""
Base Agent Class
All specialized agents inherit from this base class
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.config import get_config, DEFAULT_MODEL

try:
    from src.utils.cost_tracker import CostTracker
    _PRICING = CostTracker.PRICING
except Exception:  # pragma: no cover
    _PRICING = {}


class AgentRole(Enum):
    """Agent roles in the system."""

    ORCHESTRATOR = "orchestrator"
    COORDINATOR = "coordinator"
    DOCUMENT_PREPROCESSING = "document_preprocessing"
    QUERY_UNDERSTANDING = "query_understanding"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CITATION = "citation"
    WRITING = "writing"


@dataclass
class AgentInput:
    """Input to an agent."""

    query: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class AgentOutput:
    """Output from an agent."""

    result: Any
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        role: AgentRole,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.role = role
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Per-call usage telemetry (tokens / latency / cost). Reset with reset_usage().
        self.usage_log: List[Dict[str, Any]] = []
        self.last_usage: Dict[str, Any] = {}

        config = get_config()

        # Resilience: retries (with backoff) + per-request timeout from config.
        max_retries = getattr(config, "max_agent_retries", 3)
        timeout = getattr(config, "agent_timeout", 60)

        # Initialize LLM
        if "claude" in model.lower():
            from src.utils.llm import make_chat
            # make_chat omits `temperature` for models that don't accept it
            # (e.g. claude-sonnet-5, claude-opus-4-8).
            self.llm = make_chat(
                model=model,
                api_key=config.anthropic_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                timeout=timeout,
            )
        elif "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=config.google_api_key,
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        logger.info(f"Initialized {self.role.value} agent with {model}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process input and return output."""
        pass

    def invoke_llm(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Invoke the LLM with a message, recording token/latency/cost telemetry."""

        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            t0 = time.time()
            response = self.llm.invoke(messages)
            latency = time.time() - t0
            self._record_usage(response, latency)
            return response.content
        except Exception as e:
            logger.error(f"{self.role.value} agent error: {e}")
            raise

    def invoke_structured(self, user_message: str, schema, system_prompt: Optional[str] = None):
        """Invoke the LLM and return a schema-validated object (Pydantic model).

        Uses the provider's tool/structured-output support so the result is
        guaranteed to match `schema` — eliminating brittle regex JSON parsing.
        Token/latency/cost telemetry is still recorded. Returns the parsed model,
        or raises if the model could not produce a valid structure.
        """
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        structured_llm = self.llm.with_structured_output(schema, include_raw=True)
        t0 = time.time()
        result = structured_llm.invoke(messages)
        latency = time.time() - t0

        # include_raw=True -> {"raw": AIMessage, "parsed": model|None, "parsing_error": ...}
        raw = result.get("raw") if isinstance(result, dict) else None
        if raw is not None:
            self._record_usage(raw, latency)

        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is None:
            err = result.get("parsing_error") if isinstance(result, dict) else "unknown"
            raise ValueError(f"Structured output parsing failed: {err}")
        return parsed

    # ── usage telemetry ──────────────────────────────────────────────────────────

    def _record_usage(self, response, latency: float) -> None:
        """Capture token counts, latency and estimated cost for one LLM call."""
        usage = getattr(response, "usage_metadata", None) or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)

        pricing = _PRICING.get(self.model_name, {"input": 3.0, "output": 15.0})
        cost = (in_tok / 1_000_000) * pricing["input"] + (out_tok / 1_000_000) * pricing["output"]

        record = {
            "agent": self.role.value,
            "model": self.model_name,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "latency_s": round(latency, 3),
            "cost_usd": round(cost, 6),
        }
        self.last_usage = record
        self.usage_log.append(record)

    def reset_usage(self) -> None:
        """Clear accumulated usage telemetry (e.g. between queries)."""
        self.usage_log = []
        self.last_usage = {}

    def usage_summary(self) -> Dict[str, Any]:
        """Aggregate usage across all calls made by this agent instance."""
        return {
            "calls": len(self.usage_log),
            "input_tokens": sum(u["input_tokens"] for u in self.usage_log),
            "output_tokens": sum(u["output_tokens"] for u in self.usage_log),
            "total_tokens": sum(u["total_tokens"] for u in self.usage_log),
            "latency_s": round(sum(u["latency_s"] for u in self.usage_log), 3),
            "cost_usd": round(sum(u["cost_usd"] for u in self.usage_log), 6),
        }

    def log_input_output(self, agent_input: AgentInput, output: AgentOutput):
        """Log agent input and output for debugging."""
        logger.debug(f"[{self.role.value}] Input: {agent_input.query[:100]}...")
        logger.debug(f"[{self.role.value}] Output: {str(output.result)[:100]}...")
        logger.debug(f"[{self.role.value}] Success: {output.success}")
