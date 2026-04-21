"""
Base Agent Class
All specialized agents inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.config import get_config


class AgentRole(Enum):
    """Agent roles in the system."""

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
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.role = role
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        config = get_config()

        # Initialize LLM
        if "claude" in model.lower():
            self.llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=config.anthropic_api_key,
            )
        elif "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=config.google_api_key,
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
        """Invoke the LLM with a message."""

        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"{self.role.value} agent error: {e}")
            raise

    def log_input_output(self, agent_input: AgentInput, output: AgentOutput):
        """Log agent input and output for debugging."""
        logger.debug(f"[{self.role.value}] Input: {agent_input.query[:100]}...")
        logger.debug(f"[{self.role.value}] Output: {str(output.result)[:100]}...")
        logger.debug(f"[{self.role.value}] Success: {output.success}")
