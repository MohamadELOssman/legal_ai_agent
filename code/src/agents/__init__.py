"""Legal AI Agents."""

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.document_preprocessing_agent import DocumentPreprocessingAgent
from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.citation_agent import CitationAgent
from src.agents.writing_agent import WritingAgent

__all__ = [
    "BaseAgent",
    "AgentRole",
    "AgentInput",
    "AgentOutput",
    "OrchestratorAgent",
    "DocumentPreprocessingAgent",
    "QueryUnderstandingAgent",
    "ResearchAgent",
    "AnalysisAgent",
    "ReasoningAgent",
    "CitationAgent",
    "WritingAgent",
]
