"""Configuration management for Legal AI System."""

from pathlib import Path
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


# Canonical model ID — single source of truth for the whole system.
# Standardized on Claude Sonnet 4.5.
DEFAULT_MODEL = "claude-sonnet-5"  # Sonnet 5 rejects `temperature`; make_chat handles that


class LLMConfig(BaseSettings):
    """LLM configuration."""

    primary: str = DEFAULT_MODEL
    reasoning: str = DEFAULT_MODEL
    fallback: str = "gemini-1.5-pro"
    temperature: float = 0.1
    max_tokens: int = 4096


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""

    vector_store: str = "chroma"
    persist_directory: str = "./data_processed/vectorstore"
    collection_name: str = "lebanese_legal_docs"

    retrieval_strategy: Literal["hybrid", "semantic", "bm25"] = "hybrid"
    top_k: int = 5
    similarity_threshold: float = 0.7

    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: Literal["semantic", "fixed", "article"] = "semantic"

    use_reranking: bool = True
    rerank_top_k: int = 10


class AgentConfig(BaseSettings):
    """Individual agent configuration."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3


class SystemConfig(BaseSettings):
    """Main system configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Model configuration
    primary_llm: str = DEFAULT_MODEL
    reasoning_llm: str = DEFAULT_MODEL
    embedding_model: str = "text-embedding-3-large"

    # RAG
    retrieval_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # System
    max_agent_retries: int = 2
    # Per-request timeout (seconds). Must accommodate long legal generation:
    # the Analysis and Writing agents legitimately take 1-3 minutes, so a short
    # timeout causes spurious timeouts + retries that fail the pipeline.
    agent_timeout: int = 300
    debug: bool = False
    log_level: str = "INFO"
    log_file: str = "./logs/legal_ai.log"

    @classmethod
    def load_from_yaml(cls, config_path: str = "config/config.yaml"):
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            return cls()

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get global configuration instance."""
    return config


def load_yaml_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    config_file = Path(config_path)

    if not config_file.exists():
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
