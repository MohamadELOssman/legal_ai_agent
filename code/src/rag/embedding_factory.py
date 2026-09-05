"""
Embedding Factory - Flexible embedding model initialization
Supports: HuggingFace (local/free), Google Gemini, OpenAI
"""

from typing import Literal, Optional
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import get_config


def get_embeddings(
    provider: Literal["huggingface", "google", "openai"] = "huggingface",
    model_name: Optional[str] = None,
    **kwargs
) -> Embeddings:
    """
    Get embedding model based on provider.

    Args:
        provider: Embedding provider ("huggingface", "google", "openai")
        model_name: Model name (provider-specific)
        **kwargs: Additional provider-specific parameters

    Returns:
        Embeddings instance

    Examples:
        # Free local embeddings
        >>> embeddings = get_embeddings("huggingface")

        # Google Gemini embeddings (requires GOOGLE_API_KEY)
        >>> embeddings = get_embeddings("google", model_name="models/text-embedding-004")

        # OpenAI embeddings (requires OPENAI_API_KEY)
        >>> embeddings = get_embeddings("openai", model_name="text-embedding-3-large")
    """

    config = get_config()

    if provider == "huggingface":
        # Local/Free embeddings - no API key needed
        default_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        model = model_name or default_model

        logger.info(f"Initializing HuggingFace embeddings: {model}")

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=kwargs.get('model_kwargs', {'device': 'cpu'}),
            encode_kwargs=kwargs.get('encode_kwargs', {'normalize_embeddings': True})
        )

    elif provider == "google":
        # Google Gemini embeddings - requires GOOGLE_API_KEY
        default_model = "models/text-embedding-004"  # Latest multilingual model
        model = model_name or default_model

        if not config.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in .env file:\n"
                "GOOGLE_API_KEY=your_key_here"
            )

        logger.info(f"Initializing Google Gemini embeddings: {model}")

        # Available models:
        # - models/text-embedding-004 (768 dims, multilingual, recommended)
        # - models/embedding-001 (legacy, 768 dims)

        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # lazy: google only
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=config.google_api_key,
            task_type=kwargs.get('task_type', 'retrieval_document'),  # or 'retrieval_query'
        )

    elif provider == "openai":
        # OpenAI embeddings - requires OPENAI_API_KEY
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require: pip install langchain-openai"
            )

        default_model = "text-embedding-3-large"
        model = model_name or default_model

        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env file:\n"
                "OPENAI_API_KEY=your_key_here"
            )

        logger.info(f"Initializing OpenAI embeddings: {model}")

        # Available models:
        # - text-embedding-3-large (3072 dims, best quality)
        # - text-embedding-3-small (1536 dims, faster/cheaper)
        # - text-embedding-ada-002 (1536 dims, legacy)

        return OpenAIEmbeddings(
            model=model,
            openai_api_key=config.openai_api_key,
            dimensions=kwargs.get('dimensions'),  # Optional: reduce dimensions
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: 'huggingface', 'google', 'openai'"
        )


def get_google_embedding_models() -> dict:
    """List available Google embedding models with specs."""
    return {
        "text-embedding-004": {
            "full_name": "models/text-embedding-004",
            "dimensions": 768,
            "description": "Latest multilingual model (recommended)",
            "languages": "100+ languages including Arabic",
            "max_input": "2048 tokens",
            "cost": "$0.00001/1K tokens"
        },
        "embedding-001": {
            "full_name": "models/embedding-001",
            "dimensions": 768,
            "description": "Legacy model",
            "languages": "Limited multilingual",
            "max_input": "2048 tokens",
            "cost": "$0.00001/1K tokens"
        }
    }


def get_huggingface_embedding_models() -> dict:
    """List recommended HuggingFace embedding models."""
    return {
        "multilingual-mpnet": {
            "full_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "dimensions": 768,
            "description": "Multilingual MPNet (50+ languages)",
            "languages": "50+ languages including Arabic",
            "cost": "FREE (local)"
        },
        "multilingual-e5": {
            "full_name": "intfloat/multilingual-e5-large",
            "dimensions": 1024,
            "description": "Multilingual E5 (100+ languages, high quality)",
            "languages": "100+ languages",
            "cost": "FREE (local)"
        },
        "labse": {
            "full_name": "sentence-transformers/LaBSE",
            "dimensions": 768,
            "description": "Language-agnostic BERT (109 languages)",
            "languages": "109 languages",
            "cost": "FREE (local)"
        },
        "arabic-mpnet": {
            "full_name": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
            "dimensions": 768,
            "description": "Arabic-specific BERT",
            "languages": "Arabic only",
            "cost": "FREE (local)"
        }
    }


if __name__ == "__main__":
    # Example usage
    print("=== Available Embedding Models ===\n")

    print("Google Models (requires API key):")
    for name, specs in get_google_embedding_models().items():
        print(f"  {name}: {specs['description']}")

    print("\nHuggingFace Models (free, local):")
    for name, specs in get_huggingface_embedding_models().items():
        print(f"  {name}: {specs['description']}")
