"""
Baseline 2: No RAG (Pure LLM Knowledge)
Tests the value of RAG for Lebanese legal research
"""

from typing import Dict, Any
from loguru import logger

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config


class NoRAGBaseline:
    """
    Baseline: Pure LLM without RAG

    Uses only Claude's pre-trained knowledge.
    Tests whether RAG provides significant improvement.
    """

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.model_name = model
        config = get_config()

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.2,
            max_tokens=4096,
            anthropic_api_key=config.anthropic_api_key,
        )

        logger.info(f"Initialized no-RAG baseline with {model}")

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process legal query without RAG."""

        logger.info(f"[NO-RAG] Processing: {user_query[:100]}...")

        try:
            # System prompt - emphasizes Lebanese law knowledge
            system_prompt = """You are a Lebanese Legal AI Assistant with knowledge of Lebanese law.

Analyze the user's legal question and provide a legal memorandum based on your knowledge of Lebanese law, particularly:
- Lebanese Code of Obligations and Contracts
- Lebanese Penal Code
- Lebanese Civil Procedure
- Court of Cassation precedents

Structure your response as a professional legal memorandum:
1. Introduction
2. Facts
3. Legal Analysis (cite specific articles if known)
4. Conclusion

Important:
- If you're not certain about specific article numbers, acknowledge this
- Provide general legal principles even if specific citations are unavailable
- Be honest about limitations in your knowledge"""

            user_message = f"""Legal Question:
{user_query}

Provide a complete legal analysis based on your knowledge of Lebanese law."""

            # Single LLM call (no retrieved documents)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            response = self.llm.invoke(messages)
            memorandum = response.content

            logger.info("[NO-RAG] Analysis complete")

            return {
                "memorandum": memorandum,
                "documents_retrieved": 0,  # No RAG
                "success": True,
                "system": "no_rag_baseline",
            }

        except Exception as e:
            logger.error(f"[NO-RAG] Error: {e}")
            return {
                "memorandum": "",
                "error": str(e),
                "success": False,
                "system": "no_rag_baseline",
            }


def main():
    """Test baseline."""
    baseline = NoRAGBaseline()

    test_query = "ما هي المسؤولية المدنية للمدير المالي الذي قام بتحويل أموال الشركة؟"

    result = baseline.process_query(test_query)

    if result["success"]:
        print("\n" + "=" * 80)
        print("NO-RAG BASELINE RESULT")
        print("=" * 80)
        print(result["memorandum"])
        print("\n[Note: This is based purely on LLM knowledge, no RAG]")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
