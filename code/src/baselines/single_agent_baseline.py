"""
Baseline 1: Single-Agent System with RAG
For comparison with multi-agent system
"""

from typing import Dict, Any
from loguru import logger

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.vectorstore import LegalVectorStore
from src.config import get_config


class SingleAgentBaseline:
    """
    Baseline: Single-agent system (no multi-agent pipeline)

    Uses ONE Claude call with retrieved documents.
    Tests whether multi-agent approach provides benefit over simple RAG.
    """

    def __init__(self, model: str = "claude-sonnet-4-5", vectorstore: "LegalVectorStore" = None):
        self.model_name = model
        config = get_config()

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.2,
            max_tokens=4096,
            anthropic_api_key=config.anthropic_api_key,
        )

        # Reuse a shared vector store if provided (avoids reloading), else load one.
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = LegalVectorStore()
            try:
                self.vectorstore.load_vectorstore()
            except FileNotFoundError:
                logger.warning("Vector store not found. Build it first.")

        logger.info(f"Initialized single-agent baseline with {model}")

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process legal query with single agent."""

        logger.info(f"[BASELINE] Processing: {user_query[:100]}...")

        try:
            # 1. Retrieve documents
            documents = self.vectorstore.search(
                query=user_query, k=5, strategy="hybrid"
            )

            # 2. Format documents
            docs_text = "\n\n---\n\n".join(
                [
                    f"Document {i+1}:\n{doc.page_content}"
                    for i, doc in enumerate(documents)
                ]
            )

            # 3. Single comprehensive prompt
            system_prompt = """You are a Lebanese Legal AI Assistant.

Analyze the user's legal question and provide a comprehensive legal memorandum in the user's language.

Use the retrieved legal documents to support your analysis.

Structure your response as a professional legal memorandum:
1. Introduction
2. Facts
3. Legal Analysis (with citations)
4. Conclusion

Be thorough, precise, and cite specific articles."""

            user_message = f"""Legal Question:
{user_query}

Retrieved Legal Documents:
{docs_text}

Provide a complete legal analysis."""

            # 4. Single LLM call
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            response = self.llm.invoke(messages)
            memorandum = response.content

            logger.info("[BASELINE] Analysis complete")

            return {
                "memorandum": memorandum,
                "documents_retrieved": len(documents),
                "success": True,
                "system": "single_agent_baseline",
            }

        except Exception as e:
            logger.error(f"[BASELINE] Error: {e}")
            return {
                "memorandum": "",
                "error": str(e),
                "success": False,
                "system": "single_agent_baseline",
            }


def main():
    """Test baseline."""
    baseline = SingleAgentBaseline()

    test_query = "ما هي المسؤولية المدنية للمدير المالي الذي قام بتحويل أموال الشركة؟"

    result = baseline.process_query(test_query)

    if result["success"]:
        print("\n" + "=" * 80)
        print("SINGLE-AGENT BASELINE RESULT")
        print("=" * 80)
        print(result["memorandum"])
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
