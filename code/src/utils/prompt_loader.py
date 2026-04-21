"""
Prompt Loader Utility
Loads agent prompts from external files
"""

from pathlib import Path
from typing import Dict
from loguru import logger


class PromptLoader:
    """Load and manage agent prompts from files."""

    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}

        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")

    def load_prompt(self, agent_name: str) -> str:
        """
        Load prompt for a specific agent.

        Args:
            agent_name: Name of agent (e.g., 'query_understanding', 'research')

        Returns:
            Prompt text
        """

        # Check cache first
        if agent_name in self._cache:
            return self._cache[agent_name]

        # Map agent names to files
        prompt_files = {
            "document_preprocessing": "agent_0_usecases_processing.txt",
            "query_understanding": "agent_1_query_understanding.txt",
            "research": "agent_2_research.txt",
            "analysis": "agent_3_analysis.txt",
            "reasoning": "agent_4_reasoning.txt",
            "citation": "agent_5_citation.txt",
            "writing": "agent_6_writing.txt",
        }

        if agent_name not in prompt_files:
            logger.error(f"Unknown agent: {agent_name}")
            return ""

        # Load from file
        prompt_file = self.prompts_dir / prompt_files[agent_name]

        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            return ""

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            # Cache it
            self._cache[agent_name] = prompt

            logger.debug(f"Loaded prompt for {agent_name} ({len(prompt)} chars)")
            return prompt

        except Exception as e:
            logger.error(f"Failed to load prompt for {agent_name}: {e}")
            return ""

    def reload_prompts(self):
        """Clear cache and reload all prompts."""
        self._cache.clear()
        logger.info("Prompt cache cleared")


# Global instance
_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


def load_agent_prompt(agent_name: str) -> str:
    """Load prompt for agent (convenience function)."""
    loader = get_prompt_loader()
    return loader.load_prompt(agent_name)


if __name__ == "__main__":
    # Test prompt loading
    loader = PromptLoader()

    agents = [
        "document_preprocessing",
        "query_understanding",
        "research",
        "analysis",
        "reasoning",
        "citation",
        "writing",
    ]

    print("\n" + "=" * 60)
    print("TESTING PROMPT LOADER")
    print("=" * 60)

    for agent in agents:
        prompt = loader.load_prompt(agent)
        if prompt:
            print(f"✓ {agent}: {len(prompt)} characters")
        else:
            print(f"✗ {agent}: FAILED")

    print("=" * 60 + "\n")
