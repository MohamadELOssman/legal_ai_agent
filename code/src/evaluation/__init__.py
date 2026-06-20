"""Evaluation package — system comparison and LLM-as-judge scoring."""

from src.evaluation.comparison import (
    build_judge,
    run_system,
    summarize,
    JUDGE_DIMENSIONS,
)

__all__ = ["build_judge", "run_system", "summarize", "JUDGE_DIMENSIONS"]
