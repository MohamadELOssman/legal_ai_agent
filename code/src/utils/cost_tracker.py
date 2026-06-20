"""
Cost Tracking Utility
Track API costs for thesis budget management
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class APICall:
    """Record of a single API call."""

    timestamp: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


class CostTracker:
    """Track and analyze API costs."""

    # Pricing (per 1M tokens)
    PRICING = {
        "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
        "claude-opus-4-8": {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    }

    def __init__(self, log_file: str = "./logs/cost_tracking.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log
        self.calls = self._load_log()

    def _load_log(self):
        """Load existing cost log."""
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                data = json.load(f)
                return [APICall(**call) for call in data]
        return []

    def _save_log(self):
        """Save cost log."""
        with open(self.log_file, "w") as f:
            data = [asdict(call) for call in self.calls]
            json.dump(data, f, indent=2)

    def track_call(
        self, agent: str, model: str, input_tokens: int, output_tokens: int
    ):
        """Track a single API call."""

        # Get pricing
        pricing = self.PRICING.get(
            model, {"input": 3.0, "output": 15.0}  # Default to Sonnet
        )

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Create record
        call = APICall(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )

        # Save
        self.calls.append(call)
        self._save_log()

        return call

    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return sum(call.total_cost for call in self.calls)

    def get_cost_by_agent(self) -> Dict[str, float]:
        """Get cost breakdown by agent."""
        costs = {}
        for call in self.calls:
            costs[call.agent] = costs.get(call.agent, 0.0) + call.total_cost
        return costs

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage."""
        return {
            "input_tokens": sum(call.input_tokens for call in self.calls),
            "output_tokens": sum(call.output_tokens for call in self.calls),
            "total_tokens": sum(
                call.input_tokens + call.output_tokens for call in self.calls
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""

        if not self.calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
            }

        return {
            "total_calls": len(self.calls),
            "total_cost": self.get_total_cost(),
            "cost_by_agent": self.get_cost_by_agent(),
            "token_usage": self.get_token_usage(),
            "avg_cost_per_call": self.get_total_cost() / len(self.calls),
        }

    def print_summary(self):
        """Print cost summary."""

        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("API COST SUMMARY")
        print("=" * 60)
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Total Tokens: {stats['token_usage']['total_tokens']:,}")
        print(
            f"  - Input: {stats['token_usage']['input_tokens']:,} tokens (${stats['token_usage']['input_tokens'] * 0.000003:.4f})"
        )
        print(
            f"  - Output: {stats['token_usage']['output_tokens']:,} tokens (${stats['token_usage']['output_tokens'] * 0.000015:.4f})"
        )

        if stats["total_calls"] > 0:
            print(f"\nAverage Cost per Call: ${stats['avg_cost_per_call']:.4f}")

        if stats.get("cost_by_agent"):
            print("\nCost by Agent:")
            for agent, cost in stats["cost_by_agent"].items():
                print(f"  {agent}: ${cost:.4f}")

        print("=" * 60 + "\n")


# Global tracker instance
_tracker = None


def get_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def track_api_call(agent: str, model: str, input_tokens: int, output_tokens: int):
    """Track an API call (convenience function)."""
    tracker = get_tracker()
    return tracker.track_call(agent, model, input_tokens, output_tokens)


def print_cost_summary():
    """Print cost summary (convenience function)."""
    tracker = get_tracker()
    tracker.print_summary()


if __name__ == "__main__":
    # Example usage
    tracker = CostTracker()

    # Simulate some API calls
    tracker.track_call("query_understanding", "claude-sonnet-4-5", 1000, 300)
    tracker.track_call("research", "claude-sonnet-4-5", 1300, 100)
    tracker.track_call("analysis", "claude-sonnet-4-5", 4200, 800)
    tracker.track_call("reasoning", "claude-sonnet-4-5", 2300, 1200)
    tracker.track_call("citation", "claude-sonnet-4-5", 1400, 200)
    tracker.track_call("writing", "claude-sonnet-4-5", 3500, 2500)

    # Print summary
    tracker.print_summary()
