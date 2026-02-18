"""
Cost tracking and monitoring for LLM API calls.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for a request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def total(self) -> int:
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)


@dataclass
class CostEstimate:
    """Estimate costs for LLM API calls."""
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostTracker:
    """
    Track and limit costs for LLM API calls.
    Implements cost controls to prevent runaway expenses.
    """
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.requests: list[CostEstimate] = []
        self.daily_limit = 100.0  # $100 daily limit
        self.request_limit = 1.0  # $1 per request limit
    
    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> CostEstimate:
        """
        Estimate cost for a request.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        
        Returns:
            CostEstimate object
        """
        pricing = self.PRICING.get(model, {"prompt": 0.01, "completion": 0.03})
        
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        return CostEstimate(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            model=model
        )
    
    def check_limits(self, estimated_cost: float) -> bool:
        """
        Check if request would exceed cost limits.
        
        Args:
            estimated_cost: Estimated cost for the request
        
        Returns:
            True if within limits, False otherwise
        """
        if estimated_cost > self.request_limit:
            logger.warning(f"Request cost ${estimated_cost:.4f} exceeds limit ${self.request_limit}")
            return False
        
        if self.total_cost + estimated_cost > self.daily_limit:
            logger.warning(f"Daily cost limit ${self.daily_limit} would be exceeded")
            return False
        
        return True
    
    def record_usage(self, cost_estimate: CostEstimate) -> None:
        """Record usage for a request."""
        self.requests.append(cost_estimate)
        self.total_cost += cost_estimate.total_cost
        logger.info(f"Request cost: ${cost_estimate.total_cost:.4f}, Total: ${self.total_cost:.4f}")
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model name for tokenizer
        
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using approximation")
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            "total_cost": self.total_cost,
            "total_requests": len(self.requests),
            "average_cost": self.total_cost / len(self.requests) if self.requests else 0,
            "daily_limit": self.daily_limit,
            "remaining_budget": self.daily_limit - self.total_cost
        }


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    return _cost_tracker
