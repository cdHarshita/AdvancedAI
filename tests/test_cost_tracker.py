"""
Tests for cost tracking functionality.
"""

import pytest
from src.core.cost_tracker import CostTracker, TokenUsage, CostEstimate


class TestCostTracker:
    """Test cost tracking and limits."""
    
    def test_estimate_cost_gpt4(self):
        tracker = CostTracker()
        
        estimate = tracker.estimate_cost(
            model="gpt-4-turbo-preview",
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        assert estimate.model == "gpt-4-turbo-preview"
        assert estimate.prompt_cost == 0.01  # 1000 tokens * $0.01/1k
        assert estimate.completion_cost == 0.015  # 500 tokens * $0.03/1k
        assert estimate.total_cost == 0.025
    
    def test_estimate_cost_gpt35(self):
        tracker = CostTracker()
        
        estimate = tracker.estimate_cost(
            model="gpt-3.5-turbo",
            prompt_tokens=2000,
            completion_tokens=1000
        )
        
        assert estimate.prompt_cost == 0.001  # 2000 * $0.0005/1k
        assert estimate.completion_cost == 0.0015  # 1000 * $0.0015/1k
    
    def test_check_limits_per_request(self):
        tracker = CostTracker()
        tracker.request_limit = 1.0
        
        # Should reject expensive request
        assert not tracker.check_limits(1.5)
        
        # Should accept cheap request
        assert tracker.check_limits(0.5)
    
    def test_check_limits_daily(self):
        tracker = CostTracker()
        tracker.daily_limit = 10.0
        tracker.total_cost = 9.0
        
        # Should reject if would exceed daily limit
        assert not tracker.check_limits(2.0)
        
        # Should accept if within daily limit
        assert tracker.check_limits(0.5)
    
    def test_record_usage(self):
        tracker = CostTracker()
        
        estimate = CostEstimate(
            prompt_cost=0.01,
            completion_cost=0.02,
            total_cost=0.03,
            model="gpt-4"
        )
        
        tracker.record_usage(estimate)
        
        assert tracker.total_cost == 0.03
        assert len(tracker.requests) == 1
    
    def test_count_tokens(self):
        tracker = CostTracker()
        
        text = "Hello, how are you?"
        tokens = tracker.count_tokens(text, model="gpt-4")
        
        # Should return a reasonable token count
        assert tokens > 0
        assert tokens < len(text)  # Tokens < characters
    
    def test_get_summary(self):
        tracker = CostTracker()
        
        # Add some usage
        for _ in range(5):
            estimate = CostEstimate(
                prompt_cost=0.01,
                completion_cost=0.02,
                total_cost=0.03,
                model="gpt-4"
            )
            tracker.record_usage(estimate)
        
        summary = tracker.get_summary()
        
        assert summary["total_cost"] == 0.15
        assert summary["total_requests"] == 5
        assert summary["average_cost"] == 0.03
        assert "remaining_budget" in summary


class TestTokenUsage:
    """Test token usage tracking."""
    
    def test_token_usage_total(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert usage.total == 150
    
    def test_token_usage_with_total_set(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.total == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
