"""
Tests for retry and circuit breaker functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from src.core.retry_handler import (
    retry_with_backoff,
    CircuitBreaker,
    RateLimiter
)


class TestRetryWithBackoff:
    """Test retry decorator."""
    
    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=1)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_retry_max_attempts(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=1)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            await always_fails()
        
        assert call_count == 3
    
    def test_sync_retry_success(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_closed_initially(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "CLOSED"
    
    def test_circuit_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        def failing_function():
            raise Exception("Failure")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            try:
                cb.call(failing_function)
            except Exception:
                pass
        
        assert cb.state == "OPEN"
        assert cb.failure_count == 3
    
    def test_circuit_rejects_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        
        def failing_function():
            raise Exception("Failure")
        
        # Fail twice to open circuit
        for _ in range(2):
            try:
                cb.call(failing_function)
            except Exception:
                pass
        
        # Should reject calls when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(lambda: "success")
    
    def test_circuit_allows_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        def success_function():
            return "success"
        
        result = cb.call(success_function)
        assert result == "success"
        assert cb.state == "CLOSED"


class TestRateLimiter:
    """Test rate limiter."""
    
    def test_initial_allowance(self):
        limiter = RateLimiter(rate=10, per=60)
        assert limiter.allowance == 10
    
    def test_allows_requests_within_limit(self):
        limiter = RateLimiter(rate=5, per=60)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert limiter.is_allowed()
        
        # Should reject 6th request
        assert not limiter.is_allowed()
    
    def test_refills_over_time(self):
        import time
        
        limiter = RateLimiter(rate=10, per=1)  # 10 per second
        
        # Consume all tokens
        for _ in range(10):
            limiter.is_allowed()
        
        # Wait for refill
        time.sleep(0.2)
        
        # Should allow new requests
        assert limiter.is_allowed()
    
    @pytest.mark.asyncio
    async def test_async_wait_if_needed(self):
        limiter = RateLimiter(rate=5, per=1)
        
        # Consume tokens
        for _ in range(5):
            limiter.is_allowed()
        
        # This should wait and then succeed
        await limiter.wait_if_needed()
        assert True  # If we get here, wait worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
