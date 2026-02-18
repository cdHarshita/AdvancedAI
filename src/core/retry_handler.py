"""
Robust retry mechanisms with exponential backoff and circuit breaker pattern.
"""

from typing import TypeVar, Callable, Any, Optional, Type, Tuple
from functools import wraps
import asyncio
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from src.core.logging_config import get_logger
from config import get_retry_config

logger = get_logger(__name__)
T = TypeVar('T')


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker reset to CLOSED state")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise e


def retry_with_backoff(
    max_attempts: Optional[int] = None,
    min_wait: Optional[int] = None,
    max_wait: Optional[int] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        exceptions: Tuple of exception types to retry on
    
    Example:
        @retry_with_backoff(max_attempts=3)
        async def call_api():
            ...
    """
    config = get_retry_config()
    
    max_attempts = max_attempts or config.max_retries
    min_wait = min_wait or config.retry_min_wait
    max_wait = max_wait or config.retry_max_wait
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            return await func(*args, **kwargs)
        
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: int, per: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of tokens
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)
        
        if self.allowance > self.rate:
            self.allowance = self.rate
        
        if self.allowance < 1.0:
            return False
        
        self.allowance -= 1.0
        return True
    
    async def wait_if_needed(self) -> None:
        """Async wait if rate limit exceeded."""
        while not self.is_allowed():
            await asyncio.sleep(0.1)


import logging
