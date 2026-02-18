# Best Practices - AI Systems Development

## 🎯 General Principles

### 1. Separation of Concerns
**DO:**
```python
# Clear module boundaries
from src.llm.langchain_wrapper import SafeLangChainWrapper
from src.rag.rag_pipeline import EfficientRAGPipeline
from src.security.input_validator import validate_input
```

**DON'T:**
```python
# Mixing concerns in a single module
class AISystem:
    def __init__(self):
        self.llm = ...
        self.rag = ...
        self.db = ...
        self.auth = ...  # Too many responsibilities
```

### 2. Error Handling
**DO:**
```python
from src.core.retry_handler import retry_with_backoff

@retry_with_backoff(max_attempts=3)
async def call_llm(prompt: str):
    try:
        result = await llm.generate(prompt)
        return result
    except RateLimitError as e:
        logger.warning(f"Rate limit hit: {e}")
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise
```

**DON'T:**
```python
# Silent failures
try:
    result = llm.generate(prompt)
except:
    pass  # Never do this!
```

### 3. Configuration Management
**DO:**
```python
from config import get_settings

settings = get_settings()
model = settings.llm.default_model
max_tokens = settings.llm.default_max_tokens
```

**DON'T:**
```python
# Hardcoded values
model = "gpt-4"  # No flexibility
max_tokens = 2000
```

## 🔒 Security Best Practices

### 1. Input Validation
**DO:**
```python
from src.security.input_validator import validate_input

validation = validate_input(user_prompt, check_injection=True)
if not validation.is_valid:
    raise ValueError(f"Invalid input: {validation.issues}")
sanitized = validation.sanitized_input
```

**DON'T:**
```python
# Direct use of user input
result = llm.generate(user_prompt)  # Vulnerable to injection
```

### 2. Secret Management
**DO:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

**DON'T:**
```python
# Hardcoded secrets
api_key = "sk-abc123..."  # Never commit secrets!
```

### 3. Cost Control
**DO:**
```python
from src.core.cost_tracker import get_cost_tracker

tracker = get_cost_tracker()
estimate = tracker.estimate_cost(model, prompt_tokens, completion_tokens)
if not tracker.check_limits(estimate.total_cost):
    raise ValueError("Cost limit exceeded")
```

**DON'T:**
```python
# Unlimited spending
result = llm.generate(prompt, max_tokens=100000)  # Could be expensive!
```

## 🚀 Performance Optimization

### 1. Async Operations
**DO:**
```python
import asyncio

async def process_multiple_queries(queries: list[str]):
    tasks = [rag.query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

**DON'T:**
```python
# Synchronous processing
def process_multiple_queries(queries: list[str]):
    results = []
    for q in queries:
        results.append(rag.query(q))  # Slow!
    return results
```

### 2. Caching
**DO:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_data: str) -> str:
    # Cached for repeated calls
    return result
```

**DON'T:**
```python
# Repeated expensive operations
def get_embedding(text: str):
    return embeddings.embed(text)  # Called multiple times with same text
```

### 3. RAG Optimization
**DO:**
```python
# Optimized chunking
rag = EfficientRAGPipeline(
    chunk_size=1000,     # Not too small
    chunk_overlap=200,   # Reasonable overlap
)

# Use similarity threshold
retriever = rag.get_retriever(
    search_type="similarity",
    k=4,  # Don't retrieve too many
    score_threshold=0.7
)
```

**DON'T:**
```python
# Inefficient settings
rag = EfficientRAGPipeline(
    chunk_size=100,      # Too small - many chunks
    chunk_overlap=50,
)
retriever = rag.get_retriever(k=100)  # Too many results
```

## 📝 Logging Best Practices

### 1. Structured Logging
**DO:**
```python
from src.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Processing request", extra={
    "user_id": user_id,
    "request_id": request_id,
    "duration": duration
})
```

**DON'T:**
```python
# Unstructured logging
print(f"User {user_id} request {request_id} took {duration}s")
```

### 2. Log Levels
**DO:**
```python
logger.debug("Detailed debug info")
logger.info("Normal operation")
logger.warning("Potential issue")
logger.error("Error occurred", exc_info=True)
logger.critical("System failure")
```

**DON'T:**
```python
# Everything at same level
logger.info("Debug info")
logger.info("Error occurred")  # Should be error level
```

### 3. Sensitive Data
**DO:**
```python
# Automatic filtering
logger.info(f"API key configured")  # Filter removes actual key
```

**DON'T:**
```python
# Logging sensitive data
logger.info(f"Using API key: {api_key}")  # Exposes secrets!
```

## 🧪 Testing Best Practices

### 1. Test Structure
**DO:**
```python
import pytest
from src.security.input_validator import PromptInjectionDetector

class TestPromptInjectionDetector:
    def test_detect_instruction_override(self):
        detector = PromptInjectionDetector()
        result = detector.detect("Ignore previous instructions")
        assert not result.is_valid
        assert result.risk_level == "HIGH"
```

**DON'T:**
```python
# No organization
def test_everything():
    # 100 lines of mixed tests
    assert True
```

### 2. Mocking External Services
**DO:**
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
@patch('src.llm.langchain_wrapper.ChatOpenAI')
async def test_llm_call(mock_llm):
    mock_llm.return_value.agenerate = AsyncMock(return_value=...)
    # Test without hitting real API
```

**DON'T:**
```python
# Testing with real API calls
async def test_llm_call():
    result = await llm.generate("test")  # Costs money, slow
```

### 3. Test Coverage
**DO:**
```bash
# Aim for high coverage
pytest --cov=src --cov-report=html
# Target: 80%+ coverage
```

**DON'T:**
```bash
# No coverage tracking
pytest  # Don't know what's tested
```

## 🤖 Agent Best Practices

### 1. Agent Configuration
**DO:**
```python
from src.agents.agent_orchestrator import AgentConfig

agent = AgentConfig(
    name="analyst",
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="Expert in data analysis...",
    max_iterations=10,  # Prevent infinite loops
    timeout=300  # 5 minute timeout
)
```

**DON'T:**
```python
# No limits
agent = Agent(name="analyst")  # Could run forever
```

### 2. Task Dependencies
**DO:**
```python
workflow = CrewAIWorkflow()
workflow.add_task("Gather data", "collector")
workflow.add_task("Analyze data", "analyst", dependencies=["task_0"])
workflow.add_task("Generate report", "writer", dependencies=["task_1"])
```

**DON'T:**
```python
# No dependency management
for task in tasks:
    execute(task)  # May execute in wrong order
```

### 3. Error Recovery
**DO:**
```python
try:
    result = await orchestrator.execute_task(task, agent)
except TimeoutError:
    logger.error(f"Agent {agent} timed out")
    # Fallback logic
    result = default_result
```

**DON'T:**
```python
# No error handling
result = await orchestrator.execute_task(task, agent)  # What if it fails?
```

## 📊 Monitoring Best Practices

### 1. Metrics
**DO:**
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@REQUEST_DURATION.time()
async def handle_request():
    REQUEST_COUNT.labels(endpoint='/api/v1/completions').inc()
    # Handle request
```

**DON'T:**
```python
# No metrics
async def handle_request():
    # Just process, no tracking
    pass
```

### 2. Health Checks
**DO:**
```python
@app.get("/health")
async def health_check():
    # Check dependencies
    db_ok = await check_database()
    redis_ok = await check_redis()
    
    if not (db_ok and redis_ok):
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    return {"status": "healthy"}
```

**DON'T:**
```python
# Always returns healthy
@app.get("/health")
async def health_check():
    return {"status": "healthy"}  # Doesn't check anything
```

### 3. Alerting
**DO:**
```python
# Alert on high error rate
if error_rate > 0.05:  # 5%
    send_alert("High error rate detected")

# Alert on cost spike
if hourly_cost > budget_threshold:
    send_alert("Cost threshold exceeded")
```

**DON'T:**
```python
# No alerting
if error_rate > 0.05:
    logger.error("High error rate")  # Just log, no action
```

## 🔄 CI/CD Best Practices

### 1. Automated Testing
**DO:**
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest --cov=src
      - name: Security scan
        run: bandit -r src/
```

**DON'T:**
```yaml
# No CI
# Manual testing only
```

### 2. Docker Image Optimization
**DO:**
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
USER appuser  # Non-root
```

**DON'T:**
```dockerfile
# Single stage, root user
FROM python:3.11
COPY . .
RUN pip install -r requirements.txt
# Runs as root!
```

### 3. Environment Management
**DO:**
```bash
# Separate configs per environment
.env.development
.env.staging
.env.production
```

**DON'T:**
```bash
# Single config for all environments
.env  # Same for dev and prod - risky!
```

## 📋 Code Review Checklist

Before submitting:
- [ ] All tests pass
- [ ] Security scan clean
- [ ] No hardcoded secrets
- [ ] Proper error handling
- [ ] Logging added
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Cost implications considered
- [ ] Rate limiting checked
- [ ] Input validation present
- [ ] Resource limits set
- [ ] Metrics instrumented

## 🎓 Common Anti-Patterns to Avoid

### 1. God Objects
```python
# BAD: One class does everything
class AISystem:
    def __init__(self):
        self.llm = ...
        self.rag = ...
        self.agents = ...
        self.db = ...
        self.cache = ...
        # Too many responsibilities!
```

### 2. Magic Numbers
```python
# BAD: Unexplained constants
if len(prompt) > 4000:  # Why 4000?
    chunk_text(prompt, 1000)  # Why 1000?
```

### 3. Callback Hell
```python
# BAD: Nested callbacks
def process(data, callback):
    fetch(data, lambda result:
        validate(result, lambda valid:
            transform(valid, lambda final:
                callback(final))))
```

### 4. Premature Optimization
```python
# BAD: Optimizing before measuring
# Complex caching for rarely-used function
@custom_cache_with_redis_and_ttl_and_lru
def rarely_called_function():
    return simple_value
```

## ✨ Code Quality Tools

```bash
# Type checking
mypy src/

# Linting
pylint src/
flake8 src/

# Security
bandit -r src/

# Formatting
black src/
isort src/

# Dependency checking
safety check
```

## 📚 Further Reading

- Clean Code (Robert C. Martin)
- Python Best Practices
- Twelve-Factor App
- OWASP Security Guidelines
- API Design Best Practices
- Microservices Patterns
