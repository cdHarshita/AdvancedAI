## Description
<!-- Provide a clear and concise description of the changes -->

## Type of Change
- [ ] Month 1: LangChain/RAG Implementation
- [ ] Month 2: Multi-Agent Systems (AutoGen/CrewAI)
- [ ] Month 3: Production Systems (LlamaIndex/Semantic Kernel/Deployment)
- [ ] Documentation
- [ ] Bug Fix
- [ ] Infrastructure/DevOps

## Framework(s) Used
<!-- Check all that apply -->
- [ ] LangChain
- [ ] LlamaIndex
- [ ] AutoGen
- [ ] CrewAI
- [ ] Semantic Kernel
- [ ] FastAPI
- [ ] Docker
- [ ] Other: ___________

## Architectural Decisions

### Design Pattern
<!-- Describe the architectural pattern used (e.g., RAG, Agent-based, Multi-agent, etc.) -->

### Rationale
<!-- Explain why this approach was chosen over alternatives -->

### Trade-offs
<!-- Discuss any trade-offs made (performance, cost, complexity, etc.) -->

## AI Systems Checklist

### 🔒 Security
- [ ] No hardcoded API keys or credentials
- [ ] Environment variables used for sensitive configuration
- [ ] Input validation and sanitization implemented
- [ ] Rate limiting considered for API calls
- [ ] No PII or sensitive data in logs

### 🏛️ Architecture
- [ ] Proper separation of concerns
- [ ] Clean abstractions with minimal coupling
- [ ] Design is production-scalable
- [ ] Error handling with graceful degradation
- [ ] Retry logic with exponential backoff

### 🤖 AI Framework Best Practices

#### LangChain (if applicable)
- [ ] Appropriate chain type selected
- [ ] Callback handlers for monitoring
- [ ] Memory properly managed (token limits)
- [ ] Streaming implemented where beneficial
- [ ] Tools properly defined with clear descriptions

#### LlamaIndex (if applicable)
- [ ] Appropriate index type selected
- [ ] ServiceContext properly configured
- [ ] Query engine optimized for use case
- [ ] Chunk size and overlap justified
- [ ] Metadata filtering utilized

#### AutoGen (if applicable)
- [ ] Agent roles clearly defined
- [ ] Max consecutive auto-reply set
- [ ] Human-in-the-loop where appropriate
- [ ] Termination conditions defined
- [ ] Code execution properly sandboxed

#### CrewAI (if applicable)
- [ ] Task delegation properly structured
- [ ] Agent capabilities match tasks
- [ ] Process flow validated
- [ ] Tool integration tested
- [ ] Output parsing handled

### 📚 RAG Implementation (if applicable)
- [ ] Chunk size appropriate for content type (typically 500-1000 tokens)
- [ ] Chunk overlap configured (typically 10-20%)
- [ ] Embedding model choice justified
- [ ] Vector store selection appropriate for scale
- [ ] Retrieval strategy tested (similarity, MMR, etc.)
- [ ] Reranking considered for improved relevance
- [ ] Metadata filtering utilized

### 💬 Prompt Engineering
- [ ] Prompts are clear, specific, and unambiguous
- [ ] System prompts define role and constraints
- [ ] Few-shot examples provided where beneficial
- [ ] Hallucination mitigation strategies employed
- [ ] Input variables explicitly defined
- [ ] Prompt versioning/tracking considered

### 💰 Cost & Performance
- [ ] Token usage optimized
- [ ] Caching strategy implemented where appropriate
- [ ] Model selection justified (cost vs capability)
- [ ] Async/parallel processing where beneficial
- [ ] Batch processing for bulk operations

### 📊 Observability
- [ ] Structured logging implemented
- [ ] Key metrics tracked (latency, cost, success rate)
- [ ] Tracing for debugging complex flows
- [ ] Callback handlers for monitoring
- [ ] Error tracking and alerting

### 🧪 Testing
- [ ] Unit tests for core logic
- [ ] Integration tests for AI components
- [ ] Edge cases handled
- [ ] Mock external API calls in tests
- [ ] Evaluation metrics defined for AI outputs

### 📖 Documentation
- [ ] README updated with usage examples
- [ ] Architecture decisions documented
- [ ] API endpoints documented (if applicable)
- [ ] Environment setup instructions provided
- [ ] Dependencies clearly listed

## Testing Evidence
<!-- Describe how you tested these changes -->
<!-- Include sample outputs, metrics, or screenshots if applicable -->

## Performance Metrics
<!-- If applicable, provide performance metrics -->
- Response time:
- Token usage:
- Cost per request:
- Accuracy/relevance score:

## Deployment Considerations
- [ ] Environment variables documented
- [ ] Secrets management strategy defined
- [ ] Scaling considerations addressed
- [ ] Monitoring setup planned
- [ ] Rollback strategy considered

## Additional Context
<!-- Add any other context about the PR here -->

---

## Reviewer Checklist
**For reviewers - ensure the following:**

### Principal Engineer Review Points
1. **Architectural Soundness**
   - Is this the right pattern for the problem?
   - Are there better alternatives?
   - Is it over-engineered or too simplistic?

2. **Framework Misuse**
   - Are chains/agents/tools used correctly?
   - Is the framework choice justified?
   - Are best practices followed?

3. **Production Readiness**
   - Will this scale?
   - Is error handling robust?
   - Are costs manageable?

4. **Security & Privacy**
   - Are API keys properly secured?
   - Is input properly validated?
   - Is sensitive data protected?

5. **Code Quality**
   - Is the code maintainable?
   - Are abstractions clean?
   - Is documentation adequate?

### Hallucination Risk Assessment
- [ ] Prompts have appropriate constraints
- [ ] Factual grounding mechanisms in place
- [ ] Citation/source tracking implemented
- [ ] Confidence scoring considered

### Specific Anti-Patterns to Check
- [ ] No infinite loops in agent execution
- [ ] No unbounded memory buffers
- [ ] No missing retry logic on API calls
- [ ] No hardcoded prompts that should be templates
- [ ] No excessive model calls (inefficient chaining)
- [ ] No missing validation on LLM outputs
- [ ] No improper tool definitions
- [ ] No missing context window management
