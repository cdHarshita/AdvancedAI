# Review Guidelines Index

## 📚 Overview

This directory contains comprehensive guidelines for reviewing AI systems code. These guides are designed for **Principal Engineer-level reviews** of production AI systems.

## 📖 Available Guides

### 1. [Architectural Review Guide](ARCHITECTURAL_REVIEW.md)
**Purpose**: Comprehensive best practices for AI framework usage

**Covers**:
- LangChain patterns and anti-patterns
- LlamaIndex configuration
- AutoGen multi-agent systems
- CrewAI task orchestration
- Semantic Kernel production patterns
- Common anti-patterns across frameworks
- Cost optimization
- Monitoring and observability

**When to use**: For all code reviews involving AI frameworks

**Key sections**:
- Framework-specific best practices
- Prompt engineering patterns
- Error handling and resilience
- Testing AI components
- Performance optimization

---

### 2. [Security Review Guide](SECURITY_REVIEW.md)
**Purpose**: Security best practices for LLM-powered applications

**Covers**:
- API key and secrets management
- Prompt injection protection
- Input/output validation
- PII detection and handling
- Rate limiting and cost controls
- OWASP LLM Top 10
- Security testing

**When to use**: ALWAYS - every PR must pass security review

**Critical checks**:
- ❌ No hardcoded secrets
- ❌ No SQL injection vulnerabilities
- ❌ No path traversal risks
- ✅ Environment-based configuration
- ✅ Input validation implemented
- ✅ Rate limiting in place

---

### 3. [RAG Review Guide](RAG_REVIEW.md)
**Purpose**: Comprehensive RAG implementation patterns

**Covers**:
- Chunking strategies by content type
- Embedding model selection
- Vector store comparison
- Retrieval optimization (MMR, hybrid search, re-ranking)
- Generation best practices
- Evaluation metrics
- Common RAG anti-patterns

**When to use**: For any RAG implementation

**Key topics**:
- Chunk size selection (500-1500 tokens depending on type)
- Chunk overlap (10-20%)
- Retrieval k values (typically 3-6)
- Prompt engineering for RAG
- Hallucination mitigation
- Performance monitoring

---

## 🎯 Quick Reference

### Security (Critical - Always Check)
```python
# ❌ NEVER
api_key = "sk-proj-xxxxx"

# ✅ ALWAYS
api_key = os.getenv("OPENAI_API_KEY")
```

### Memory Management (Important)
```python
# ❌ Unbounded
memory = ConversationBufferMemory()

# ✅ Bounded
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

### Agent Guardrails (Important)
```python
# ❌ No limits
agent = initialize_agent(tools, llm)

# ✅ With limits
agent = initialize_agent(
    tools, llm,
    max_iterations=5,
    max_execution_time=30
)
```

### RAG Chunking (Important)
```python
# ❌ Default
splitter = RecursiveCharacterTextSplitter()

# ✅ Configured
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

## 🔍 Review Process

### 1. First Pass - Security & Critical Issues
- [ ] No hardcoded secrets
- [ ] No security vulnerabilities
- [ ] Proper error handling
- [ ] Input validation

**Time**: 5-10 minutes  
**Block PR if**: Any critical security issue found

### 2. Second Pass - Architecture & Patterns
- [ ] Framework choice appropriate
- [ ] Design patterns correct
- [ ] No anti-patterns
- [ ] Proper abstractions

**Time**: 10-15 minutes  
**Request changes if**: Fundamental architectural issues

### 3. Third Pass - Best Practices & Optimization
- [ ] Performance optimized
- [ ] Cost-effective
- [ ] Monitoring in place
- [ ] Tests included
- [ ] Documentation updated

**Time**: 10-15 minutes  
**Suggest improvements**: For optimization opportunities

### Total Review Time
**30-40 minutes** for thorough principal-level review

## 📊 Review Checklist

Use this for every review:

### Critical (Must Fix)
- [ ] No hardcoded secrets
- [ ] No security vulnerabilities
- [ ] Agents have max_iterations
- [ ] Memory is bounded
- [ ] Error handling exists
- [ ] Input validation implemented

### Important (Should Fix)
- [ ] Retry logic for API calls
- [ ] Monitoring/logging added
- [ ] RAG chunking configured
- [ ] Prompt injection protection
- [ ] Tests included
- [ ] Documentation updated

### Nice to Have
- [ ] Caching implemented
- [ ] Hybrid retrieval used
- [ ] Re-ranking for quality
- [ ] Streaming for UX
- [ ] Metrics tracked

## 🏗️ Framework-Specific Quick Checks

### LangChain
- [ ] Callbacks for monitoring
- [ ] Memory bounded (TokenBufferMemory, not BufferMemory)
- [ ] Appropriate chain type
- [ ] k value configured (retrieval)
- [ ] Retry logic present

### LlamaIndex
- [ ] ServiceContext configured
- [ ] Chunk size set explicitly
- [ ] Index type appropriate
- [ ] Query engine optimized

### AutoGen
- [ ] max_consecutive_auto_reply set
- [ ] Code execution sandboxed
- [ ] Termination conditions clear
- [ ] Cost limits in place

### CrewAI
- [ ] Tasks specific and measurable
- [ ] Agent roles clearly defined
- [ ] Expected outputs specified
- [ ] Tools properly integrated

## 📈 Metrics to Track

### Code Quality
- Lines of code changed
- Complexity score
- Test coverage
- Documentation coverage

### Security
- Hardcoded secrets: 0
- Critical vulnerabilities: 0
- Input validation: 100%

### Performance
- API call latency
- Token usage per query
- Cost per query
- Cache hit rate

### AI Quality
- Relevance score
- Hallucination rate
- Source citation rate
- User satisfaction

## 🎓 Training Resources

### For Reviewers
1. Read all three guides thoroughly
2. Review the examples in `/examples`
3. Practice on test PRs
4. Get familiar with OWASP LLM Top 10
5. Understand each framework's patterns

### For Contributors
1. Read [ARCHITECTURAL_REVIEW.md](ARCHITECTURAL_REVIEW.md)
2. Read [SECURITY_REVIEW.md](SECURITY_REVIEW.md)
3. Read [RAG_REVIEW.md](RAG_REVIEW.md) (if applicable)
4. Review `/examples/good_rag_example.py`
5. Understand what NOT to do from `/examples/bad_rag_example.py`

## 🔗 External Resources

### Official Docs
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://docs.crewai.com/)
- [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)

### Security
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management](https://www.nist.gov/itl/ai-risk-management-framework)
- [OpenAI Safety](https://platform.openai.com/docs/guides/safety-best-practices)

### Best Practices
- [LangChain Production](https://python.langchain.com/docs/guides/productionization/)
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 🤝 Contributing to Guidelines

These guidelines are living documents. To contribute:

1. Open an issue with `documentation` label
2. Describe the improvement or addition
3. Submit a PR with changes
4. Get review from maintainers

**Examples of good contributions**:
- New framework patterns discovered
- Additional anti-patterns identified
- Better code examples
- Clarifications on existing content
- New security vulnerabilities

---

## 💬 Questions?

If you have questions about:
- **Applying guidelines**: Comment on the PR
- **Guideline content**: Open an issue
- **Security concerns**: Email privately
- **Framework patterns**: Reference official docs first

---

**Remember**: The goal is **production-grade code**, not just working code. Review accordingly.
