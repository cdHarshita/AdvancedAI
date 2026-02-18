# Senior AI Systems Architect Review Guidelines

## Overview

This document provides comprehensive guidelines for reviewing code in the Advanced AI Systems Lab. As a reviewer, you are expected to think like a **Principal Engineer** building an **enterprise-grade AI platform**.

## Review Philosophy

### Core Principles
1. **Be Critical but Constructive**: Point out issues clearly, but always suggest improvements
2. **Explain the Why**: Don't just say something is wrong—explain why and what the correct pattern is
3. **Think Production-First**: Every review should consider scalability, cost, monitoring, and maintainability
4. **Prefer Simplicity**: Flag over-engineering; simpler solutions are often better
5. **Security-First**: Always check for security vulnerabilities and data protection issues

### What Makes a Good Review
- Specific, actionable feedback
- Code examples for suggested improvements
- References to official documentation or best practices
- Performance and cost implications explained
- Alternative approaches when current design is suboptimal

## Review Checklist by Category

## 1. Architectural Correctness

### Questions to Ask
- Is this the right pattern for the problem?
- Is the separation of concerns clear?
- Are abstractions at the right level?
- Is the design extensible and maintainable?
- Are there better alternatives?

### Common Issues
❌ **Tight Coupling**: Direct dependencies between components that should be loosely coupled
```python
# Bad: Direct dependency
class RAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")  # Hardcoded
```

✅ **Proper Abstraction**:
```python
# Good: Dependency injection
class RAGSystem:
    def __init__(self, llm: BaseLLM):
        self.llm = llm  # Injected, testable
```

❌ **Mixed Concerns**: Business logic mixed with infrastructure code
```python
# Bad: Everything in one place
def answer_question(question: str):
    # Setup
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(embedding_function=embeddings)
    # Business logic
    docs = vectorstore.similarity_search(question)
    # LLM call
    llm = ChatOpenAI()
    return llm.predict(f"Answer based on: {docs}")
```

✅ **Separated Concerns**:
```python
# Good: Clear separation
class DocumentRetriever:
    def retrieve(self, query: str) -> List[Document]: ...

class QuestionAnswerer:
    def __init__(self, retriever: DocumentRetriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm
    
    def answer(self, question: str) -> str: ...
```

## 2. LangChain Best Practices

### Chain Selection
- **Simple Q&A**: Use `LLMChain` or `LCEL` (LangChain Expression Language)
- **Document Q&A**: Use `RetrievalQA` or `ConversationalRetrievalChain`
- **Multi-step reasoning**: Use `SequentialChain` or agents
- **Conversational**: Use `ConversationChain` with appropriate memory

### Common Anti-Patterns

❌ **No Memory Limits**:
```python
# Bad: Unbounded memory
memory = ConversationBufferMemory()
```

✅ **Bounded Memory**:
```python
# Good: Token-limited memory
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
# Or use window-based
memory = ConversationBufferWindowMemory(k=5)
```

❌ **Missing Callbacks**:
```python
# Bad: No observability
chain = LLMChain(llm=llm, prompt=prompt)
```

✅ **With Callbacks**:
```python
# Good: Monitoring enabled
from langchain.callbacks import StdOutCallbackHandler

chain = LLMChain(
    llm=llm, 
    prompt=prompt,
    callbacks=[StdOutCallbackHandler()]
)
```

❌ **Inefficient Retrieval**:
```python
# Bad: Too many documents, no reranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
```

✅ **Optimized Retrieval**:
```python
# Good: Reasonable k, with MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)
```

### Agent Guidelines

❌ **No Guardrails**:
```python
# Bad: Infinite loop risk
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

✅ **With Guardrails**:
```python
# Good: Bounded execution
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5,
    max_execution_time=30,
    early_stopping_method="generate"
)
```

## 3. RAG Implementation Review

### Chunking Strategy

**Chunk Size Selection**:
- **Code**: 500-1000 tokens (logical blocks)
- **General Text**: 500-1500 tokens
- **Legal/Medical**: 1000-2000 tokens (more context needed)
- **Chat/Social**: 200-500 tokens

❌ **Default Chunking**:
```python
# Bad: No configuration
splitter = RecursiveCharacterTextSplitter()
```

✅ **Configured Chunking**:
```python
# Good: Explicit, justified parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Based on content type
    chunk_overlap=200,  # 20% overlap for context
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### Embedding Selection

| Use Case | Recommended Model | Reasoning |
|----------|------------------|-----------|
| General Purpose | OpenAI text-embedding-3-small | Good balance of cost/performance |
| High Accuracy | OpenAI text-embedding-3-large | Best quality, higher cost |
| Local/Privacy | sentence-transformers/all-MiniLM-L6-v2 | Fast, free, local |
| Multilingual | intfloat/multilingual-e5-large | Supports 100+ languages |
| Code | OpenAI text-embedding-3-small | Code-aware embeddings |

### Vector Store Selection

| Scale | Recommended Store | Reasoning |
|-------|------------------|-----------|
| Prototyping | Chroma (local) | Easy setup, no infrastructure |
| Small-Medium (<1M vectors) | Chroma or FAISS | Simple, cost-effective |
| Large (1M-10M) | Pinecone or Weaviate | Managed, scalable |
| Enterprise (>10M) | Pinecone, Weaviate, or Qdrant | Production-grade, high performance |

### Retrieval Strategy

❌ **Basic Similarity Only**:
```python
# Bad: May return redundant results
docs = vectorstore.similarity_search(query, k=4)
```

✅ **MMR for Diversity**:
```python
# Good: Balanced relevance and diversity
docs = vectorstore.max_marginal_relevance_search(
    query, 
    k=4,
    fetch_k=20  # Fetch more, rerank
)
```

✅ **With Metadata Filtering**:
```python
# Best: Filtered retrieval
docs = vectorstore.similarity_search(
    query,
    k=4,
    filter={"source": "trusted_docs", "year": {"$gte": 2023}}
)
```

## 4. LlamaIndex Patterns

### Index Selection

| Use Case | Index Type | When to Use |
|----------|-----------|-------------|
| Simple Q&A | VectorStoreIndex | Most common, good default |
| Structured Data | SQLTableIndex | Database queries |
| Hierarchical Docs | TreeIndex | Document hierarchies |
| Lists/Summaries | ListIndex | Exhaustive retrieval needed |
| Keywords | KeywordTableIndex | Keyword-based lookup |

### ServiceContext Configuration

❌ **Default Settings**:
```python
# Bad: No configuration
index = VectorStoreIndex.from_documents(documents)
```

✅ **Explicit Configuration**:
```python
# Good: Production-ready setup
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding

llm = OpenAI(model="gpt-4", temperature=0.1)
embed_model = OpenAIEmbedding()

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=1024,
    chunk_overlap=20,
    context_window=4096,
    num_output=512
)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)
```

## 5. AutoGen Multi-Agent Systems

### Agent Design

❌ **Unlimited Auto-Reply**:
```python
# Bad: Infinite conversation risk
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config
)
```

✅ **Bounded Conversation**:
```python
# Good: Limited iterations
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    max_consecutive_auto_reply=10,
    human_input_mode="NEVER"  # or "TERMINATE" or "ALWAYS"
)
```

### Code Execution Safety

❌ **Unrestricted Execution**:
```python
# Bad: Security risk
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"}
)
```

✅ **Sandboxed Execution**:
```python
# Good: Docker isolation
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,
        "timeout": 60,
        "last_n_messages": 3
    }
)
```

## 6. CrewAI Orchestration

### Task Definition

❌ **Vague Tasks**:
```python
# Bad: Unclear task
task = Task(
    description="Analyze the data",
    agent=analyst
)
```

✅ **Well-Defined Tasks**:
```python
# Good: Clear, specific, measurable
task = Task(
    description="""
    Analyze the Q4 sales data and produce a report including:
    1. Total revenue by region
    2. Top 5 products by sales volume
    3. Year-over-year growth percentage
    4. Key trends and anomalies
    
    Output format: Structured JSON
    """,
    agent=analyst,
    expected_output="JSON with keys: revenue, top_products, yoy_growth, trends"
)
```

## 7. Security Best Practices

### API Key Management

❌ **Hardcoded Secrets**:
```python
# CRITICAL: Never do this
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxx"
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
```

✅ **Environment Variables**:
```python
# Good: From environment
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
```

✅ **Secrets Management (Production)**:
```python
# Best: Secrets manager
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

api_key = get_secret("prod/openai/api_key")
```

### Input Validation

❌ **No Validation**:
```python
# Bad: SQL injection risk
def query_database(user_input: str):
    query = f"SELECT * FROM docs WHERE title = '{user_input}'"
    return execute_query(query)
```

✅ **Validated Input**:
```python
# Good: Parameterized queries
def query_database(user_input: str):
    # Validate input
    if not user_input or len(user_input) > 200:
        raise ValueError("Invalid input")
    
    # Use parameterized query
    query = "SELECT * FROM docs WHERE title = ?"
    return execute_query(query, (user_input,))
```

### Prompt Injection Protection

❌ **Direct User Input**:
```python
# Bad: Prompt injection risk
prompt = f"Answer this question: {user_question}"
```

✅ **Template with Validation**:
```python
# Good: Structured template
from langchain.prompts import PromptTemplate

template = """You are a helpful assistant. Answer the following question based only on the provided context.

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Validate and sanitize
def sanitize_input(text: str) -> str:
    # Remove system prompts, special tokens
    forbidden = ["ignore previous", "system:", "assistant:"]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in forbidden):
        raise ValueError("Potentially malicious input detected")
    return text[:500]  # Limit length
```

## 8. Prompt Engineering

### Hallucination Mitigation

❌ **Unbounded Response**:
```python
# Bad: May hallucinate
prompt = "Tell me about {topic}"
```

✅ **Grounded Response**:
```python
# Good: Constrained to provided context
prompt = """Based ONLY on the following context, answer the question.
If the answer is not in the context, respond with "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer: """
```

### Few-Shot Examples

❌ **Zero-Shot for Complex Tasks**:
```python
# Bad: May produce inconsistent format
prompt = "Extract entities from: {text}"
```

✅ **Few-Shot for Consistency**:
```python
# Good: Examples guide format
prompt = """Extract person names, organizations, and locations.

Example 1:
Text: "John works at Google in California."
Output: {{"people": ["John"], "orgs": ["Google"], "locations": ["California"]}}

Example 2:
Text: "Microsoft CEO Satya Nadella visited Seattle."
Output: {{"people": ["Satya Nadella"], "orgs": ["Microsoft"], "locations": ["Seattle"]}}

Now extract from:
Text: {text}
Output: """
```

## 9. Cost & Performance Optimization

### Token Usage

❌ **Inefficient Prompts**:
```python
# Bad: Verbose, costly
prompt = """I would like you to please analyze the following document 
and provide me with a comprehensive summary that includes all the 
main points and key takeaways. Please be thorough and detailed..."""
```

✅ **Concise Prompts**:
```python
# Good: Clear and brief
prompt = """Summarize the key points from this document:

{document}

Summary:"""
```

### Caching Strategy

❌ **No Caching**:
```python
# Bad: Repeated embeddings
for query in queries:
    embedding = embed_model.embed_query(query)
    results = vectorstore.similarity_search_by_vector(embedding)
```

✅ **With Caching**:
```python
# Good: Cache frequent queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embed_model.embed_query(text)

for query in queries:
    embedding = get_embedding(query)
    results = vectorstore.similarity_search_by_vector(embedding)
```

### Batch Processing

❌ **Sequential Processing**:
```python
# Bad: Slow for many documents
for doc in documents:
    embedding = embed_model.embed_query(doc.page_content)
    vectorstore.add_documents([doc])
```

✅ **Batch Processing**:
```python
# Good: Much faster
embeddings = embed_model.embed_documents(
    [doc.page_content for doc in documents]
)
vectorstore.add_documents(documents)
```

## 10. Error Handling & Resilience

### Retry Logic

❌ **No Retry**:
```python
# Bad: Fails on transient errors
response = llm.predict(prompt)
```

✅ **Exponential Backoff**:
```python
# Good: Resilient to transient failures
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_llm(prompt: str) -> str:
    return llm.predict(prompt)
```

### Graceful Degradation

❌ **Hard Failure**:
```python
# Bad: Crashes on error
def get_answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    return llm.predict(f"Answer based on: {docs}")
```

✅ **Fallback Strategy**:
```python
# Good: Degrades gracefully
def get_answer(question: str) -> str:
    try:
        docs = retriever.get_relevant_documents(question)
        if docs:
            return llm.predict(f"Answer based on: {docs}")
        else:
            logger.warning("No relevant documents found")
            return "I don't have enough information to answer."
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return "I'm experiencing technical difficulties. Please try again."
```

## 11. Monitoring & Observability

### Structured Logging

❌ **Print Statements**:
```python
# Bad: Unstructured, hard to search
print(f"Processing query: {query}")
```

✅ **Structured Logging**:
```python
# Good: Searchable, filterable
import logging
import json

logger = logging.getLogger(__name__)

logger.info(
    "query_processing",
    extra={
        "query": query,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id
    }
)
```

### Metrics Tracking

✅ **Track Key Metrics**:
```python
from prometheus_client import Counter, Histogram
import time

query_count = Counter('rag_queries_total', 'Total RAG queries')
query_latency = Histogram('rag_query_latency_seconds', 'Query latency')
query_cost = Counter('rag_query_cost_usd', 'Total cost in USD')

def process_query(query: str):
    start_time = time.time()
    query_count.inc()
    
    try:
        result = rag_chain.run(query)
        
        # Track metrics
        latency = time.time() - start_time
        query_latency.observe(latency)
        
        # Estimate cost (rough)
        tokens_used = estimate_tokens(query, result)
        cost = calculate_cost(tokens_used)
        query_cost.inc(cost)
        
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
```

## 12. Testing AI Components

### Unit Testing

✅ **Mock External Calls**:
```python
# Good: Test logic without API calls
from unittest.mock import Mock, patch

def test_rag_chain():
    # Mock LLM and retriever
    mock_llm = Mock()
    mock_llm.predict.return_value = "Mocked answer"
    
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content="Context")
    ]
    
    # Test your logic
    chain = RAGChain(llm=mock_llm, retriever=mock_retriever)
    result = chain.run("test question")
    
    assert result == "Mocked answer"
    mock_llm.predict.assert_called_once()
```

### Evaluation Metrics

✅ **Track AI Quality**:
```python
from langchain.evaluation import load_evaluator

# Relevance
relevance_evaluator = load_evaluator("relevance")

# Evaluate
eval_result = relevance_evaluator.evaluate_strings(
    prediction=answer,
    reference=ground_truth,
    input=question
)

# Track over time
metrics = {
    "accuracy": eval_result["score"],
    "latency_ms": response_time * 1000,
    "cost_usd": estimated_cost
}
```

## Review Process

### Step-by-Step Review

1. **First Pass - High Level**
   - Understand the purpose and approach
   - Verify architectural soundness
   - Check if framework choice is appropriate

2. **Second Pass - Code Quality**
   - Review implementation details
   - Check for anti-patterns
   - Verify error handling

3. **Third Pass - Production Readiness**
   - Security review
   - Performance considerations
   - Monitoring and observability
   - Documentation quality

4. **Feedback Delivery**
   - Group related issues
   - Provide code examples
   - Explain rationale
   - Suggest specific improvements

### Review Comments Template

```markdown
## 🔴 Critical Issues
- **Security**: [Specific issue with code reference]
  - Why: [Explanation]
  - Fix: [Code example]

## 🟡 Warnings
- **Performance**: [Issue]
  - Impact: [Explain impact]
  - Suggestion: [Alternative approach]

## 💡 Suggestions
- **Best Practice**: [Improvement]
  - Benefit: [Why this is better]
  - Example: [Code snippet]
```

## Framework-Specific Gotchas

### LangChain
- ⚠️ `ConversationBufferMemory` grows unbounded—use `ConversationTokenBufferMemory`
- ⚠️ Default retrievers return too many docs—configure `k` appropriately
- ⚠️ Chains don't auto-retry—add retry logic
- ⚠️ Missing callbacks—no visibility into token usage

### LlamaIndex
- ⚠️ Default chunk size may not fit your use case—always configure
- ⚠️ `GPTVectorStoreIndex` is deprecated—use `VectorStoreIndex`
- ⚠️ Query engines don't stream by default—enable for better UX
- ⚠️ Missing ServiceContext optimization—configure for production

### AutoGen
- ⚠️ Infinite conversation loops—set `max_consecutive_auto_reply`
- ⚠️ Code execution without Docker—security risk
- ⚠️ No cost limits—can run up bills quickly
- ⚠️ Missing human-in-the-loop—add for critical decisions

### CrewAI
- ⚠️ Vague task descriptions—leads to poor results
- ⚠️ Missing output validation—parse and verify outputs
- ⚠️ No task dependencies—define execution order
- ⚠️ Unconstrained agents—add role-specific limits

## Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [AutoGen Docs](https://microsoft.github.io/autogen/)
- [CrewAI Docs](https://docs.crewai.com/)
- [Semantic Kernel Docs](https://learn.microsoft.com/en-us/semantic-kernel/)

### Best Practices Guides
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [LangChain Production](https://python.langchain.com/docs/guides/productionization/)

---

**Remember**: The goal is to build production-grade AI systems, not just working prototypes. Every review should elevate code quality toward enterprise standards.
