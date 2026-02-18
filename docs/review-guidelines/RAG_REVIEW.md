# RAG (Retrieval-Augmented Generation) Review Guidelines

## Overview

RAG is one of the most critical patterns in production AI systems. This guide provides comprehensive best practices for reviewing RAG implementations.

## RAG Pipeline Components

A production RAG system consists of:
1. **Document Loading** - Ingestion from various sources
2. **Text Splitting** - Chunking into appropriate sizes
3. **Embedding** - Converting text to vectors
4. **Vector Storage** - Storing and indexing embeddings
5. **Retrieval** - Finding relevant documents
6. **Generation** - LLM-based answer synthesis
7. **Evaluation** - Measuring quality

Each component requires careful review.

## 1. Document Loading

### ✅ Best Practices

```python
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    DirectoryLoader
)

class DocumentLoader:
    """Robust document loading with error handling"""
    
    def __init__(self, supported_types: List[str] = None):
        self.supported_types = supported_types or ['.pdf', '.txt', '.md']
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
        }
    
    def load_documents(self, path: str) -> List[Document]:
        """Load documents with error handling"""
        documents = []
        errors = []
        
        # Use DirectoryLoader for batch processing
        for file_type in self.supported_types:
            try:
                loader = DirectoryLoader(
                    path,
                    glob=f"**/*{file_type}",
                    loader_cls=self.loaders[file_type],
                    show_progress=True,
                    use_multithreading=True,
                    max_concurrency=4
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {file_type} documents")
            except Exception as e:
                errors.append(f"Error loading {file_type}: {e}")
                logger.error(f"Failed to load {file_type} files: {e}")
        
        if errors and not documents:
            raise Exception(f"Failed to load any documents: {errors}")
        
        return documents
```

### Review Checklist
- [ ] Error handling for missing/corrupted files
- [ ] Metadata preservation (source, page numbers, etc.)
- [ ] Character encoding handled (UTF-8, etc.)
- [ ] Large file handling (streaming, chunking)
- [ ] Progress tracking for batch loads

## 2. Text Splitting (Most Critical)

### Chunk Size Selection Guide

| Content Type | Recommended Size | Overlap | Reasoning |
|--------------|------------------|---------|-----------|
| Technical Docs | 800-1200 tokens | 20% | Needs full context for accuracy |
| Code | 500-800 tokens | 15% | Logical code blocks |
| General Text | 500-1000 tokens | 10-20% | Balance context and specificity |
| Legal/Medical | 1000-1500 tokens | 20% | Critical to maintain complete context |
| Chat/Social | 200-500 tokens | 10% | Short, self-contained messages |
| News Articles | 500-800 tokens | 15% | Paragraph-level chunks |

### ❌ Common Mistakes

```python
# Bad: Default settings without thought
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(documents)
```

```python
# Bad: Chunk size too small (loses context)
splitter = RecursiveCharacterTextSplitter(chunk_size=100)

# Bad: Chunk size too large (poor retrieval precision)
splitter = RecursiveCharacterTextSplitter(chunk_size=5000)

# Bad: No overlap (context discontinuity)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
```

### ✅ Best Practices

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_splitter(content_type: str = "general") -> RecursiveCharacterTextSplitter:
    """Create appropriate splitter for content type"""
    
    configs = {
        "general": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", ". ", " ", ""]
        },
        "code": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "separators": ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
        },
        "technical": {
            "chunk_size": 1200,
            "chunk_overlap": 240,
            "separators": ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        },
        "legal": {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "separators": ["\n\n", "\n", ". ", " "]
        }
    }
    
    config = configs.get(content_type, configs["general"])
    
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=config["separators"],
        length_function=len,  # Or use token counter
    )

# Better: Use token-based splitting for accuracy
from langchain.text_splitter import CharacterTextSplitter
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def token_length(text: str) -> int:
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # tokens, not characters
    chunk_overlap=200,
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### Advanced: Semantic Chunking

```python
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# Good: Semantic-aware chunking
semantic_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Even better: Hybrid approach
from langchain.text_splitter import MarkdownHeaderTextSplitter

# First split by structure
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# Then by size
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Two-stage splitting
structured_chunks = markdown_splitter.split_text(markdown_text)
final_chunks = size_splitter.split_documents(structured_chunks)
```

### Review Checklist
- [ ] Chunk size justified for content type
- [ ] Overlap configured (typically 10-20%)
- [ ] Appropriate separators for content structure
- [ ] Token-based splitting if using token-limited models
- [ ] Metadata preserved through splitting
- [ ] Edge cases tested (very short/long documents)

## 3. Embedding Selection

### Model Comparison

| Model | Dimensions | Max Tokens | Best For | Cost |
|-------|------------|------------|----------|------|
| OpenAI text-embedding-3-small | 1536 | 8191 | General, cost-effective | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | 8191 | High accuracy needed | $0.13/1M tokens |
| OpenAI ada-002 (legacy) | 1536 | 8191 | Legacy support | $0.10/1M tokens |
| Cohere embed-english-v3 | 1024 | 512 | English text | $0.10/1M tokens |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 256 | Local, fast | Free (self-hosted) |
| intfloat/multilingual-e5-large | 1024 | 512 | Multilingual | Free (self-hosted) |

### ✅ Best Practices

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

class EmbeddingFactory:
    """Factory for creating appropriate embeddings"""
    
    @staticmethod
    def create(
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        **kwargs
    ):
        if provider == "openai":
            return OpenAIEmbeddings(
                model=model,
                show_progress_bar=True,
                **kwargs
            )
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cuda'},  # or 'cpu'
                encode_kwargs={'normalize_embeddings': True},
                **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

# With caching for repeated queries
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

underlying_embeddings = OpenAIEmbeddings()
store = LocalFileStore("./cache/embeddings")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)
```

### Review Checklist
- [ ] Model choice justified (accuracy vs. cost vs. speed)
- [ ] Appropriate for language (multilingual if needed)
- [ ] Dimension size appropriate for use case
- [ ] Caching implemented for repeated queries
- [ ] Batch embedding for large document sets
- [ ] Error handling for API failures

## 4. Vector Store Selection

### Vector Store Comparison

| Vector DB | Best For | Max Scale | Highlights |
|-----------|----------|-----------|------------|
| **Chroma** | Prototyping, small-medium | ~1M vectors | Easy setup, local-first |
| **FAISS** | Fast local search | ~10M vectors | Facebook's library, very fast |
| **Pinecone** | Production, serverless | 100M+ vectors | Managed, reliable, costly |
| **Weaviate** | Enterprise, hybrid search | 100M+ vectors | GraphQL, filters, modules |
| **Qdrant** | High performance | 100M+ vectors | Fast, open-source, filters |
| **Milvus** | Large scale | Billions | Open-source, distributed |

### ✅ Best Practices

```python
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.embeddings import OpenAIEmbeddings

class VectorStoreFactory:
    """Create appropriate vector store"""
    
    @staticmethod
    def create_chroma(
        documents: List[Document],
        embeddings,
        persist_directory: str = "./chroma_db"
    ):
        """For development and small-medium scale"""
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # or "l2", "ip"
        )
    
    @staticmethod
    def create_faiss(
        documents: List[Document],
        embeddings
    ):
        """For local, high-performance search"""
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        # Save for later use
        vectorstore.save_local("faiss_index")
        return vectorstore
    
    @staticmethod
    def create_pinecone(
        documents: List[Document],
        embeddings,
        index_name: str,
        namespace: str = "default"
    ):
        """For production, serverless"""
        import pinecone
        
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # Match embedding dimensions
                metric="cosine",
                pods=1,
                pod_type="p1.x1"
            )
        
        return Pinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )

# With metadata for filtering
documents_with_metadata = [
    Document(
        page_content=doc.page_content,
        metadata={
            "source": doc.metadata.get("source", "unknown"),
            "date": doc.metadata.get("date", ""),
            "category": doc.metadata.get("category", "general"),
            "chunk_id": i
        }
    )
    for i, doc in enumerate(documents)
]
```

### Review Checklist
- [ ] Vector store appropriate for scale
- [ ] Persistence configured (data won't be lost)
- [ ] Metadata schema defined
- [ ] Distance metric appropriate (cosine vs. euclidean)
- [ ] Index configuration optimized
- [ ] Backup strategy for production
- [ ] Cost estimated for cloud vector DBs

## 5. Retrieval Strategy

### Basic Retrieval

❌ **Too Simple**:
```python
# Bad: Default k, no diversity
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents(query)
```

✅ **Optimized Retrieval**:
```python
# Good: Configured retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 4,  # Number of results
        "fetch_k": 20,  # Fetch more for reranking
        "lambda_mult": 0.5  # Diversity vs. relevance (0=diverse, 1=relevant)
    }
)

# Better: With metadata filtering
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.7,  # Minimum similarity score
        "filter": {
            "category": "technical",
            "date": {"$gte": "2023-01-01"}
        }
    }
)
```

### Advanced: Hybrid Retrieval

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever

class HybridRetriever:
    """Combine semantic and keyword search"""
    
    def __init__(self, documents: List[Document], embeddings):
        # Semantic retriever (vector-based)
        vectorstore = FAISS.from_documents(documents, embeddings)
        self.semantic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )
        
        # Keyword retriever (BM25)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.keyword_retriever.k = 6
        
        # Ensemble (weighted combination)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=[0.7, 0.3]  # 70% semantic, 30% keyword
        )
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve using hybrid approach"""
        return self.ensemble_retriever.get_relevant_documents(query)[:k]
```

### Re-ranking for Quality

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class ReRankingRetriever:
    """Retrieve and rerank for better relevance"""
    
    def __init__(self, base_retriever, llm):
        # Compressor that extracts relevant parts
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve and compress to relevant content"""
        return self.retriever.get_relevant_documents(query)

# Usage
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
reranking_retriever = ReRankingRetriever(base_retriever, llm)
docs = reranking_retriever.retrieve(query)  # Returns compressed, relevant docs
```

### Multi-Query Retrieval

```python
from langchain.retrievers import MultiQueryRetriever

# Generate multiple query variations for better recall
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# This will generate query variations and retrieve for each
docs = multi_query_retriever.get_relevant_documents(
    "What is the performance of RAG systems?"
)
# Internally generates queries like:
# - "How do RAG systems perform?"
# - "Performance metrics for RAG"
# - "RAG system benchmarks"
```

### Review Checklist
- [ ] Appropriate k value (typically 3-6 for generation)
- [ ] Diversity considered (MMR for varied results)
- [ ] Score thresholds to filter irrelevant docs
- [ ] Metadata filtering utilized
- [ ] Hybrid search for critical applications
- [ ] Re-ranking for quality improvement
- [ ] Multi-query for better recall

## 6. Generation (LLM Synthesis)

### Prompt Engineering for RAG

❌ **Poor Prompt**:
```python
# Bad: Vague, may hallucinate
prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
```

✅ **Strong Prompt**:
```python
from langchain.prompts import PromptTemplate

template = """You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Instructions:
1. Use only information from the context below
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Cite the source when possible
4. Be concise but complete
5. If you're uncertain, express your uncertainty

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the provided information."""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

### Chain Selection

```python
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Basic Q&A
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# With source citations
qa_with_sources = load_qa_with_sources_chain(
    llm=llm,
    chain_type="stuff"
)

# Conversational RAG
from langchain.chains import ConversationalRetrievalChain

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
)
```

### Chain Type Selection Guide

| Chain Type | When to Use | Pros | Cons |
|------------|-------------|------|------|
| **stuff** | Docs fit in context | Simple, one LLM call | Limited by context window |
| **map_reduce** | Many docs | Handles large doc sets | Multiple LLM calls (costly) |
| **refine** | Iterative improvement | Builds on previous answers | Sequential (slow) |
| **map_rerank** | Best single answer | Returns highest scored answer | Requires scoring |

### Review Checklist
- [ ] Prompt constrains to provided context
- [ ] Instruction to refuse when uncertain
- [ ] Source citation included
- [ ] Chain type appropriate for doc count
- [ ] Memory managed (bounded for conversations)
- [ ] Streaming enabled for better UX
- [ ] Error handling for LLM failures

## 7. Evaluation

### Metrics to Track

```python
from langchain.evaluation import load_evaluator

class RAGEvaluator:
    """Evaluate RAG system quality"""
    
    def __init__(self, llm):
        self.llm = llm
        self.relevance_evaluator = load_evaluator("labeled_criteria", criteria="relevance", llm=llm)
        self.correctness_evaluator = load_evaluator("labeled_criteria", criteria="correctness", llm=llm)
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        ground_truth_docs: List[str]
    ) -> dict:
        """Evaluate retrieval quality"""
        # Precision: % of retrieved docs that are relevant
        retrieved_set = set(doc.metadata.get("source", "") for doc in retrieved_docs)
        ground_truth_set = set(ground_truth_docs)
        
        if not retrieved_set:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        true_positives = len(retrieved_set & ground_truth_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_generation(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: str
    ) -> dict:
        """Evaluate generation quality"""
        # Relevance
        relevance_result = self.relevance_evaluator.evaluate_strings(
            prediction=answer,
            input=question
        )
        
        # Correctness (vs ground truth)
        correctness_result = self.correctness_evaluator.evaluate_strings(
            prediction=answer,
            reference=ground_truth,
            input=question
        )
        
        # Faithfulness (to context)
        faithfulness_prompt = f"""
        Does the answer stay faithful to the context? 
        Answer "yes" or "no" and explain.
        
        Context: {context}
        Answer: {answer}
        """
        faithfulness = self.llm.predict(faithfulness_prompt)
        
        return {
            "relevance_score": relevance_result["score"],
            "correctness_score": correctness_result["score"],
            "faithfulness": faithfulness
        }

# Usage
evaluator = RAGEvaluator(llm)

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    query="What is RAG?",
    retrieved_docs=docs,
    ground_truth_docs=["doc1.pdf", "doc2.pdf"]
)

# Evaluate generation
generation_metrics = evaluator.evaluate_generation(
    question="What is RAG?",
    answer=generated_answer,
    ground_truth=expected_answer,
    context=context
)
```

### Performance Monitoring

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGMetrics:
    """Track RAG system metrics"""
    query: str
    retrieval_time: float
    generation_time: float
    total_time: float
    num_docs_retrieved: int
    num_tokens_prompt: int
    num_tokens_completion: int
    cost_usd: float
    relevance_score: Optional[float] = None

class RAGMonitor:
    """Monitor RAG system performance"""
    
    def __init__(self):
        self.metrics: List[RAGMetrics] = []
    
    def track_query(
        self,
        query: str,
        retriever,
        chain,
        evaluator: Optional[RAGEvaluator] = None
    ) -> tuple[str, RAGMetrics]:
        """Track metrics for a single query"""
        # Retrieval
        retrieval_start = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start
        
        # Generation
        generation_start = time.time()
        result = chain.run(query)
        generation_time = time.time() - generation_start
        
        # Calculate cost (example for GPT-4)
        # Estimate tokens (rough)
        num_tokens_prompt = len(query.split()) * 1.3  # Rough estimate
        num_tokens_completion = len(result.split()) * 1.3
        cost_usd = (num_tokens_prompt / 1000 * 0.03) + (num_tokens_completion / 1000 * 0.06)
        
        # Evaluate
        relevance_score = None
        if evaluator:
            eval_result = evaluator.evaluate_generation(
                question=query,
                answer=result,
                ground_truth="",  # If available
                context="\n".join(doc.page_content for doc in docs)
            )
            relevance_score = eval_result.get("relevance_score")
        
        # Create metrics
        metrics = RAGMetrics(
            query=query,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            num_docs_retrieved=len(docs),
            num_tokens_prompt=int(num_tokens_prompt),
            num_tokens_completion=int(num_tokens_completion),
            cost_usd=cost_usd,
            relevance_score=relevance_score
        )
        
        self.metrics.append(metrics)
        return result, metrics
    
    def get_statistics(self) -> dict:
        """Get aggregate statistics"""
        if not self.metrics:
            return {}
        
        return {
            "total_queries": len(self.metrics),
            "avg_retrieval_time": sum(m.retrieval_time for m in self.metrics) / len(self.metrics),
            "avg_generation_time": sum(m.generation_time for m in self.metrics) / len(self.metrics),
            "avg_total_time": sum(m.total_time for m in self.metrics) / len(self.metrics),
            "total_cost_usd": sum(m.cost_usd for m in self.metrics),
            "avg_docs_retrieved": sum(m.num_docs_retrieved for m in self.metrics) / len(self.metrics),
        }
```

## 8. Common RAG Anti-Patterns

### ❌ Anti-Pattern 1: No Chunk Overlap
**Problem**: Context lost at chunk boundaries
**Fix**: Add 10-20% overlap

### ❌ Anti-Pattern 2: Retrieving Too Many Docs
**Problem**: Irrelevant info confuses LLM, increases cost
**Fix**: Limit to 3-6 most relevant docs

### ❌ Anti-Pattern 3: No Metadata
**Problem**: Can't filter by source, date, category
**Fix**: Add rich metadata during ingestion

### ❌ Anti-Pattern 4: Default Embeddings
**Problem**: May not be optimal for domain
**Fix**: Benchmark and choose appropriate model

### ❌ Anti-Pattern 5: No Evaluation
**Problem**: Quality degradation goes unnoticed
**Fix**: Implement continuous evaluation

### ❌ Anti-Pattern 6: Ignoring Hybrid Search
**Problem**: Misses exact keyword matches
**Fix**: Combine semantic + keyword search

### ❌ Anti-Pattern 7: No Re-ranking
**Problem**: First-pass retrieval may be noisy
**Fix**: Add re-ranking layer

### ❌ Anti-Pattern 8: Weak Prompts
**Problem**: LLM hallucinates despite context
**Fix**: Use strict prompts that constrain to context

## RAG Review Checklist

### Critical
- [ ] Chunk size justified and appropriate
- [ ] Chunk overlap configured (10-20%)
- [ ] Embeddings model choice documented
- [ ] Vector store appropriate for scale
- [ ] Retrieval k value reasonable (3-6)
- [ ] Prompt constrains to provided context
- [ ] Source citations included
- [ ] Error handling comprehensive

### Important
- [ ] Metadata schema defined and utilized
- [ ] Hybrid search for critical applications
- [ ] Re-ranking implemented
- [ ] Token-based splitting (not character)
- [ ] Distance metric appropriate
- [ ] Chain type matches use case
- [ ] Memory bounded (for conversational)
- [ ] Evaluation metrics tracked

### Nice to Have
- [ ] Semantic chunking
- [ ] Multi-query retrieval
- [ ] Caching for repeated queries
- [ ] Streaming for better UX
- [ ] A/B testing different configurations
- [ ] Continuous evaluation pipeline
- [ ] Cost monitoring and alerts

## Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)
- [Pinecone RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)

---

**Remember**: RAG is not one-size-fits-all. The optimal configuration depends on your specific use case, data, and requirements. Always benchmark and iterate.
