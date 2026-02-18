"""
Example: Good RAG Implementation
This demonstrates best practices for building a production-grade RAG system.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import Document

# Load environment variables (never hardcode!)
load_dotenv()


class ProductionRAGSystem:
    """
    Production-ready RAG system with best practices.
    
    Features:
    - Environment-based configuration
    - Proper chunking strategy
    - Bounded retrieval
    - Monitored LLM calls
    - Error handling
    - Source citations
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
        temperature: float = 0.1
    ):
        """
        Initialize RAG system with production configurations.
        
        Args:
            collection_name: Name for the vector store collection
            chunk_size: Size of text chunks (in characters by default)
            chunk_overlap: Overlap between chunks (for context continuity)
            k: Number of documents to retrieve
            temperature: LLM temperature (lower for factual responses)
            
        Note:
            For token-based chunking, provide a custom length_function.
        """
        # Validate API key exists
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Initialize components with proper configuration
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            show_progress_bar=True
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=temperature,
            model="gpt-4",  # Production-grade model
            request_timeout=30,  # Timeout for safety
            max_retries=3  # Retry on failures
        )
        
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.k = k
        
        # Monitoring callback
        self.callback = StdOutCallbackHandler()
    
    def load_documents(self, documents: List[Document]) -> None:
        """
        Load and index documents with proper error handling.
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("No documents provided")
        
        try:
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            
            # Create vector store with persistence
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db",
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # Configure retriever with MMR for diversity
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": self.k * 3,  # Fetch more for reranking
                    "lambda_mult": 0.7  # Balance relevance (1.0) vs diversity (0.0)
                }
            )
            
            # Create QA chain
            self._create_qa_chain()
            
            print("Documents loaded and indexed successfully")
            
        except Exception as e:
            raise Exception(f"Error loading documents: {e}")
    
    def _create_qa_chain(self) -> None:
        """Create QA chain with proper prompt engineering."""
        # Prompt that mitigates hallucination
        prompt_template = """You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Instructions:
1. Use only information from the context below
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Cite the source when possible
4. Be concise but complete
5. If uncertain, express your uncertainty

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the provided information."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Appropriate for bounded retrieval
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": False
            },
            callbacks=[self.callback]
        )
    
    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> dict:
        """
        Query the RAG system with proper error handling.
        
        Args:
            question: User question
            return_sources: Whether to return source documents
            
        Returns:
            dict with 'answer' and optionally 'sources'
        """
        if not self.qa_chain:
            raise ValueError("System not initialized. Call load_documents first.")
        
        # Input validation
        if not question or len(question.strip()) == 0:
            raise ValueError("Question cannot be empty")
        
        if len(question) > 2000:
            raise ValueError("Question too long (max 2000 characters)")
        
        try:
            # Execute query
            result = self.qa_chain({"query": question})
            
            response = {
                "answer": result["result"]
            }
            
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",  # Truncate for display
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            
            return response
            
        except Exception as e:
            # Log the error (in production, use proper logging)
            print(f"Error during query: {e}")
            return {
                "answer": "I'm experiencing technical difficulties. Please try again.",
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain-docs", "page": 1}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.",
            metadata={"source": "rag-paper", "page": 1}
        ),
    ]
    
    # Initialize system
    rag = ProductionRAGSystem(
        chunk_size=1000,
        chunk_overlap=200,
        k=4,
        temperature=0.1
    )
    
    # Load documents
    rag.load_documents(documents)
    
    # Query
    result = rag.query("What is RAG?")
    print("\nAnswer:", result["answer"])
    print("\nSources:", len(result.get("sources", [])))
