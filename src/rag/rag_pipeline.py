"""
Production-grade RAG implementation with efficiency optimizations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader, PDFLoader
from langchain.schema import Document

from src.core.logging_config import get_logger
from src.core.retry_handler import retry_with_backoff
from config import get_settings

logger = get_logger(__name__)


class EfficientRAGPipeline:
    """
    Efficient RAG pipeline with chunking optimization and caching.
    Implements best practices for retrieval quality and performance.
    """
    
    def __init__(
        self,
        vector_store_type: str = "chroma",
        embedding_model: str = "openai",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.vector_store_type = vector_store_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        else:
            # Use local model for cost efficiency
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
    
    async def load_documents(
        self,
        directory: str,
        file_types: List[str] = ["txt", "pdf", "md"]
    ) -> List[Document]:
        """
        Load documents from directory.
        
        Args:
            directory: Directory path
            file_types: List of file extensions to load
        
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_type in file_types:
            try:
                if file_type == "pdf":
                    loader = DirectoryLoader(
                        directory,
                        glob=f"**/*.{file_type}",
                        loader_cls=PDFLoader
                    )
                else:
                    loader = DirectoryLoader(
                        directory,
                        glob=f"**/*.{file_type}",
                        loader_cls=TextLoader
                    )
                
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {file_type} documents")
                
            except Exception as e:
                logger.error(f"Error loading {file_type} files: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into optimized chunks.
        
        Args:
            documents: List of documents
        
        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    @retry_with_backoff(max_attempts=3)
    async def build_vector_store(
        self,
        documents: List[Document],
        persist_directory: Optional[str] = None
    ):
        """
        Build vector store from documents.
        
        Args:
            documents: List of documents
            persist_directory: Directory to persist vector store
        """
        if self.vector_store_type == "chroma":
            persist_dir = persist_directory or "./chroma_db"
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            logger.info(f"Created Chroma vector store with {len(documents)} documents")
            
        elif self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            if persist_directory:
                self.vector_store.save_local(persist_directory)
            logger.info(f"Created FAISS vector store with {len(documents)} documents")
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: float = 0.7
    ):
        """
        Get retriever with optimized parameters.
        
        Args:
            search_type: Type of search ("similarity" or "mmr")
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        
        Returns:
            Retriever object
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        if search_type == "mmr":
            # Maximum Marginal Relevance for diversity
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2}
            )
        else:
            # Similarity search with threshold
            return self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": k}
            )
    
    async def query(
        self,
        query: str,
        k: int = 4,
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store.
        
        Args:
            query: Query string
            k: Number of results
            return_scores: Whether to return similarity scores
        
        Returns:
            List of results with metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        if return_scores:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        else:
            results = self.vector_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]


class HybridSearch:
    """
    Hybrid search combining dense and sparse retrieval.
    Improves retrieval quality for RAG applications.
    """
    
    def __init__(self, vector_store, bm25_retriever=None):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
    
    async def search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Perform hybrid search.
        
        Args:
            query: Query string
            k: Number of results
            alpha: Weight for dense retrieval (1-alpha for sparse)
        
        Returns:
            List of documents
        """
        # Dense retrieval
        dense_results = self.vector_store.similarity_search(query, k=k)
        
        # If BM25 available, combine results
        if self.bm25_retriever:
            sparse_results = self.bm25_retriever.get_relevant_documents(query)
            
            # Simple weighted combination
            # In production, use reciprocal rank fusion
            combined = dense_results[:int(k * alpha)] + sparse_results[:int(k * (1 - alpha))]
            return combined[:k]
        
        return dense_results
