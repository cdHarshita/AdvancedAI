"""
Example: BAD RAG Implementation (for demonstration purposes)
This intentionally shows anti-patterns that the review system will catch.

DO NOT USE THIS CODE IN PRODUCTION!
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# ❌ CRITICAL: Hardcoded API key (will be caught by security scan)
# Using obviously fake key pattern to avoid confusion with real keys
OPENAI_API_KEY = "sk-fake-example-key-for-demonstration-only-do-not-use"

class BadRAGSystem:
    """
    BAD example - intentionally shows anti-patterns.
    
    Issues this code has:
    1. Hardcoded API key
    2. No error handling
    3. Unbounded memory
    4. No chunking configuration
    5. Default retrieval settings
    6. No monitoring
    7. Missing input validation
    8. No timeout on LLM calls
    """
    
    def __init__(self):
        # ❌ Using hardcoded API key
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # ❌ Default text splitter - no configuration!
        self.text_splitter = RecursiveCharacterTextSplitter()
        
        # ❌ No timeout, no retries
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0  # ⚠️ May be too deterministic
        )
        
        # ❌ Unbounded memory - will grow forever!
        self.memory = ConversationBufferMemory()
        
    def load_documents(self, documents):
        # ❌ No error handling
        # ❌ No validation of inputs
        split_docs = self.text_splitter.split_documents(documents)
        
        # ❌ No persistence configured
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
        
        # ❌ Default retriever - no k specified, no search type
        self.retriever = self.vectorstore.as_retriever()
        
        # ❌ No callback handlers - no monitoring
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory  # ❌ Using unbounded memory
        )
    
    def query(self, question):
        # ❌ No input validation
        # ❌ No error handling
        # ❌ No retry logic
        return self.qa_chain.run(question)

# ❌ Example usage without any checks
if __name__ == "__main__":
    # This code will trigger multiple review warnings!
    rag = BadRAGSystem()
    # No documents loaded but trying to query anyway
    result = rag.query("What is RAG?")
    print(result)
