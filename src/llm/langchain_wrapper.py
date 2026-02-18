"""
Production-grade LangChain integration with proper error handling and monitoring.
"""

from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

from src.core.logging_config import get_logger
from src.core.retry_handler import retry_with_backoff
from src.core.cost_tracker import get_cost_tracker, TokenUsage
from src.security.input_validator import validate_input
from config import get_llm_config

logger = get_logger(__name__)


class CostTrackingCallback(BaseCallbackHandler):
    """Callback to track costs and token usage."""
    
    def __init__(self):
        self.cost_tracker = get_cost_tracker()
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Track token usage when LLM call completes."""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                model = response.llm_output.get('model_name', 'gpt-4')
                cost_estimate = self.cost_tracker.estimate_cost(
                    model=model,
                    prompt_tokens=token_usage.get('prompt_tokens', 0),
                    completion_tokens=token_usage.get('completion_tokens', 0)
                )
                self.cost_tracker.record_usage(cost_estimate)


class SafeLangChainWrapper:
    """
    Production-grade LangChain wrapper with safety controls.
    Implements proper error handling, cost tracking, and security.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        config = get_llm_config()
        
        self.model_name = model_name or config.default_model
        self.temperature = temperature or config.default_temperature
        self.max_tokens = max_tokens or config.default_max_tokens
        
        # Initialize LLM with callbacks
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            callbacks=[CostTrackingCallback()],
            request_timeout=60,
        )
        
        self.cost_tracker = get_cost_tracker()
    
    @retry_with_backoff(max_attempts=3)
    async def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        validate_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Generate completion with safety checks.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            validate_prompt: Whether to validate input for injection
        
        Returns:
            Dict with response and metadata
        """
        # Validate input
        if validate_prompt:
            validation = validate_input(prompt)
            if not validation.is_valid:
                logger.error(f"Invalid prompt detected: {validation.issues}")
                raise ValueError(f"Invalid prompt: {validation.issues}")
            prompt = validation.sanitized_input
        
        # Check cost limits
        estimated_tokens = self.cost_tracker.count_tokens(prompt, self.model_name)
        estimated_cost = self.cost_tracker.estimate_cost(
            self.model_name,
            estimated_tokens,
            self.max_tokens
        )
        
        if not self.cost_tracker.check_limits(estimated_cost.total_cost):
            raise ValueError("Request would exceed cost limits")
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # Generate completion
        try:
            response = await self.llm.agenerate([messages])
            
            result = {
                "content": response.generations[0][0].text,
                "model": self.model_name,
                "finish_reason": "completed"
            }
            
            logger.info(f"Generated completion for prompt length: {len(prompt)}")
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def create_chain_with_memory(
        self,
        prompt_template: str,
        memory_type: str = "buffer"
    ) -> LLMChain:
        """
        Create LLM chain with memory.
        
        Args:
            prompt_template: Prompt template string
            memory_type: Type of memory ("buffer" or "summary")
        
        Returns:
            LLMChain with memory
        """
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_template
        )
        
        if memory_type == "summary":
            memory = ConversationSummaryMemory(llm=self.llm)
        else:
            memory = ConversationBufferMemory()
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        return chain


class RAGChainBuilder:
    """Build RAG chains with proper error handling."""
    
    @staticmethod
    def create_qa_chain(
        llm: ChatOpenAI,
        retriever: Any,
        prompt_template: Optional[str] = None
    ):
        """
        Create a QA chain for RAG.
        
        Args:
            llm: Language model
            retriever: Vector store retriever
            prompt_template: Optional custom prompt template
        
        Returns:
            QA chain
        """
        from langchain.chains import RetrievalQA
        
        default_template = """Use the following context to answer the question.
        If you don't know the answer, say so - don't make up information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            prompt = PromptTemplate(
                template=default_template,
                input_variables=["context", "question"]
            )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
