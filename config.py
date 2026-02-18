"""
Configuration management for AI Systems Lab.
Implements proper separation of concerns with environment-specific configs.
"""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import os


class SecurityConfig(BaseSettings):
    """Security configuration with strict validation."""
    
    secret_key: str = Field(..., min_length=32)
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class LLMConfig(BaseSettings):
    """LLM configuration with cost and safety controls."""
    
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    
    default_model: str = Field(default="gpt-4-turbo-preview")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=2000, ge=1, le=128000)
    
    # Cost controls
    max_tokens_per_request: int = Field(default=4000, ge=1)
    max_cost_per_request: float = Field(default=0.50, ge=0.0)
    
    @validator("default_temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class RetryConfig(BaseSettings):
    """Retry configuration with exponential backoff."""
    
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_factor: int = Field(default=2, ge=1)
    retry_min_wait: int = Field(default=1, ge=0)
    retry_max_wait: int = Field(default=60, ge=1)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/ai_systems.log")
    log_format: str = Field(default="json")
    log_rotation: str = Field(default="midnight")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    
    rate_limit_per_minute: int = Field(default=60, ge=1)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""
    
    vector_store_type: str = Field(default="chroma")
    vector_store_path: str = Field(default="./chroma_db")
    
    @validator("vector_store_type")
    def validate_vector_store_type(cls, v):
        allowed_types = ["chroma", "faiss", "pinecone"]
        if v not in allowed_types:
            raise ValueError(f"Vector store type must be one of {allowed_types}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class APIConfig(BaseSettings):
    """API server configuration."""
    
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=4, ge=1, le=16)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""
    
    environment: str = Field(default="development")
    database_url: str = Field(default="sqlite:///./ai_systems.db")
    
    # Sub-configurations
    security: SecurityConfig = SecurityConfig()
    llm: LLMConfig = LLMConfig()
    retry: RetryConfig = RetryConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    api: APIConfig = APIConfig()
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_settings().llm


def get_retry_config() -> RetryConfig:
    """Get retry configuration."""
    return get_settings().retry


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_settings().monitoring
