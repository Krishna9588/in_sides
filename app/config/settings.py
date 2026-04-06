"""
Configuration settings for Founder Intelligence System
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    APP_NAME: str = "Founder Intelligence System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Database Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    
    # AI/ML Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Local Models Configuration
    SENTENCE_MODEL: str = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CLASSIFICATION_MODEL: str = os.getenv("CLASSIFICATION_MODEL", "distilbert-base-uncased")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "microsoft/DialoGPT-medium")
    
    # Apify Configuration
    APIFY_TOKEN: str = os.getenv("APIFY_TOKEN")
    
    # Agent Configuration
    AGENT_BATCH_SIZE: int = int(os.getenv("AGENT_BATCH_SIZE", "100"))
    AGENT_MAX_RETRIES: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    
    # Processing Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MIN_CLUSTER_SIZE: int = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
    MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", "10"))
    
    # LLM Configuration
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TOP_P: float = float(os.getenv("TOP_P", "0.8"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    
    # Conversation Configuration
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    MAX_CONVERSATION_TURNS: int = int(os.getenv("MAX_CONVERSATION_TURNS", "50"))
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://yourapp.vercel.app").split(",")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "/var/log/founder-intelligence.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get database connection URL"""
    return settings.SUPABASE_URL


def get_api_keys() -> Dict[str, str]:
    """Get all API keys"""
    return {
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY,
        "supabase_service_key": settings.SUPABASE_SERVICE_KEY,
        "gemini_api_key": settings.GEMINI_API_KEY,
        "apify_token": settings.APIFY_TOKEN,
    }


def validate_settings() -> bool:
    """Validate required settings"""
    required_settings = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "GEMINI_API_KEY",
        "APIFY_TOKEN"
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not getattr(settings, setting, None):
            missing_settings.append(setting)
    
    if missing_settings:
        print(f"Missing required settings: {', '.join(missing_settings)}")
        return False
    
    return True
