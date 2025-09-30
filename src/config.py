"""Configuration settings for ConvFinQA implementation."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ConvFinQAConfig:
    """Configuration for ConvFinQA agent."""
    
    # API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    DEFAULT_MODEL: str = "gpt-5"
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.1
    
    # Dataset Configuration
    DATASET_PATH: str = "data/convfinqa_dataset.json"
    
    # Agent Configuration
    MAX_CONTEXT_TURNS: int = 8
    ENABLE_SELF_CORRECTION: bool = True
    
    # Evaluation Configuration
    NUMERICAL_TOLERANCE: float = 0.01
    PERCENTAGE_TOLERANCE: float = 0.05
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Check if API key is available."""
        return cls.OPENAI_API_KEY is not None and len(cls.OPENAI_API_KEY) > 0
