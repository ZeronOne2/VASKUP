"""
VASKUP Utils Module

Utility functions for the VASKUP patent analysis system.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for VASKUP.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary containing configuration settings
    """
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "serpapi_key": os.getenv("SERPAPI_API_KEY"),
        "chromadb_persist_dir": os.getenv(
            "CHROMADB_PERSIST_DIRECTORY", "./vectorstore"
        ),
        "streamlit_port": int(os.getenv("STREAMLIT_PORT", "8501")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "max_patents_per_search": int(os.getenv("MAX_PATENTS_PER_SEARCH", "10")),
        "search_timeout": int(os.getenv("SEARCH_TIMEOUT", "30")),
        "default_llm_model": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "reranking_model": os.getenv(
            "RERANKING_MODEL", "jina-reranker-v2-base-multilingual"
        ),
    }
    return config


def print_model_config() -> None:
    """Print current model configuration."""
    config = load_env_config()
    print("Model Configuration:")
    print(f"  LLM Model: {config['default_llm_model']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  Reranking Model: {config['reranking_model']}")


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = ["data", "vectorstore", "reports", "temp"]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are present.

    Returns:
        Dictionary with validation results for each API key
    """
    config = load_env_config()

    return {
        "openai": bool(config.get("openai_api_key")),
        "serpapi": bool(config.get("serpapi_key")),
    }
