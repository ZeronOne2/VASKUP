"""
VASKUP: Advanced Agentic RAG System for Patent Analysis using LangGraph

An advanced patent analysis system that leverages LangGraph and multiple AI agents
to provide comprehensive patent research and analysis capabilities.
"""

__version__ = "0.1.0"
__author__ = "VASKUP Team"
__description__ = "Advanced Agentic RAG System for Patent Analysis using LangGraph"

# Package imports
from . import core
from . import utils

__all__ = [
    "core",
    "utils",
    "__version__",
    "__author__",
    "__description__",
]
