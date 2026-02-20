"""Core module - LLM client, task routing, context management."""

from src.core.competence_model import CompetenceEntry, CompetenceMap
from src.core.llm_client import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse", "CompetenceEntry", "CompetenceMap"]
