"""
models — version-agnostic domain models for Cell AI.

Each model accepts a cell_system (CellAI v1/v2/v3 instance) as its backbone.
Use models.registry.get_model() to instantiate any combination.
"""
from models.registry import get_model, list_models
from models.base_model import CellAIModel

__all__ = ["get_model", "list_models", "CellAIModel"]
