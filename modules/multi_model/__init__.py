"""
Multi-Model Intelligence System for PHOENIX
Enables PHOENIX to leverage multiple AI models for different tasks
"""

from .model_manager import ModelManager, ModelSpec
from .model_router import ModelRouter, TaskType

__all__ = [
    'ModelManager',
    'ModelSpec',
    'ModelRouter',
    'TaskType'
]