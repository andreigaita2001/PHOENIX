"""
Self-Modification Framework for PHOENIX
A modular, extensible system for autonomous code improvement
"""

from .code_analyzer import CodeAnalyzer
from .modification_engine import ModificationEngine
from .test_framework import TestFramework
from .reflection_system import ReflectionSystem
from .safety_monitor import SafetyMonitor
from .self_improver import SelfImprover

__all__ = [
    'CodeAnalyzer',
    'ModificationEngine',
    'TestFramework',
    'ReflectionSystem',
    'SafetyMonitor',
    'SelfImprover'
]