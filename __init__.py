"""
PLAIground v2: Compoundable Model Prototype

Implements the Three Contracts pattern:
- Data Contract: Input/output schema validation
- Task Contract: Capability + TaskConfig
- System Contract: Registry + Connectors
"""

from .contracts import DataContract, TaskContract, TaskConfig, Capability
from .core import CompoundableModel, compoundable_model, CompoundableModelRegistry
from .workflow import Workflow, WorkflowBuilder, WorkflowExecutor, Edge, DataMapping
from .connectors import ConnectorPool

__all__ = [
    'DataContract',
    'TaskContract',
    'TaskConfig',
    'Capability',
    'CompoundableModel',
    'compoundable_model',
    'CompoundableModelRegistry',
    'Workflow',
    'WorkflowBuilder',
    'WorkflowExecutor',
    'Edge',
    'DataMapping',
    'ConnectorPool',
]
