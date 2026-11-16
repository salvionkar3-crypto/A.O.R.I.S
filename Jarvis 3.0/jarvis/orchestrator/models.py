"""Task orchestration models, states, and lifecycle records."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from jarvis.agents.verifier import VerificationResult
from jarvis.core.enums import AgentRole, TaskStatus
from jarvis.core.types import JarvisBaseModel, generate_id, utc_now


class TaskState(str, Enum):
    """Execution state of a managed task."""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStep(JarvisBaseModel):
    """An individual execution step within a task plan."""
    step_id: int
    description: str
    agent_role: AgentRole = AgentRole.COMPUTER
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    status: TaskState = TaskState.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    completion_criteria: Optional[str] = None


class TaskRecord(JarvisBaseModel):
    """Complete, persistable record of a user task from submission to verification."""
    task_id: str = Field(default_factory=lambda: generate_id("task"))
    session_id: str
    user_request: str
    state: TaskState = TaskState.PENDING
    plan: List[TaskStep] = Field(default_factory=list)
    current_step_index: int = 0
    pending_confirmation: Optional[Dict[str, Any]] = None
    retrieved_memory_ids: List[str] = Field(default_factory=list)
    final_response: Optional[str] = None
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
    workflow_type: Optional[str] = None
    intent: Optional[str] = None
    complexity: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    replanned: bool = False
    replan_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
