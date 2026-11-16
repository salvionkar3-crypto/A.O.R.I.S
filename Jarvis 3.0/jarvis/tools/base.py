"""Base interfaces, permissions, and schemas for guarded tool execution."""

from abc import ABC, abstractmethod
from enum import Enum
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolPermission(str, Enum):
    """Permission level required to execute a tool."""
    READ_ONLY = "read_only"          # Safe, read-only inspection (no state modification)
    CONSEQUENTIAL = "consequential"  # Modifies state (file writes, database modifications)
    DESTRUCTIVE = "destructive"      # Irreversible deletion or system alteration


class ToolResult(BaseModel):
    """Structured result returned by a tool execution."""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    """Abstract base class for all JARVIS tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Clear description of what the tool does and when to use it."""
        pass

    @property
    @abstractmethod
    def permission(self) -> ToolPermission:
        """Permission level required."""
        pass

    @property
    def requires_user_confirmation(self) -> bool:
        """Whether this tool requires explicit human approval before running."""
        return self.permission in (ToolPermission.CONSEQUENTIAL, ToolPermission.DESTRUCTIVE)

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema dictionary describing required and optional parameters."""
        pass

    @property
    def timeout(self) -> float:
        """Default execution timeout in seconds."""
        return 10.0

    def preview_action(self, **kwargs: Any) -> str:
        """Human-readable description of what will happen if executed."""
        return f"Execute tool '{self.name}' with arguments: {kwargs}"

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Synchronously execute the tool with given validated arguments."""
        pass

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """Asynchronously execute the tool."""
        start_time = time.time()
        # Default implementation delegates to synchronous execute
        result = self.execute(**kwargs)
        result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
        return result
