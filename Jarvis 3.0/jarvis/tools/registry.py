"""Centralized ToolRegistry with strict validation and permission gating."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from jarvis.core.exceptions import (
    ToolExecutionError,
    ToolPermissionError,
    ToolValidationError,
)
from jarvis.tools.base import BaseTool, ToolPermission, ToolResult
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


def resolve_safe_path(target_path: str | Path, allowed_roots: List[Path]) -> Optional[Path]:
    """
    Resolve target_path within allowed_roots.
    If target_path is relative, it is resolved against the primary allowed root.
    Returns resolved Path if safe, or None if outside permitted directories.
    """
    try:
        p = Path(target_path)
        if not allowed_roots:
            return p.resolve()

        primary_root = allowed_roots[0].resolve()
        if not p.is_absolute():
            candidate = (primary_root / p).resolve()
            if candidate == primary_root or primary_root in candidate.parents:
                return candidate

        resolved = p.resolve()
        for root in allowed_roots:
            r = root.resolve()
            if resolved == r or r in resolved.parents:
                return resolved
        return None
    except Exception:
        return None


def is_path_safe(target_path: str | Path, allowed_roots: List[Path]) -> bool:
    """
    Verify that target_path resolves strictly within one of the allowed_roots,
    preventing path traversal and unauthorized directory access.
    """
    return resolve_safe_path(target_path, allowed_roots) is not None


class ToolRegistry:
    """
    Central tool registry enforcing least-privilege access, schema validation,
    path traversal guards, user confirmation gates, and timeout policies.
    """

    def __init__(self, allowed_roots: Optional[List[Path]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        # Default allowed roots: workspace directory and data directory
        self.allowed_roots: List[Path] = allowed_roots or [
            Path.cwd(),
            Path.cwd() / "data",
        ]
        self._executor = ThreadPoolExecutor(max_workers=4)

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool '{tool.name}' (Permission: {tool.permission.value})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Fetch a tool by name."""
        return self._tools.get(name)

    def get(self, name: str, default: Optional[BaseTool] = None) -> Optional[BaseTool]:
        """Fetch a tool by name (dict-like convenience)."""
        return self._tools.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """List metadata and schemas for all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "permission": t.permission.value,
                "requires_confirmation": t.requires_user_confirmation,
                "parameters_schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def validate_arguments(self, tool: BaseTool, arguments: Dict[str, Any]) -> None:
        """Validate arguments against tool parameters schema."""
        schema = tool.parameters_schema
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                raise ToolValidationError(
                    f"Missing required parameter '{req}' for tool '{tool.name}'."
                )

        # Check for file path parameters and enforce path safety
        for key, val in arguments.items():
            if ("path" in key.lower() or "file" in key.lower()) and isinstance(val, (str, Path)):
                # If it's a file path, ensure it doesn't traverse outside allowed roots
                safe_p = resolve_safe_path(val, self.allowed_roots)
                if not safe_p:
                    raise ToolValidationError(
                        f"Access denied: path '{val}' is outside permitted directories."
                    )
                arguments[key] = str(safe_p)

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_confirmed: bool = False,
    ) -> ToolResult:
        """
        Execute a tool through the security and permission pipeline.
        
        Enforces:
        1. Tool existence
        2. Argument validation & path traversal check
        3. Human confirmation gate for consequential/destructive actions
        4. Timeout enforcement
        5. Execution measurement & honest error capture
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered.",
            )

        # Validate arguments
        try:
            self.validate_arguments(tool, arguments)
        except ToolValidationError as e:
            return ToolResult(tool_name=name, success=False, error=str(e))

        # Check permission gate
        if tool.requires_user_confirmation and not user_confirmed:
            preview = tool.preview_action(**arguments)
            logger.info(f"Tool '{name}' requires confirmation. Preview: {preview}")
            return ToolResult(
                tool_name=name,
                success=False,
                requires_confirmation=True,
                confirmation_prompt=preview,
                error="Action requires explicit user confirmation before execution.",
            )

        # Execute with timeout
        start_time = time.time()
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, lambda: tool.execute(**arguments)),
                timeout=tool.timeout,
            )
            result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
            return result
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' timed out after {tool.timeout} seconds.",
                execution_time_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool execution failed: {e}",
                execution_time_ms=round((time.time() - start_time) * 1000, 2),
            )
