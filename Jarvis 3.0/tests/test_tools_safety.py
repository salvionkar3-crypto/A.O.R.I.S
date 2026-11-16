"""Automated tests for tool security, path traversal protection, and permission gates."""

from pathlib import Path
import tempfile
import pytest

from jarvis.core.exceptions import ToolValidationError
from jarvis.tools.base import ToolPermission
from jarvis.tools.builtin import CalculatorTool, FileReaderTool, FileWriterTool
from jarvis.tools.registry import ToolRegistry, is_path_safe


@pytest.fixture
def temp_workspace() -> Path:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td).resolve()


@pytest.mark.asyncio
async def test_path_traversal_blocked(temp_workspace: Path):
    """Verify that path traversal attempts outside permitted roots are strictly blocked."""
    registry = ToolRegistry(allowed_roots=[temp_workspace])
    reader = FileReaderTool()
    registry.register(reader)

    # 1. Traversal relative path
    bad_rel_path = temp_workspace / ".." / ".." / "secret.txt"
    res1 = await registry.execute_tool("file_reader", {"file_path": str(bad_rel_path)})
    assert res1.success is False
    assert "Access denied" in res1.error

    # 2. Outside absolute path
    bad_abs_path = "C:/Windows/System32/drivers/etc/hosts"
    res2 = await registry.execute_tool("file_reader", {"file_path": bad_abs_path})
    assert res2.success is False
    assert "Access denied" in res2.error


@pytest.mark.asyncio
async def test_file_writer_confirmation_gate(temp_workspace: Path):
    """Verify that consequential tools pause and require confirmation before modifying disk."""
    registry = ToolRegistry(allowed_roots=[temp_workspace])
    writer = FileWriterTool()
    registry.register(writer)

    target_file = temp_workspace / "output.txt"
    assert not target_file.exists()

    # Attempt execution without user confirmation
    res = await registry.execute_tool(
        "file_writer",
        {"file_path": str(target_file), "content": "Important payload"},
        user_confirmed=False,
    )

    # Must NOT have written to disk
    assert not target_file.exists()
    assert res.requires_confirmation is True
    assert "requires explicit user confirmation" in res.error

    # Now execute WITH confirmation
    res_confirmed = await registry.execute_tool(
        "file_writer",
        {"file_path": str(target_file), "content": "Important payload"},
        user_confirmed=True,
    )
    assert res_confirmed.success is True
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "Important payload"


@pytest.mark.asyncio
async def test_file_reader_safe_execution(temp_workspace: Path):
    """Verify FileReaderTool reads valid files within boundaries."""
    registry = ToolRegistry(allowed_roots=[temp_workspace])
    registry.register(FileReaderTool())

    test_file = temp_workspace / "sample.log"
    test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

    res = await registry.execute_tool("file_reader", {"file_path": str(test_file), "max_lines": 2})
    assert res.success is True
    assert res.data["lines_read"] == 2
    assert "Line 1\nLine 2\n" == res.data["content"]


@pytest.mark.asyncio
async def test_calculator_safe_ast_evaluation():
    """Verify CalculatorTool correctly computes arithmetic and rejects unsafe operations."""
    calc = CalculatorTool()

    # Valid arithmetic
    r1 = calc.execute(expression="1024 * 768")
    assert r1.success is True
    assert r1.data["result"] == 786432.0

    r2 = calc.execute(expression="(100 - 20) / 4 + 2**3")
    assert r2.success is True
    assert r2.data["result"] == 28.0

    # Unsafe operation (arbitrary import / function call)
    r3 = calc.execute(expression="__import__('os').system('dir')")
    assert r3.success is False
    assert "Calculation error" in r3.error


@pytest.mark.asyncio
async def test_argument_validation(temp_workspace: Path):
    """Verify missing required arguments are caught and reported."""
    registry = ToolRegistry(allowed_roots=[temp_workspace])
    registry.register(FileWriterTool())

    res = await registry.execute_tool("file_writer", {"file_path": str(temp_workspace / "x.txt")})
    assert res.success is False
    assert "Missing required parameter 'content'" in res.error
