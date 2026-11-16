"""Automated tests for agent orchestration, intent routing, and verification."""

from pathlib import Path
import tempfile
import pytest

from jarvis.agents.verifier import VerificationVerdict, VerifierAgent
from jarvis.core.enums import AgentRole
from jarvis.memory.extractors import RuleBasedMemoryExtractor
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.orchestrator.engine import OrchestratorEngine
from jarvis.orchestrator.models import TaskState
from jarvis.orchestrator.router import AgentRouter
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.offline import OfflineProvider
from jarvis.tools.builtin import CalculatorTool, FileReaderTool, FileWriterTool
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def test_engine():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        db_file = root / "jarvis_orch_test.db"

        mem_store = SQLiteMemoryStore(db_path=db_file)
        mem_store.initialize()
        mem_service = MemoryService(store=mem_store, extractor=RuleBasedMemoryExtractor())
        mem_service.initialize()

        task_store = TaskStore(db_path=db_file)
        task_store.initialize()

        registry = ToolRegistry(allowed_roots=[root])
        registry.register(CalculatorTool())
        registry.register(FileReaderTool())
        registry.register(FileWriterTool())

        provider = OfflineProvider()
        engine = OrchestratorEngine(
            memory_service=mem_service,
            provider=provider,
            tool_registry=registry,
            task_store=task_store,
        )
        engine.initialize()
        yield engine, root
        engine.close()


def test_agent_router_intent_classification():
    """Verify router correctly maps user inputs to appropriate specialist agents and tools."""
    router = AgentRouter()

    # Calculator
    d1 = router.route("Calculate 256 * 4")
    assert d1.primary_agent == AgentRole.COMPUTER
    assert d1.suggested_tool == "calculator"
    assert d1.tool_args["expression"] == "256 * 4"

    # File Writing
    d2 = router.route("Write a file named notes.txt with content 'Project overview'")
    assert d2.primary_agent == AgentRole.COMPUTER
    assert d2.suggested_tool == "file_writer"
    assert d2.tool_args["file_path"] == "notes.txt"

    # Coding
    d3 = router.route("Write code for a Python binary search function")
    assert d3.primary_agent == AgentRole.CODING

    # Memory
    d4 = router.route("Remember that my preferred database is SQLite")
    assert d4.intent == "memory_store"

    # Conversation
    d5 = router.route("Hello, who are you?")
    assert d5.primary_agent == AgentRole.CONVERSATION


@pytest.mark.asyncio
async def test_calculation_task_execution(test_engine):
    """Verify full execution of a direct calculation task with verifier evidence."""
    engine, root = test_engine
    task = await engine.process_request("Calculate 50 + 25 * 2")

    assert task.state == TaskState.COMPLETED
    assert "100.0" in task.final_response
    assert task.verification is not None
    assert task.verification.verdict == VerificationVerdict.VERIFIED


@pytest.mark.asyncio
async def test_consequential_tool_confirmation_workflow(test_engine):
    """Verify task pauses in awaiting_confirmation and resumes upon approval."""
    engine, root = test_engine
    target_path = str(root / "verified_output.txt")
    req = f"Write a file named {target_path} with content Hello_Autonomous_World"

    # 1. Process without pre-confirmation -> enters awaiting_confirmation
    task = await engine.process_request(req, user_confirmed=False)
    assert task.state == TaskState.AWAITING_CONFIRMATION
    assert task.pending_confirmation is not None
    assert task.verification.verdict == VerificationVerdict.INCOMPLETE
    assert not Path(target_path).exists()

    # 2. Approve the task
    resumed = await engine.approve_task(task.task_id)
    assert resumed.state == TaskState.COMPLETED
    assert resumed.verification.verdict == VerificationVerdict.VERIFIED
    assert Path(target_path).exists()
    assert "Hello_Autonomous_World" in Path(target_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_consequential_tool_denial_workflow(test_engine):
    """Verify user can deny a pending consequential task."""
    engine, root = test_engine
    target_path = str(root / "unauthorized.txt")
    req = f"Write a file named {target_path} with content Danger"

    task = await engine.process_request(req, user_confirmed=False)
    assert task.state == TaskState.AWAITING_CONFIRMATION

    denied = engine.deny_task(task.task_id, reason="User rejected file write")
    assert denied.state == TaskState.CANCELLED
    assert not Path(target_path).exists()
    assert "cancelled" in denied.final_response.lower()
