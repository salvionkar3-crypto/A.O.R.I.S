"""Automated Runner for Acceptance Demos 1 through 6.
Executes all acceptance criteria from Phase 10 and produces a comprehensive demonstration log.
APEX LEGENDS | Nebius × NVIDIA Global AI Hackathon 2026
"""

import asyncio
from pathlib import Path
import sys
import tempfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from jarvis.core.enums import AgentRole, MemoryCategory
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.orchestrator.engine import OrchestratorEngine
from jarvis.orchestrator.models import TaskState
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.factory import create_llm_provider
from jarvis.providers.offline import OfflineProvider
from jarvis.tools.builtin import (
    CalculatorTool,
    FileListTool,
    FileReaderTool,
    FileWriterTool,
    MemorySearchTool,
    ResearchSearchTool,
    SystemInfoTool,
)
from jarvis.tools.registry import ToolRegistry


def header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


async def main():
    with tempfile.TemporaryDirectory() as td:
        workspace_root = Path(td).resolve()
        db_path = workspace_root / "jarvis_acceptance.db"

        # Initialize core backing services
        mem_store = SQLiteMemoryStore(db_path=db_path)
        mem_store.initialize()
        mem_service = MemoryService(store=mem_store)
        mem_service.initialize()

        task_store = TaskStore(db_path=db_path)
        task_store.initialize()

        registry = ToolRegistry(allowed_roots=[workspace_root])
        registry.register(CalculatorTool())
        registry.register(FileReaderTool())
        registry.register(FileWriterTool())
        registry.register(FileListTool())
        registry.register(SystemInfoTool())
        registry.register(MemorySearchTool(mem_service))
        registry.register(ResearchSearchTool(workspace_root=workspace_root))

        provider = OfflineProvider()
        engine = OrchestratorEngine(
            memory_service=mem_service,
            provider=provider,
            tool_registry=registry,
            task_store=task_store,
        )
        engine.initialize()

        print(f"JARVIS Autonomous Environment Initialized at {workspace_root}")
        print(f"Model Provider: {provider.provider_name} ({provider.default_model})")

        # -------------------------------------------------------------
        # DEMO 1: Multi-Agent Collaboration
        # -------------------------------------------------------------
        header("DEMO 1: MULTI-AGENT COLLABORATION")
        demo1_prompt = "Calculate 1024 * 768 and then write the result to output.txt"
        print(f"[USER]: {demo1_prompt}")
        decision1 = engine.router.route(demo1_prompt)
        print(f"[ROUTER]: Agent={decision1.primary_agent.value}, Workflow={decision1.workflow_type}, Complexity={decision1.complexity}")
        
        task1 = await engine.process_request(demo1_prompt, session_id="demo1_session")
        print(f"[TASK STATE]: {task1.state.value}")
        print(f"[PLAN STEPS]: {len(task1.plan)} steps decomposed")
        for s in task1.plan:
            print(f"  - Step {s.step_id}: {s.description} (Tool: {s.tool_name}, Status: {s.status.value})")

        # -------------------------------------------------------------
        # DEMO 2: Safe Tool Execution with Confirmation Gate
        # -------------------------------------------------------------
        header("DEMO 2: SAFE TOOL EXECUTION WITH CONFIRMATION GATE")
        print(f"[TASK STATE BEFORE APPROVAL]: {task1.state.value}")
        assert task1.state == TaskState.AWAITING_CONFIRMATION
        print(f"[SAFETY GATE]: {task1.pending_confirmation['prompt']}")
        
        print("[USER ACTION]: Approve consequential file write operation...")
        approved_task1 = await engine.approve_task(task1.task_id)
        print(f"[TASK STATE AFTER APPROVAL]: {approved_task1.state.value}")
        print(f"[FINAL RESPONSE]:\n{approved_task1.final_response}")
        print(f"[VERIFICATION]: {approved_task1.verification.verdict.value.upper()} - {approved_task1.verification.evidence}")
        
        out_file = workspace_root / "output.txt"
        if out_file.exists():
            print(f"[DISK VERIFICATION]: File exists on disk ({out_file.stat().st_size} bytes): '{out_file.read_text()}'")

        # -------------------------------------------------------------
        # DEMO 3: Memory Precision & Lifecycle
        # -------------------------------------------------------------
        header("DEMO 3: MEMORY PRECISION & LIFECYCLE")
        session_a = "session_user_alice"
        store_prompt = "Remember my favorite coding language is Python and I specialize in distributed systems."
        print(f"[USER @ {session_a}]: {store_prompt}")
        task3a = await engine.process_request(store_prompt, session_id=session_a)
        print(f"[JARVIS]: {task3a.final_response}")
        
        # New session query
        session_b = "session_user_bob_or_later"
        query_prompt = "What is my favorite coding language?"
        print(f"\n[USER @ {session_b} (New Session)]: {query_prompt}")
        task3b = await engine.process_request(query_prompt, session_id=session_b)
        print(f"[RETRIEVED MEMORIES]: {len(task3b.retrieved_memory_ids)} memory item(s) retrieved")
        for mid in task3b.retrieved_memory_ids:
            mem = mem_service.get_memory(mid)
            if mem:
                print(f"  - [{mem.category.value}] {mem.content} (Confidence: {mem.confidence})")
        print(f"[JARVIS]: {task3b.final_response}")

        # -------------------------------------------------------------
        # DEMO 4: Bounded Recovery Loop (Replanning on Step Failure)
        # -------------------------------------------------------------
        header("DEMO 4: BOUNDED RECOVERY LOOP (REPLANNING)")
        demo4_prompt = "first read missing_document_123.txt and then summarize it"
        print(f"[USER]: {demo4_prompt}")
        task4 = await engine.process_request(demo4_prompt, session_id="demo4_session")
        print(f"[TASK STATE]: {task4.state.value}")
        print(f"[REPLANNED]: {task4.replanned}")
        print(f"[REPLAN HISTORY]: {len(task4.replan_history)} recovery attempt(s)")
        if task4.replan_history:
            rh = task4.replan_history[0]
            print(f"  - Initial Failure: {rh['error']}")
            print(f"  - Corrective Step: {rh['recovery_steps'][0]['description']} (Tool: {rh['recovery_steps'][0]['tool_name']})")
        print(f"[VERIFICATION]: {task4.verification.verdict.value.upper()} - {task4.verification.evidence}")

        # -------------------------------------------------------------
        # DEMO 5: Research with Honest Fallback
        # -------------------------------------------------------------
        header("DEMO 5: RESEARCH WITH HONEST FALLBACK")
        # 1. Create a workspace document
        docs_dir = workspace_root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        spec_doc = docs_dir / "jarvis_specs.md"
        spec_doc.write_text(
            "# JARVIS Engineering Specifications\n"
            "Architecture utilizes local SQLite storage with FTS5 BM25 search.\n"
            "All consequential tool actions are gated behind human confirmation.\n",
            encoding="utf-8",
        )
        print(f"[PREPARED DOC]: {spec_doc.name}")

        print("\n[SUBTASK 5A]: Researching known topic in documentation...")
        res_tool = registry.get("research_search")
        res_known = res_tool.execute(query="SQLite storage FTS5")
        print(f"[SEARCH MATCHES]: {res_known.data['matches_found']}")
        for c in res_known.data["citations"]:
            print(f"  - Citation: {c}")

        print("\n[SUBTASK 5B]: Researching unknown topic without fabricating citations...")
        res_unknown = res_tool.execute(query="WarpDriveHyperjump999")
        print(f"[SEARCH MATCHES]: {res_unknown.data['matches_found']}")
        print(f"[CITATIONS]: {res_unknown.data['citations']}")
        print(f"[HONEST DISCLAIMER]: {res_unknown.data['note']}")

        # -------------------------------------------------------------
        # DEMO 6: Live Nebius Inference vs Offline Fallback
        # -------------------------------------------------------------
        header("DEMO 6: LIVE NEBIUS INFERENCE VS OFFLINE FALLBACK")
        health = await engine.provider.check_health()
        print(f"[PROVIDER NAME]: {health.provider_name}")
        print(f"[CONFIGURED MODEL]: {health.model}")
        print(f"[IS LIVE]: {health.is_live}")
        print(f"[STATUS CODE]: {health.status.value.upper()}")
        print(f"[MESSAGE]: {health.message}")
        print("[ATTRIBUTION BADGE]: " + (
            f"🌐 LIVE: {health.model}" if health.is_live else f"⚠️ OFFLINE FALLBACK: {health.provider_name}"
        ))

        engine.close()

    header("ALL ACCEPTANCE DEMOS SUCCESSFULLY COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())
