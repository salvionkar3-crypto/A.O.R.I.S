"""Command-line interface and main entry points for A.O.R.I.S."""

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Optional
import uvicorn

# Configure UTF-8 stdout/stderr encoding on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from jarvis.config import get_settings
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.orchestrator.engine import OrchestratorEngine
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.base import ProviderStatusCode
from jarvis.providers.factory import create_llm_provider
from jarvis.tools.builtin import (
    CalculatorTool,
    FileListTool,
    FileReaderTool,
    FileWriterTool,
    MemorySearchTool,
    PDFGeneratorTool,
    ResearchSearchTool,
    SystemInfoTool,
)
from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_engine() -> OrchestratorEngine:
    """Build and initialize orchestrator engine for CLI operations."""
    settings = get_settings()
    settings.ensure_directories()
    mem_store = SQLiteMemoryStore(db_path=settings.jarvis_db_path)
    mem_store.initialize()

    mem_service = MemoryService(store=mem_store, settings=settings)
    mem_service.initialize()

    task_store = TaskStore(db_path=settings.jarvis_db_path)
    task_store.initialize()

    registry = ToolRegistry(
        allowed_roots=[Path.cwd(), settings.jarvis_data_dir, settings.aoris_docs_dir]
    )
    registry.register(FileReaderTool())
    registry.register(FileWriterTool())
    registry.register(FileListTool())
    registry.register(CalculatorTool())
    registry.register(SystemInfoTool())
    registry.register(MemorySearchTool(mem_service))
    registry.register(ResearchSearchTool())
    registry.register(PDFGeneratorTool(output_dir=settings.aoris_docs_dir))

    provider = create_llm_provider(settings)

    engine = OrchestratorEngine(
        memory_service=mem_service,
        provider=provider,
        tool_registry=registry,
        task_store=task_store,
    )
    engine.initialize()
    return engine


async def run_chat_command(message: str, specialist: Optional[str] = None) -> None:
    """Execute a single chat command in CLI with multi-model routing and attribution."""
    engine = create_engine()
    try:
        print(f"\n[USER]: {message}\n")
        task = await engine.process_request(user_request=message, specialist_override=specialist)

        # Multi-model provider source attribution
        orch = task.metadata.get("orchestrator", "NVIDIA Nemotron")
        orch_m = task.metadata.get("orchestrator_model", engine.provider.default_model)
        spec = task.metadata.get("specialist_used")
        synth = task.metadata.get("synthesis_performed", False)
        doc_url = task.metadata.get("download_url")

        print(f"[ORCHESTRATOR]: {orch} ({orch_m})")
        if spec:
            print(f"[SPECIALIST]:   {spec} ({task.metadata.get('specialist_model', 'active')})")
        if synth:
            print(f"[SYNTHESIS]:    NVIDIA Nemotron final synthesis verified")
        if doc_url:
            print(f"[DOCUMENT]:     {doc_url} ({task.metadata.get('file_name', 'document.pdf')})")

        print(f"\n[A.O.R.I.S.]:\n{task.final_response}\n")

        if task.verification:
            print(f"[VERIFICATION]: {task.verification.verdict.value.upper()} - {task.verification.evidence}")
    finally:
        engine.close()


async def run_health_command() -> None:
    """Check health and live inference capability across all 4 multi-model providers."""
    engine = create_engine()
    try:
        health_map = await engine.model_registry.check_all_health()
        statuses = engine.model_registry.get_all_statuses()

        print("\n" + "=" * 68)
        print("           A.O.R.I.S. MULTI-MODEL SYSTEM DIAGNOSTICS & HEALTH")
        print("=" * 68)
        print(f"{'PROVIDER':<12} | {'ROLE':<14} | {'MODEL':<24} | {'STATUS'}")
        print("-" * 68)

        for pid, prov in statuses.items():
            h = health_map.get(pid)
            stat = prov.status.value.upper()
            role_desc = prov.role.value.capitalize()
            print(f"{prov.display_name[:12]:<12} | {role_desc:<14} | {prov.model[:24]:<24} | {stat}")

        print("-" * 68)
        orch_h = health_map.get("nemotron")
        if orch_h:
            print(f"Primary Orchestrator: {orch_h.provider_name} ({orch_h.model})")
            print(f"Orchestrator Probe:   {orch_h.message}")
            if orch_h.latency_ms:
                print(f"Probe Latency:        {orch_h.latency_ms} ms")

        print(f"Persistent Storage:   {engine.memory_service.store.db_path}")
        print(f"Documents Directory:  {engine.memory_service.settings.aoris_docs_dir}")
        print(f"Registered Tools:     {[t['name'] for t in engine.tool_registry.list_tools()]}")
        print("=" * 68 + "\n")
    finally:
        engine.close()


def cli():
    """Main CLI entry point for A.O.R.I.S."""
    parser = argparse.ArgumentParser(
        prog="aoris",
        description="A.O.R.I.S. — Multi-Model AI Orchestration System (APEX LEGENDS | Nebius × NVIDIA)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch the A.O.R.I.S. Web UI & API server")
    serve_parser.add_argument("--host", default=None, help="Host address (default: 0.0.0.0 or $HOST)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port (default: 8000 or $PORT)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a command or query to A.O.R.I.S.")
    chat_parser.add_argument("message", help="Message or prompt to process")
    chat_parser.add_argument(
        "--specialist",
        "--model",
        dest="specialist",
        default=None,
        help="Optional specialist to route task to: openai, anthropic, google",
    )

    # Health command
    subparsers.add_parser("health", help="Run multi-model health probes across all 4 providers")

    args = parser.parse_args()

    if args.subcommand == "serve":
        settings = get_settings()
        host = args.host or settings.jarvis_host
        port = args.port or settings.jarvis_port
        reload = args.reload and settings.jarvis_env != "production"

        print("\n" + "=" * 68)
        print("       A.O.R.I.S. MULTI-MODEL AI ORCHESTRATION SYSTEM")
        print("=" * 68)
        print(f"Environment:       {settings.jarvis_env}")
        print(f"Listening On:      http://{host}:{port}")
        print(f"Persistent Data:   {settings.jarvis_data_dir}")
        print(f"Database Path:     {settings.jarvis_db_path}")
        print(f"Documents Dir:     {settings.aoris_docs_dir}")
        print(f"Auth Protected:    {'YES (JARVIS_API_KEY active)' if settings.jarvis_api_key else 'NO (Unrestricted/Dev Mode)'}")
        print(f"Orchestrator:      Nebius Token Factory with NVIDIA Nemotron")
        print(f"Orchestrator Model:{settings.nebius_default_model}")
        print(f"OpenAI Specialist: {'CONFIGURED' if settings.openai_api_key else 'UNCONFIGURED'}")
        print(f"Claude Specialist: {'CONFIGURED' if settings.anthropic_api_key else 'UNCONFIGURED'}")
        print(f"Gemini Specialist: {'CONFIGURED' if settings.google_api_key else 'UNCONFIGURED'}")
        print("=" * 68 + "\n")

        uvicorn.run("jarvis.web.app:app", host=host, port=port, reload=reload)
    elif args.subcommand == "chat":
        asyncio.run(run_chat_command(args.message, specialist=getattr(args, "specialist", None)))
    elif args.subcommand == "health":
        asyncio.run(run_health_command())
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()


