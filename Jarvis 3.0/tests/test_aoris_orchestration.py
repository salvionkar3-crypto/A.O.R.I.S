"""Comprehensive tests for A.O.R.I.S. multi-model orchestration system.

Covers:
1. MultiModelRegistry and status reporting across all 4 providers.
2. Specialist provider adapters (OpenAI, Anthropic, Google Gemini) with mocked HTTP responses and error handling.
3. AgentRouter intent classification for specialists and user overrides.
4. NVIDIA Nemotron final synthesis workflow.
5. Specialist fallback when credentials are unconfigured or unavailable.
6. ReportLab PDF generation tool and document structure verification.
7. Secure download endpoint with strict path traversal defense.
8. Web API status and chat badges.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from jarvis.config import Settings, get_settings, reset_settings
from jarvis.core.enums import AgentRole, MemoryCategory
from jarvis.core.exceptions import AuthenticationError, RateLimitError
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.orchestrator.engine import OrchestratorEngine
from jarvis.orchestrator.router import AgentRouter
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.base import (
    BaseLLMProvider,
    CompletionMessage,
    CompletionResponse,
    MultiModelProviderInfo,
    MultiModelStatus,
    ProviderRole,
)
from jarvis.providers.offline import OfflineProvider
from jarvis.providers.registry import MultiModelRegistry
from jarvis.providers.specialists import (
    AnthropicCodingProvider,
    GeminiVisionProvider,
    OpenAIReasoningProvider,
)
from jarvis.tools.builtin import PDFGeneratorTool
from jarvis.tools.registry import ToolRegistry
from jarvis.web.app import app, init_services


# --- Fixtures ---

@pytest.fixture
def clean_engine(tmp_path: Path):
    db_file = tmp_path / "test_aoris.db"
    docs_dir = tmp_path / "test_docs"
    settings = Settings(jarvis_db_path=db_file, jarvis_data_dir=tmp_path, aoris_docs_dir=docs_dir)
    settings.ensure_directories()

    store = SQLiteMemoryStore(db_path=db_file)
    store.initialize()
    mem_service = MemoryService(store=store, settings=settings)
    mem_service.initialize()
    task_store = TaskStore(db_path=db_file)
    task_store.initialize()

    tool_registry = ToolRegistry(allowed_roots=[tmp_path, docs_dir])
    pdf_tool = PDFGeneratorTool(output_dir=docs_dir)
    tool_registry.register(pdf_tool)

    provider = OfflineProvider()
    engine = OrchestratorEngine(
        memory_service=mem_service,
        provider=provider,
        tool_registry=tool_registry,
        task_store=task_store,
    )
    engine.initialize()
    yield engine
    engine.close()


# --- 1. MultiModelRegistry & Status Reporting ---

def test_multimodel_status_reporting_all_providers():
    """Verify MultiModelRegistry tracks statuses for Nemotron, OpenAI, Anthropic, and Google."""
    settings = Settings()
    registry = MultiModelRegistry(settings=settings)
    statuses = registry.get_all_statuses()

    assert "nemotron" in statuses
    assert "openai" in statuses
    assert "anthropic" in statuses
    assert "google" in statuses

    # Nemotron is orchestrator
    assert statuses["nemotron"].role == ProviderRole.ORCHESTRATOR
    # OpenAI is reasoning
    assert statuses["openai"].role == ProviderRole.REASONING
    assert statuses["openai"].status == MultiModelStatus.CREDENTIALS_MISSING
    # Anthropic is coding
    assert statuses["anthropic"].role == ProviderRole.CODING
    assert statuses["anthropic"].status == MultiModelStatus.CREDENTIALS_MISSING
    # Google is vision
    assert statuses["google"].role == ProviderRole.VISION
    assert statuses["google"].status == MultiModelStatus.CREDENTIALS_MISSING


# --- 2. Specialist Provider Adapters ---

@pytest.mark.asyncio
async def test_openai_reasoning_provider_success():
    """Verify OpenAIReasoningProvider parses chat completions response."""
    provider = OpenAIReasoningProvider(api_key="test-key-openai", model="gpt-4o-mini")
    mock_resp = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"role": "assistant", "content": "Analytical deduction result: X=42"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = mock_resp
        mock_post.return_value = mock_response

        res = await provider.generate_response([CompletionMessage(role="user", content="Solve for X")])
        assert res.content == "Analytical deduction result: X=42"
        assert res.model == "gpt-4o-mini"
        assert res.provider_name == "openai_reasoning"
        assert res.is_live is True
        assert provider._last_status == MultiModelStatus.AVAILABLE_VERIFIED


@pytest.mark.asyncio
async def test_openai_reasoning_provider_auth_failure():
    """Verify OpenAIReasoningProvider handles 401/403 auth failure and updates status."""
    provider = OpenAIReasoningProvider(api_key="invalid-key")
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock(status_code=401, text="Incorrect API key provided")
        mock_post.return_value = mock_response

        with pytest.raises(AuthenticationError):
            await provider.generate_response([CompletionMessage(role="user", content="Test")])
        assert provider._last_status == MultiModelStatus.AUTH_FAILED


@pytest.mark.asyncio
async def test_anthropic_coding_provider_success():
    """Verify AnthropicCodingProvider formats messages and parses response."""
    provider = AnthropicCodingProvider(api_key="test-key-claude", model="claude-3-5-sonnet-20241022")
    mock_resp = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "def solve():\n    return 42"}],
        "usage": {"input_tokens": 12, "output_tokens": 9},
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = mock_resp
        mock_post.return_value = mock_response

        messages = [
            CompletionMessage(role="system", content="You are a senior python engineer."),
            CompletionMessage(role="user", content="Write solve() function"),
        ]
        res = await provider.generate_response(messages)
        assert "def solve():" in res.content
        assert res.model == "claude-3-5-sonnet-20241022"
        assert provider._last_status == MultiModelStatus.AVAILABLE_VERIFIED


@pytest.mark.asyncio
async def test_gemini_vision_provider_success():
    """Verify GeminiVisionProvider handles multimodal response format."""
    provider = GeminiVisionProvider(api_key="test-key-gemini", model="gemini-1.5-flash")
    mock_resp = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Visual Analysis: Clean modern dashboard layout with navy palette."}]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 10},
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = mock_resp
        mock_post.return_value = mock_response

        res = await provider.generate_response([CompletionMessage(role="user", content="Describe layout")])
        assert "Visual Analysis" in res.content
        assert provider._last_status == MultiModelStatus.AVAILABLE_VERIFIED


# --- 3. AgentRouter Specialist Routing & Overrides ---

def test_agent_router_specialist_routing_and_overrides():
    """Verify AgentRouter routes coding to Anthropic, reasoning to OpenAI, vision to Google, and respects overrides."""
    router = AgentRouter()

    # Coding task -> Anthropic
    coding_dec = router.route("Please refactor this python script and fix error")
    assert coding_dec.intent == "coding"
    assert coding_dec.specialist_model == "anthropic"
    assert coding_dec.requires_synthesis is True

    # Reasoning task -> OpenAI
    reasoning_dec = router.route("Provide a mathematical proof and deep reasoning for Fermat's theorem")
    assert reasoning_dec.intent == "reasoning"
    assert reasoning_dec.specialist_model == "openai"
    assert reasoning_dec.requires_synthesis is True

    # Vision task -> Google
    vision_dec = router.route("Design a UI mockup with a color palette and visual styling")
    assert vision_dec.intent == "vision_creative"
    assert vision_dec.specialist_model == "google"
    assert vision_dec.requires_synthesis is True

    # User Override flag forces specialist
    override_dec = router.route("Hello assistant", specialist_override="openai")
    assert override_dec.specialist_model == "openai"
    assert override_dec.requires_synthesis is True

    # PDF generation route
    pdf_dec = router.route("Generate a PDF report titled System_Audit")
    assert pdf_dec.intent == "pdf_generation"
    assert pdf_dec.suggested_tool == "pdf_generator"


# --- 4. Nemotron Final Synthesis Workflow ---

@pytest.mark.asyncio
async def test_nemotron_final_synthesis_workflow(clean_engine: OrchestratorEngine):
    """Verify that when a specialist produces output, NVIDIA Nemotron synthesizes the final response."""
    # Mock specialist provider in registry
    mock_specialist = MagicMock(spec=BaseLLMProvider)
    mock_specialist.provider_name = "anthropic_coding"
    mock_specialist.default_model = "claude-3-5-sonnet-20241022"
    mock_specialist.is_live_provider = True
    mock_specialist.generate_response = AsyncMock(
        return_value=CompletionResponse(
            content="def calculate_total(a, b):\n    return a + b",
            model="claude-3-5-sonnet-20241022",
            is_live=True,
            provider_name="anthropic_coding",
        )
    )

    clean_engine.model_registry.specialists["anthropic"] = mock_specialist

    task = await clean_engine.process_request(
        user_request="Write code to calculate total",
        specialist_override="anthropic",
    )

    assert task.state.value == "completed"
    assert task.metadata.get("specialist_used") == "anthropic_coding"
    assert task.metadata.get("synthesis_performed") is True
    assert task.metadata.get("orchestrator") == "NVIDIA Nemotron"
    assert "calculate_total" in task.final_response


# --- 5. Specialist Fallback when Unconfigured ---

@pytest.mark.asyncio
async def test_specialist_fallback_to_nemotron_when_unconfigured(clean_engine: OrchestratorEngine):
    """Verify that if a specialist is requested but unconfigured, the system gracefully executes via Nemotron/offline fallback."""
    # Anthropic key is not configured in clean_engine
    task = await clean_engine.process_request(
        user_request="def fibonacci(n): pass",
        specialist_override="anthropic",
    )

    assert task.state.value == "completed"
    assert task.metadata.get("specialist_fallback") is True
    assert "fibonacci" in task.final_response.lower()


# --- 6. ReportLab PDF Generation Tool ---

def test_pdf_generator_tool_formatting_and_validation(tmp_path: Path):
    """Verify ReportLab PDF generation creates non-zero byte valid PDF with headers and tables."""
    docs_dir = tmp_path / "generated_pdfs"
    tool = PDFGeneratorTool(output_dir=docs_dir)

    result = tool.execute(
        title="APEX LEGENDS Quarterly Report",
        content="### Executive Summary\n\nA.O.R.I.S. multi-model system demonstrated superior synthesis capability.\n\n- Zero crashes across 10,000 interactions\n- Multi-model routing verified\n- Sub-second deterministic evaluation",
        table_data={
            "headers": ["Model", "Specialization", "Latency"],
            "rows": [
                ["NVIDIA Nemotron", "Orchestration & Synthesis", "180ms"],
                ["OpenAI GPT", "Reasoning & Logic", "320ms"],
                ["Anthropic Claude", "Code & Engineering", "290ms"],
                ["Google Gemini", "Vision & Creative", "240ms"],
            ],
        },
        filename="quarterly_report.pdf",
        author="A.O.R.I.S.",
    )

    assert result.success is True
    assert result.data["verified"] is True
    assert result.data["valid_header"] is True
    assert result.data["valid_eof"] is True
    assert result.data["bytes_written"] > 0
    assert result.data["download_url"] == "/api/downloads/quarterly_report.pdf"

    pdf_file = docs_dir / "quarterly_report.pdf"
    assert pdf_file.exists()
    with open(pdf_file, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


# --- 7. Download Endpoint & Path Traversal Security ---

@pytest.mark.asyncio
async def test_downloads_endpoint_security_and_traversal_defense(tmp_path: Path, monkeypatch):
    """Verify /api/downloads allows valid files and strictly blocks path traversal."""
    docs_dir = tmp_path / "safe_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    test_pdf = docs_dir / "valid_document.pdf"
    test_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    reset_settings()
    init_services(force_reload=True)
    get_settings().aoris_docs_dir = docs_dir

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Valid file download
        res = await client.get("/api/downloads/valid_document.pdf")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF-")

        # Nonexistent file -> 404
        res_404 = await client.get("/api/downloads/nonexistent.pdf")
        assert res_404.status_code == 404

        # Path traversal with .. -> 400
        res_traversal = await client.get("/api/downloads/..%2F..%2Fetc%2Fpasswd")
        assert res_traversal.status_code in (400, 404)


# --- 8. Web API Multi-Model Status & Chat Badges ---

@pytest.mark.asyncio
async def test_web_api_multi_model_status_endpoint():
    """Verify /api/providers/status returns operational status of all 4 providers."""
    init_services(force_reload=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/providers/status")
        assert res.status_code == 200
        data = res.json()
        assert "orchestrator" in data
        assert "providers" in data
        assert "nemotron" in data["providers"]
        assert "openai" in data["providers"]
        assert "anthropic" in data["providers"]
        assert "google" in data["providers"]


@pytest.mark.asyncio
async def test_web_api_chat_multi_model_badges():
    """Verify /api/chat returns orchestrator, specialist, and synthesis metadata."""
    init_services(force_reload=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/chat",
            json={
                "message": "def add(x, y): return x + y",
                "session_id": "test_session_badges",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["orchestrator"] == "NVIDIA Nemotron"
        assert "orchestrator_model" in data
        assert "synthesis_performed" in data
        assert "response" in data
