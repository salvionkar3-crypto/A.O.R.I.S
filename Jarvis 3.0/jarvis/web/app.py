"""FastAPI Web Application providing REST API and User Interface for A.O.R.I.S."""

from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from jarvis.config import Settings, get_settings
from jarvis.core.enums import FactStatus, MemoryCategory
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.orchestrator.engine import OrchestratorEngine
from jarvis.orchestrator.models import TaskRecord
from jarvis.orchestrator.store import TaskStore
from jarvis.providers.base import ProviderHealth
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
from jarvis.web.ui import JARVIS_HTML_PAGE

logger = setup_logger(__name__)


# --- Security Middlewares & Auth Guards ---

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforce maximum payload size to prevent memory exhaustion / DoS."""

    async def dispatch(self, request, call_next):
        cfg = get_settings()
        max_bytes = cfg.jarvis_max_request_size_bytes
        if max_bytes > 0:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"Payload Too Large (max {max_bytes} bytes allowed)"},
                        )
                except ValueError:
                    pass
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory rate limiting per client IP."""

    _instance: Optional["RateLimitMiddleware"] = None

    def __init__(self, app):
        super().__init__(app)
        self.requests: Dict[str, List[float]] = defaultdict(list)
        RateLimitMiddleware._instance = self

    @classmethod
    def reset(cls) -> None:
        """Clear rate limiting history (useful for test isolation)."""
        if cls._instance is not None:
            cls._instance.requests.clear()

    async def dispatch(self, request, call_next):
        # Skip rate limiting on static page and health probes
        if request.url.path in ("/", "/healthz", "/api/health"):
            return await call_next(request)

        cfg = get_settings()
        limit = cfg.jarvis_rate_limit_per_minute
        if limit <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0

        # Filter timestamps within current 60s sliding window
        timestamps = [t for t in self.requests[client_ip] if t > window_start]
        if len(timestamps) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Rate limit exceeded. Please wait a moment."},
                headers={"Retry-After": "60"},
            )
        timestamps.append(now)
        self.requests[client_ip] = timestamps
        return await call_next(request)



api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
) -> bool:
    """
    Verify access token when JARVIS_API_KEY is configured.
    If no key is configured in settings, allows open access (development mode).
    """
    cfg = get_settings()
    if not cfg.jarvis_api_key:
        return True

    token = x_api_key or (bearer.credentials if bearer else None)
    if not token or token.strip() != cfg.jarvis_api_key.strip():
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid API key. Provide X-API-Key header or Bearer token.",
        )
    return True


class AppServices:
    settings: Optional[Settings] = None
    memory_service: Optional[MemoryService] = None
    task_store: Optional[TaskStore] = None
    tool_registry: Optional[ToolRegistry] = None
    orchestrator: Optional[OrchestratorEngine] = None


services = AppServices()


def init_services(force_reload: bool = False) -> AppServices:
    """Ensure backing services are initialized (handles both ASGI lifespan and direct test calls)."""
    if services.orchestrator is None or force_reload:
        if force_reload:
            from jarvis.config import reset_settings
            reset_settings()
        services.settings = get_settings()
        services.settings.ensure_directories()


        db_path = services.settings.jarvis_db_path
        mem_store = SQLiteMemoryStore(db_path=db_path)
        mem_store.initialize()

        services.memory_service = MemoryService(
            store=mem_store,
            settings=services.settings,
        )
        services.memory_service.initialize()

        services.task_store = TaskStore(db_path=db_path)
        services.task_store.initialize()

        services.tool_registry = ToolRegistry(
            allowed_roots=[Path.cwd(), services.settings.jarvis_data_dir, services.settings.aoris_docs_dir]
        )
        services.tool_registry.register(FileReaderTool())
        services.tool_registry.register(FileWriterTool())
        services.tool_registry.register(FileListTool())
        services.tool_registry.register(CalculatorTool())
        services.tool_registry.register(SystemInfoTool())
        services.tool_registry.register(MemorySearchTool(services.memory_service))
        services.tool_registry.register(ResearchSearchTool())
        services.tool_registry.register(PDFGeneratorTool(output_dir=services.settings.aoris_docs_dir))

        provider = create_llm_provider(services.settings)
        services.orchestrator = OrchestratorEngine(
            memory_service=services.memory_service,
            provider=provider,
            tool_registry=services.tool_registry,
            task_store=services.task_store,
        )
        services.orchestrator.initialize()
    return services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize services on startup and safely close on shutdown."""
    init_services()
    logger.info("A.O.R.I.S. Web API and UI services initialized successfully.")
    yield
    if services.orchestrator:
        services.orchestrator.close()
        services.orchestrator = None
    logger.info("A.O.R.I.S. Web services shut down cleanly.")


app = FastAPI(
    title="A.O.R.I.S. API",
    description="A.O.R.I.S. — Multi-Model AI Orchestration System | APEX LEGENDS",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach production middlewares (CORS, Rate Limiting, Request Size Guard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().jarvis_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)


# --- Request/Response DTOs ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User instruction or dialogue")
    session_id: str = Field(default="default_session", description="Conversation session ID")
    user_confirmed: bool = Field(default=False, description="Explicit approval for consequential steps")
    specialist_override: Optional[str] = Field(default=None, description="Optional override to force specialist model")



class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=2)
    category: MemoryCategory = Field(default=MemoryCategory.GENERAL)
    tags: List[str] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    content: str = Field(..., min_length=2)
    category: Optional[MemoryCategory] = None
    tags: Optional[List[str]] = None


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page desktop web interface."""
    return HTMLResponse(content=JARVIS_HTML_PAGE)


@app.get("/healthz")
async def liveness_probe():
    """Liveness probe for cloud platforms, Docker, and Kubernetes."""
    return {"status": "ok", "service": "A.O.R.I.S."}


@app.get("/api/health")
async def health_check():
    """Readiness probe verifying application and persistent database connectivity."""
    srv = init_services()
    db_status = "healthy"
    try:
        conn = srv.memory_service.store._get_connection()
        conn.execute("SELECT 1;").fetchone()
    except Exception as e:
        db_status = f"unhealthy: {e}"
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "database": db_status},
        )
    return {
        "status": "healthy",
        "database": db_status,
        "service": "A.O.R.I.S.",
        "version": "0.1.0",
        "environment": srv.settings.jarvis_env,
    }


@app.get("/api/provider/health", response_model=ProviderHealth)
async def provider_health():
    """Verify connectivity and authentication for primary orchestrator provider."""
    srv = init_services()
    return await srv.orchestrator.provider.check_health()


@app.get("/api/providers/status")
async def get_multi_model_provider_statuses(
    token_authenticated: bool = Depends(verify_api_key),
):
    """Return real-time operational status for NVIDIA Nemotron and all 3 specialist providers."""
    srv = init_services()
    statuses = srv.orchestrator.model_registry.get_all_statuses()
    return {
        "orchestrator": srv.orchestrator.provider.provider_name,
        "orchestrator_model": srv.orchestrator.provider.default_model,
        "providers": {k: v.model_dump() for k, v in statuses.items()},
    }


@app.get("/api/downloads/{file_name}")
async def download_generated_file(
    file_name: str,
    token_authenticated: bool = Depends(verify_api_key),
):
    """Securely download generated PDFs and document artifacts with strict path traversal protection."""
    srv = init_services()
    cfg = srv.settings

    # Prevent path traversal tokens immediately
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name: directory traversal blocked.")

    clean_leaf = Path(file_name).name
    docs_root = cfg.aoris_docs_dir.resolve()
    target_path = (docs_root / clean_leaf).resolve()

    # Strict boundary check
    if not (target_path == docs_root or docs_root in target_path.parents):
        raise HTTPException(status_code=403, detail="Access denied: file resides outside allowed document directory.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{clean_leaf}' not found.")

    return FileResponse(
        path=target_path,
        media_type="application/pdf" if target_path.suffix.lower() == ".pdf" else "application/octet-stream",
        filename=clean_leaf,
    )


@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
async def chat_interaction(payload: ChatRequest):
    """Process incoming user prompt through memory, orchestrator, and specialist agents."""
    srv = init_services()
    task = await srv.orchestrator.process_request(
        user_request=payload.message,
        session_id=payload.session_id,
        user_confirmed=payload.user_confirmed,
        specialist_override=payload.specialist_override,
    )

    # Fetch loaded memories for inspection
    retrieved_memories = [
        srv.memory_service.get_memory(mid)
        for mid in task.retrieved_memory_ids
    ]
    valid_memories = [m.model_dump() for m in retrieved_memories if m]

    return {
        "task": task.model_dump(),
        "memories_used": valid_memories,
        "response": task.final_response,
        "verification": task.verification.model_dump() if task.verification else None,
        "is_live": srv.orchestrator.provider.is_live_provider,
        "provider_name": srv.orchestrator.provider.provider_name,
        "model": srv.orchestrator.provider.default_model,
        "orchestrator": task.metadata.get("orchestrator", "NVIDIA Nemotron"),
        "orchestrator_model": task.metadata.get("orchestrator_model", srv.orchestrator.provider.default_model),
        "specialist_used": task.metadata.get("specialist_used"),
        "specialist_model": task.metadata.get("specialist_model"),
        "synthesis_performed": task.metadata.get("synthesis_performed", False),
        "download_url": task.metadata.get("download_url"),
        "file_name": task.metadata.get("file_name"),
    }



@app.get("/api/tasks", response_model=List[TaskRecord], dependencies=[Depends(verify_api_key)])
async def list_tasks(session_id: Optional[str] = None, limit: int = 50):
    """List task execution records."""
    srv = init_services()
    return srv.orchestrator.list_tasks(session_id=session_id, limit=limit)


@app.get("/api/tasks/{task_id}", response_model=TaskRecord, dependencies=[Depends(verify_api_key)])
async def get_task(task_id: str):
    """Get single task details."""
    srv = init_services()
    task = srv.orchestrator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/api/tasks/{task_id}/approve", response_model=TaskRecord, dependencies=[Depends(verify_api_key)])
async def approve_task_step(task_id: str):
    """Approve a task paused in awaiting_confirmation state."""
    srv = init_services()
    try:
        return await srv.orchestrator.approve_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/tasks/{task_id}/deny", response_model=TaskRecord, dependencies=[Depends(verify_api_key)])
async def deny_task_step(task_id: str):
    """Deny execution for a task awaiting confirmation."""
    srv = init_services()
    try:
        return srv.orchestrator.deny_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskRecord, dependencies=[Depends(verify_api_key)])
async def cancel_task(task_id: str):
    """Cancel a task."""
    srv = init_services()
    try:
        return srv.orchestrator.cancel_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Memory Management Endpoints ---

@app.get("/api/memories", dependencies=[Depends(verify_api_key)])
async def get_memories(
    query: Optional[str] = None,
    category: Optional[MemoryCategory] = None,
    include_deleted: bool = False,
    limit: int = 50,
):
    """List or search persistent memories."""
    srv = init_services()
    if query and query.strip():
        items = srv.memory_service.retrieve_context(query=query.strip(), limit=limit)
        return [m.model_dump() for m in items]
    else:
        items = srv.memory_service.list_memories(
            category=category,
            include_deleted=include_deleted,
            limit=limit,
        )
        return [m.model_dump() for m in items]


@app.post("/api/memories", dependencies=[Depends(verify_api_key)])
async def create_memory(payload: MemoryCreateRequest):
    """Explicitly create a persistent memory."""
    srv = init_services()
    item = srv.memory_service.remember(
        content=payload.content,
        category=payload.category,
        tags=payload.tags,
        fact_status=FactStatus.USER_CLAIM,
    )
    return item.model_dump()


@app.put("/api/memories/{memory_id}", dependencies=[Depends(verify_api_key)])
async def update_memory(memory_id: str, payload: MemoryUpdateRequest):
    """Update an existing persistent memory."""
    srv = init_services()
    try:
        updated = srv.memory_service.update_memory(
            memory_id=memory_id,
            new_content=payload.content,
            new_category=payload.category,
            new_tags=payload.tags,
        )
        return updated.model_dump()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/memories/{memory_id}", dependencies=[Depends(verify_api_key)])
async def delete_memory(memory_id: str, soft: bool = Query(True)):
    """Delete a memory (soft-delete or permanent purge)."""
    srv = init_services()
    success = srv.memory_service.delete_memory(memory_id=memory_id, soft=soft)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or deletion failed")
    return {"status": "deleted", "memory_id": memory_id, "soft": soft}


@app.get("/api/settings")
async def get_app_settings():
    """Retrieve sanitized settings."""
    init_services()
    cfg = get_settings()
    return {
        "jarvis_env": cfg.jarvis_env,
        "jarvis_data_dir": str(cfg.jarvis_data_dir),
        "jarvis_db_path": str(cfg.jarvis_db_path),
        "jarvis_memory_max_retrieval_results": cfg.jarvis_memory_max_retrieval_results,
        "jarvis_memory_similarity_threshold": cfg.jarvis_memory_similarity_threshold,
        "auth_required": bool(cfg.jarvis_api_key),
        "rate_limit_per_minute": cfg.jarvis_rate_limit_per_minute,
        "max_request_size_bytes": cfg.jarvis_max_request_size_bytes,
        "cors_origins": cfg.jarvis_cors_origins,
        "nvidia_configured": bool(cfg.nvidia_api_key),
        "nvidia_base_url": cfg.nvidia_base_url,
        "nvidia_default_model": cfg.nvidia_default_model,
        "nebius_configured": bool(cfg.nebius_api_key),
        "nebius_base_url": cfg.nebius_base_url,
        "nebius_default_model": cfg.nebius_default_model,
        "openai_configured": bool(cfg.openai_api_key),
        "openai_model": cfg.openai_model,
        "anthropic_configured": bool(cfg.anthropic_api_key),
        "anthropic_model": cfg.anthropic_model,
        "google_configured": bool(cfg.google_api_key),
        "gemini_model": cfg.gemini_model,
    }

