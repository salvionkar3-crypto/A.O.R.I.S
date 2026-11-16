"""Automated tests for Cloud Readiness, Container Probes, Persistence & Security."""

import os
from pathlib import Path
import httpx
import pytest

from jarvis.config import Settings, get_settings, reset_settings
from jarvis.core.enums import FactStatus, MemoryCategory
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.web.app import RateLimitMiddleware, app, init_services


@pytest.mark.asyncio
async def test_cloud_platform_env_resolution(monkeypatch):
    """Verify that Settings dynamically honors cloud PORT and HOST environment variables."""
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("HOST", "0.0.0.0")
    reset_settings()
    settings = get_settings()
    assert settings.jarvis_port == 9090
    assert settings.jarvis_host == "0.0.0.0"
    reset_settings()


@pytest.mark.asyncio
async def test_custom_persistent_data_dir_resolution(tmp_path: Path):
    """Verify that custom JARVIS_DATA_DIR automatically nests jarvis_db_path when not overridden."""
    custom_dir = tmp_path / "persistent_cloud_volume"
    settings = Settings(jarvis_data_dir=custom_dir)
    assert settings.jarvis_data_dir == custom_dir
    assert settings.jarvis_db_path == custom_dir / "jarvis.db"


@pytest.mark.asyncio
async def test_container_healthz_liveness_probe():
    """Verify standard /healthz endpoint for container liveness."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/healthz")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "service": "A.O.R.I.S."}


@pytest.mark.asyncio
async def test_api_health_readiness_probe_with_db():
    """Verify /api/health probes both application and database connectivity."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["database"] == "healthy"
        assert data["service"] == "A.O.R.I.S."


@pytest.mark.asyncio
async def test_api_key_authentication_gate(monkeypatch):
    """Verify that configuring JARVIS_API_KEY restricts protected routes and allows valid tokens."""
    monkeypatch.setenv("JARVIS_API_KEY", "prod-secret-token-xyz")
    reset_settings()
    # Re-initialize services with new settings
    init_services(force_reload=True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Public endpoints should remain accessible
        res_health = await client.get("/healthz")
        assert res_health.status_code == 200

        res_settings = await client.get("/api/settings")
        assert res_settings.status_code == 200
        assert res_settings.json()["auth_required"] is True

        # Unauthenticated request to protected route must be rejected (401)
        res_unauth = await client.post(
            "/api/chat",
            json={"message": "Hello without token"},
        )
        assert res_unauth.status_code == 401

        # Invalid token must be rejected (401)
        res_invalid = await client.post(
            "/api/chat",
            json={"message": "Hello with bad token"},
            headers={"X-API-Key": "wrong-token"},
        )
        assert res_invalid.status_code == 401

        # Valid X-API-Key token must succeed (200)
        res_valid_header = await client.post(
            "/api/chat",
            json={"message": "Hello with valid token"},
            headers={"X-API-Key": "prod-secret-token-xyz"},
        )
        assert res_valid_header.status_code == 200

        # Valid Bearer token must succeed (200)
        res_valid_bearer = await client.post(
            "/api/chat",
            json={"message": "Hello with bearer token"},
            headers={"Authorization": "Bearer prod-secret-token-xyz"},
        )
        assert res_valid_bearer.status_code == 200

    # Clean up settings
    monkeypatch.delenv("JARVIS_API_KEY", raising=False)
    reset_settings()
    init_services(force_reload=True)


@pytest.mark.asyncio
async def test_request_payload_size_limit(monkeypatch):
    """Verify that requests exceeding jarvis_max_request_size_bytes return 413 Payload Too Large."""
    # Temporarily set max request size to 100 bytes
    monkeypatch.setenv("JARVIS_MAX_REQUEST_SIZE_BYTES", "100")
    reset_settings()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 150-byte payload should be rejected with 413
        oversized_data = {"message": "A" * 150}
        res = await client.post("/api/chat", json=oversized_data)
        assert res.status_code == 413

    monkeypatch.delenv("JARVIS_MAX_REQUEST_SIZE_BYTES", raising=False)
    reset_settings()


@pytest.mark.asyncio
async def test_rate_limiting_middleware(monkeypatch):
    """Verify that rapid requests exceeding rate limit trigger HTTP 429."""
    RateLimitMiddleware.reset()
    # Set limit to 2 requests per minute
    monkeypatch.setenv("JARVIS_RATE_LIMIT_PER_MINUTE", "2")
    reset_settings()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Request 1: OK
        res1 = await client.get("/api/settings")
        assert res1.status_code == 200

        # Request 2: OK
        res2 = await client.get("/api/settings")
        assert res2.status_code == 200

        # Request 3: Rate limited (429)
        res3 = await client.get("/api/settings")
        assert res3.status_code == 429
        assert "Retry-After" in res3.headers

    RateLimitMiddleware.reset()
    monkeypatch.delenv("JARVIS_RATE_LIMIT_PER_MINUTE", raising=False)
    reset_settings()



@pytest.mark.asyncio
async def test_persistent_memory_across_application_restarts(tmp_path: Path):
    """Verify persistent memory survives complete application stop, restart, and reconnection."""
    storage_dir = tmp_path / "persistent_mount"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_file = storage_dir / "jarvis.db"

    # Session 1: Store memory in persistent SQLite storage
    store1 = SQLiteMemoryStore(db_path=db_file)
    store1.initialize()
    mem_service1 = MemoryService(store=store1, settings=Settings(jarvis_data_dir=storage_dir, jarvis_db_path=db_file))
    mem_service1.initialize()

    mem_service1.remember(
        content="Production cloud deployment target is Nebius Cloud Run cluster.",
        category=MemoryCategory.PROJECT,
        tags=["deployment", "cloud"],
        fact_status=FactStatus.CONFIRMED_FACT,
    )
    store1.close()

    # Session 2: Fresh start pointing to same persistent volume mount
    assert db_file.exists()
    store2 = SQLiteMemoryStore(db_path=db_file)
    store2.initialize()
    mem_service2 = MemoryService(store=store2, settings=Settings(jarvis_data_dir=storage_dir, jarvis_db_path=db_file))
    mem_service2.initialize()

    # Retrieve memory using keyword search
    retrieved = mem_service2.retrieve_context(query="Nebius Cloud Run cluster", limit=5)
    assert len(retrieved) > 0
    assert "Nebius Cloud Run cluster" in retrieved[0].content
    assert retrieved[0].category == MemoryCategory.PROJECT
    store2.close()
