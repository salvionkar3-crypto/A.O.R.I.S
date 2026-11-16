"""End-to-End application integration and Web API test suite."""

import httpx
import pytest

from jarvis.config import Settings, reset_settings
from jarvis.core.enums import TaskStatus
from jarvis.web.app import app


@pytest.mark.asyncio
async def test_web_ui_dashboard_endpoint():
    """Verify root / endpoint serves the full interactive HTML dashboard."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "A.O.R.I.S." in res.text
        assert "Memory Matrix" in res.text
        assert "Task Ledger" in res.text


@pytest.mark.asyncio
async def test_end_to_end_chat_and_memory_retrieval_flow():
    """
    Test complete lifecycle:
    1. User states preference in Session 1
    2. Memory is stored and indexed in SQLite FTS5
    3. User queries in Session 2
    4. Relevant memory is retrieved and incorporated into response
    5. Task ledger records execution with verification report
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: Tell JARVIS a project preference
        res1 = await client.post(
            "/api/chat",
            json={
                "message": "Remember that our primary cloud provider is Nebius AI Studio.",
                "session_id": "session_e2e_1",
            },
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["task"]["state"] == "completed"

        # Step 2: Query memory in a fresh session
        res2 = await client.post(
            "/api/chat",
            json={
                "message": "Which cloud provider are we using?",
                "session_id": "session_e2e_2",
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["task"]["state"] == "completed"
        assert len(data2["memories_used"]) > 0
        assert "Nebius" in data2["memories_used"][0]["content"]

        # Step 3: Inspect task ledger
        res3 = await client.get("/api/tasks")
        assert res3.status_code == 200
        tasks = res3.json()
        assert len(tasks) >= 2


@pytest.mark.asyncio
async def test_web_memory_crud_api():
    """Verify REST API for managing persistent memories."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create
        create_res = await client.post(
            "/api/memories",
            json={
                "content": "Temporary test note for API validation",
                "category": "project",
                "tags": ["test", "api"],
            },
        )
        assert create_res.status_code == 200
        created_mem = create_res.json()
        mem_id = created_mem["id"]

        # 2. Search
        search_res = await client.get("/api/memories?query=validation")
        assert search_res.status_code == 200
        results = search_res.json()
        assert any(m["id"] == mem_id for m in results)

        # 3. Update
        update_res = await client.put(
            f"/api/memories/{mem_id}",
            json={"content": "Updated test note for API validation"},
        )
        assert update_res.status_code == 200
        assert update_res.json()["content"] == "Updated test note for API validation"

        # 4. Soft Delete
        del_res = await client.delete(f"/api/memories/{mem_id}?soft=true")
        assert del_res.status_code == 200

        # Search should no longer return soft-deleted item
        search_after = await client.get("/api/memories?query=validation")
        assert not any(m["id"] == mem_id for m in search_after.json())


@pytest.mark.asyncio
async def test_provider_health_endpoint():
    """Verify /api/provider/health reports provider status honestly."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/provider/health")
        assert res.status_code == 200
        health = res.json()
        assert "provider_name" in health
        assert "status" in health
        assert "is_live" in health
