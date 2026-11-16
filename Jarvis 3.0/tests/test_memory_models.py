"""Unit tests for memory data models and schema validation."""

import pytest
from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType, Role
from jarvis.memory.models import EpisodicRecord, MemoryItem, MemoryQuery


def test_memory_item_defaults():
    """Verify default fields and auto-generated IDs for MemoryItem."""
    item = MemoryItem(content="The user's favorite framework is FastAPI.")
    assert item.id.startswith("mem_")
    assert item.memory_type == MemoryType.SEMANTIC
    assert item.fact_status == FactStatus.USER_CLAIM
    assert item.category == MemoryCategory.GENERAL
    assert item.confidence == 1.0
    assert item.access_count == 0
    assert item.is_deleted is False
    assert item.superseded_by is None
    assert item.created_at is not None


def test_episodic_record_defaults():
    """Verify EpisodicRecord structure."""
    record = EpisodicRecord(
        session_id="session_101",
        role=Role.USER,
        content="Please remember my preferred language is Python.",
    )
    assert record.id.startswith("ep_")
    assert record.session_id == "session_101"
    assert record.role == Role.USER
    assert record.content == "Please remember my preferred language is Python."


def test_memory_query_validation():
    """Verify query model validation."""
    query = MemoryQuery(query_text="FastAPI framework", limit=10)
    assert query.query_text == "FastAPI framework"
    assert query.limit == 10
    assert query.include_deleted is False
