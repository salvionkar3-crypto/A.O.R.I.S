"""Unit tests for SQLiteMemoryStore and FTS5 search."""

import pytest
from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType, Role
from jarvis.core.exceptions import MemoryNotFoundError, StorageError
from jarvis.memory.models import EpisodicRecord, MemoryItem, MemoryQuery
from jarvis.memory.sqlite_store import SQLiteMemoryStore


def test_save_and_retrieve_memory(sqlite_store: SQLiteMemoryStore):
    """Test saving and getting a memory item."""
    item = MemoryItem(
        content="The server runs on port 8080",
        category=MemoryCategory.PROJECT,
        tags=["server", "port", "network"],
    )
    saved = sqlite_store.save_memory(item)
    assert saved.id == item.id

    fetched = sqlite_store.get_memory(item.id)
    assert fetched is not None
    assert fetched.content == "The server runs on port 8080"
    assert fetched.category == MemoryCategory.PROJECT
    assert "port" in fetched.tags


def test_update_memory(sqlite_store: SQLiteMemoryStore):
    """Test updating existing memory attributes."""
    item = MemoryItem(
        content="Preferred IDE is VSCode",
        category=MemoryCategory.PREFERENCE,
    )
    sqlite_store.save_memory(item)

    item.content = "Preferred IDE is Antigravity"
    updated = sqlite_store.update_memory(item)
    assert updated.content == "Preferred IDE is Antigravity"

    fetched = sqlite_store.get_memory(item.id)
    assert fetched is not None
    assert fetched.content == "Preferred IDE is Antigravity"


def test_soft_and_hard_delete(sqlite_store: SQLiteMemoryStore):
    """Test soft deletion vs hard deletion."""
    item = MemoryItem(content="Temporary note to delete")
    sqlite_store.save_memory(item)

    # Soft delete
    deleted = sqlite_store.delete_memory(item.id, soft=True)
    assert deleted is True

    # Fetched item still exists in db but is marked is_deleted=True
    fetched = sqlite_store.get_memory(item.id)
    assert fetched is not None
    assert fetched.is_deleted is True

    # Search should exclude soft-deleted item by default
    res = sqlite_store.search_memories(MemoryQuery(query_text="Temporary note"))
    assert len(res) == 0

    # Hard delete
    hard_deleted = sqlite_store.delete_memory(item.id, soft=False)
    assert hard_deleted is True
    assert sqlite_store.get_memory(item.id) is None


def test_fts5_keyword_search(sqlite_store: SQLiteMemoryStore):
    """Test BM25 full-text keyword ranking in FTS5."""
    m1 = MemoryItem(
        content="NVIDIA Nemotron 4 340B is an advanced open model.",
        category=MemoryCategory.PROJECT,
        tags=["nvidia", "nemotron", "llm"],
    )
    m2 = MemoryItem(
        content="Python 3.11 introduced exceptional performance enhancements.",
        category=MemoryCategory.PROJECT,
        tags=["python", "performance"],
    )
    m3 = MemoryItem(
        content="Nebius AI Studio provides compute infrastructure for AI workloads.",
        category=MemoryCategory.PROJECT,
        tags=["nebius", "cloud"],
    )

    sqlite_store.save_memory(m1)
    sqlite_store.save_memory(m2)
    sqlite_store.save_memory(m3)

    # Search for nemotron
    results = sqlite_store.search_memories(MemoryQuery(query_text="nemotron"))
    assert len(results) > 0
    assert results[0].item.id == m1.id
    assert results[0].match_reason == "fts5_bm25_keyword_match"

    # Search for nebius compute
    results_nebius = sqlite_store.search_memories(MemoryQuery(query_text="nebius compute"))
    assert len(results_nebius) > 0
    assert results_nebius[0].item.id == m3.id


def test_episodes_logging_and_retrieval(sqlite_store: SQLiteMemoryStore):
    """Test saving and retrieving chronological episodic turns."""
    session_id = "sess_demo_1"
    ep1 = EpisodicRecord(session_id=session_id, role=Role.USER, content="Hello JARVIS")
    ep2 = EpisodicRecord(
        session_id=session_id,
        role=Role.ASSISTANT,
        content="Greetings. How can I assist you?",
    )
    sqlite_store.save_episode(ep1)
    sqlite_store.save_episode(ep2)

    history = sqlite_store.get_episodes(session_id=session_id)
    assert len(history) == 2
    assert history[0].content == "Hello JARVIS"
    assert history[1].role == Role.ASSISTANT
