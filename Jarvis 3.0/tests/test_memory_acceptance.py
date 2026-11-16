"""Explicit Acceptance Test Suite for Milestone 1: Persistent Memory Foundation.

Verifies all 8 required hackathon criteria:
1. Save a memory and restart the application.
2. Retrieve the saved memory in a new session.
3. Update an existing memory without uncontrolled duplication.
4. Search for a relevant memory using different wording where supported.
5. Correct or delete a memory.
6. Preserve provenance and timestamps.
7. Confirm that unrelated memories are not injected into a request.
8. Simulate a storage failure and verify that the system reports it honestly.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from jarvis.config import Settings
from jarvis.core.enums import FactStatus, MemoryCategory
from jarvis.core.exceptions import StorageError
from jarvis.memory.extractors import RuleBasedMemoryExtractor
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore


def test_acceptance_1_restart_application(temp_db_path: Path):
    """1. Save a memory and restart the application."""
    # Instance 1: Save memory and shut down
    service1 = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(jarvis_db_path=temp_db_path),
    )
    service1.initialize()
    saved = service1.remember(
        content="The primary project architecture is a multi-agent orchestrator.",
        category=MemoryCategory.PROJECT,
        tags=["architecture", "orchestrator"],
    )
    saved_id = saved.id
    service1.close()

    # Instance 2: Simulate application restart with a fresh service instance
    service2 = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(jarvis_db_path=temp_db_path),
    )
    service2.initialize()
    recovered = service2.get_memory(saved_id)

    assert recovered is not None
    assert recovered.id == saved_id
    assert recovered.content == "The primary project architecture is a multi-agent orchestrator."
    service2.close()


def test_acceptance_2_retrieve_in_new_session(temp_db_path: Path):
    """2. Retrieve the saved memory in a new session."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(
            jarvis_db_path=temp_db_path,
            jarvis_memory_similarity_threshold=0.1,
        ),
    )
    service.initialize()

    # Interaction in Session 1
    service.ingest_interaction(
        session_id="session_alpha",
        user_message="Remember that my favorite editor is Neovim.",
        assistant_response="I will remember that your favorite editor is Neovim.",
    )

    # In Session 2, query context for editor preference
    results = service.retrieve_context(
        query="which editor does the user prefer?",
        session_id="session_beta",
    )

    assert len(results) > 0
    assert any("neovim" in m.content.lower() for m in results)

    # Verify working memory for session_beta captured the retrieved memory
    wm = service.get_working_memory("session_beta")
    assert len(wm.retrieved_memories) > 0
    service.close()


def test_acceptance_3_update_without_uncontrolled_duplication(temp_db_path: Path):
    """3. Update an existing memory without uncontrolled duplication."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(jarvis_db_path=temp_db_path),
    )
    service.initialize()

    # User declares initial preference
    initial_mem = service.remember(
        content="My preferred language is Python",
        category=MemoryCategory.PREFERENCE,
    )
    assert initial_mem is not None

    # User later updates preference
    updated_mem = service.remember(
        content="My preferred language is Rust",
        category=MemoryCategory.PREFERENCE,
    )
    assert updated_mem is not None
    assert updated_mem.id != initial_mem.id

    # Check store state: old memory must be marked as superseded
    old_record = service.get_memory(initial_mem.id)
    assert old_record is not None
    assert old_record.superseded_by == updated_mem.id

    # Active memory query should only return the new memory, not duplicate/contradictory ones
    active_memories = service.list_memories(
        category=MemoryCategory.PREFERENCE,
        include_superseded=False,
    )
    assert len(active_memories) == 1
    assert active_memories[0].content == "My preferred language is Rust"
    service.close()


def test_acceptance_4_search_with_different_wording(temp_db_path: Path):
    """4. Search for a relevant memory using different wording where supported."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(
            jarvis_db_path=temp_db_path,
            jarvis_memory_similarity_threshold=0.1,
        ),
    )
    service.initialize()

    service.remember(
        content="We use NVIDIA DGX SuperPOD for training large scale reasoning models.",
        category=MemoryCategory.PROJECT,
        tags=["nvidia", "superpod", "training"],
    )

    # Search using different phrasing and keywords
    results = service.retrieve_context(query="what infrastructure is used for model training?")
    assert len(results) > 0
    assert "NVIDIA DGX SuperPOD" in results[0].content
    service.close()


def test_acceptance_5_correct_or_delete_memory(temp_db_path: Path):
    """5. Correct or delete a memory."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(jarvis_db_path=temp_db_path),
    )
    service.initialize()

    item = service.remember(
        content="Meeting scheduled every Tuesday at 10 AM",
        category=MemoryCategory.GENERAL,
    )

    # 1. User corrects the memory
    corrected = service.update_memory(
        memory_id=item.id,
        new_content="Meeting scheduled every Wednesday at 11 AM",
    )
    assert corrected.content == "Meeting scheduled every Wednesday at 11 AM"

    # Verify search finds corrected version
    search_res = service.retrieve_context(query="Wednesday meeting")
    assert len(search_res) > 0
    assert "Wednesday" in search_res[0].content

    # 2. User deletes the memory
    deleted = service.delete_memory(item.id)
    assert deleted is True

    # Search should no longer return it
    post_delete_search = service.retrieve_context(query="Wednesday meeting")
    assert len(post_delete_search) == 0
    service.close()


def test_acceptance_6_provenance_and_timestamps(temp_db_path: Path):
    """6. Preserve provenance and timestamps."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(jarvis_db_path=temp_db_path),
    )
    service.initialize()

    session_id = "session_audit_test"
    saved = service.ingest_interaction(
        session_id=session_id,
        user_message="My preference is concise bullet-point summaries.",
    )

    assert len(saved) == 1
    item = saved[0]
    assert item.created_at is not None
    assert item.updated_at is not None
    assert item.source_context is not None
    assert session_id in item.source_context
    assert item.fact_status == FactStatus.USER_CLAIM
    assert item.confidence == 1.0
    service.close()


def test_acceptance_7_unrelated_memories_not_injected(temp_db_path: Path):
    """7. Confirm that unrelated memories are not injected into a request."""
    service = MemoryService(
        store=SQLiteMemoryStore(temp_db_path),
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(
            jarvis_db_path=temp_db_path,
            jarvis_memory_similarity_threshold=0.3,
        ),
    )
    service.initialize()

    # Store a personal dietary preference
    service.remember(
        content="User drinks black coffee with zero sugar in the morning.",
        category=MemoryCategory.PREFERENCE,
        tags=["coffee", "drink", "morning"],
    )

    # Query completely unrelated technical topic
    results = service.retrieve_context(
        query="Explain database schema migrations and rollback strategies."
    )

    # The unrelated memory must NOT be returned in the context
    assert len(results) == 0
    service.close()


def test_acceptance_8_storage_failure_honest_reporting():
    """8. Simulate a storage failure and verify that the system reports it honestly."""
    # Point store to an invalid/illegal path that causes an OS/SQLite error
    invalid_path = Path("Z:\\illegal_drive_non_existent\\invalid.db")
    broken_store = SQLiteMemoryStore(db_path=invalid_path)

    with pytest.raises(StorageError) as exc_info:
        broken_store.initialize()

    # The system must report the failure honestly without pretending success
    assert "Failed to" in str(exc_info.value)
