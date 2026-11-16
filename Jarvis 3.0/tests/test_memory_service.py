"""Unit tests for high-level MemoryService workflows."""

import pytest
from jarvis.core.enums import FactStatus, MemoryCategory
from jarvis.memory.models import MemoryItem
from jarvis.memory.service import MemoryService


def test_remember_and_retrieve_context(memory_service: MemoryService):
    """Verify that explicitly remembered items are retrievable by relevant queries."""
    memory_service.remember(
        content="User prefers dark theme for all developer interfaces",
        category=MemoryCategory.PREFERENCE,
        tags=["theme", "dark", "ui"],
    )

    results = memory_service.retrieve_context(query="what theme does the user prefer?")
    assert len(results) > 0
    assert "dark theme" in results[0].content


def test_ingest_interaction_extracts_memory(memory_service: MemoryService):
    """Verify that conversational turns automatically extract structured memories."""
    session_id = "test_sess_extract"
    saved = memory_service.ingest_interaction(
        session_id=session_id,
        user_message="Please remember that we are deploying on Nebius cloud.",
        assistant_response="Understood, I will keep in mind that we are deploying on Nebius cloud.",
    )

    assert len(saved) == 1
    assert "deploying on nebius cloud" in saved[0].content.lower()

    # Verify conversation history was recorded
    history = memory_service.get_conversation_history(session_id)
    assert len(history) == 2
    assert history[0].role.value == "user"
    assert history[1].role.value == "assistant"


def test_working_memory_lifecycle(memory_service: MemoryService):
    """Verify working memory session tracking."""
    session_id = "test_wm_1"
    wm = memory_service.get_working_memory(session_id)
    assert wm.session_id == session_id

    wm.current_task = "Implement persistent memory engine"
    wm.scratchpad["step"] = 1

    updated_wm = memory_service.get_working_memory(session_id)
    assert updated_wm.current_task == "Implement persistent memory engine"
    assert updated_wm.scratchpad["step"] == 1
