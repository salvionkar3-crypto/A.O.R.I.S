"""Pytest configuration and shared test fixtures for JARVIS."""

import os
from pathlib import Path
import tempfile
from typing import Generator
import pytest

from jarvis.config import Settings, reset_settings
from jarvis.memory.extractors import RuleBasedMemoryExtractor
from jarvis.memory.service import MemoryService
from jarvis.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture(autouse=True)
def clean_settings() -> Generator[None, None, None]:
    """Ensure clean settings before and after each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary file path for SQLite database testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "test_jarvis.db"
        yield db_file


@pytest.fixture
def sqlite_store(temp_db_path: Path) -> Generator[SQLiteMemoryStore, None, None]:
    """Provide an initialized SQLiteMemoryStore bound to a temporary database."""
    store = SQLiteMemoryStore(db_path=temp_db_path)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def in_memory_store() -> Generator[SQLiteMemoryStore, None, None]:
    """Provide an initialized in-memory SQLite store."""
    store = SQLiteMemoryStore(db_path=":memory:")
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def memory_service(temp_db_path: Path) -> Generator[MemoryService, None, None]:
    """Provide an initialized MemoryService bound to a temporary database."""
    store = SQLiteMemoryStore(db_path=temp_db_path)
    store.initialize()
    service = MemoryService(
        store=store,
        extractor=RuleBasedMemoryExtractor(),
        settings=Settings(
            jarvis_db_path=temp_db_path,
            jarvis_memory_similarity_threshold=0.1,
            jarvis_memory_max_retrieval_results=5,
        ),
    )
    yield service
    service.close()
