"""Abstract base interface for memory storage backends."""

from abc import ABC, abstractmethod
from typing import List, Optional

from jarvis.core.enums import MemoryCategory
from jarvis.memory.models import (
    EpisodicRecord,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
)


class BaseMemoryStore(ABC):
    """Abstract interface defining required storage operations for JARVIS memory."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage engine, create tables, indices, and verify schema."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Safely flush and close any storage connections."""
        pass

    @abstractmethod
    def save_memory(self, item: MemoryItem) -> MemoryItem:
        """Persist a new memory item. Raises StorageError if operation fails."""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Fetch a single memory item by ID."""
        pass

    @abstractmethod
    def update_memory(self, item: MemoryItem) -> MemoryItem:
        """Update an existing memory item."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str, soft: bool = True) -> bool:
        """Delete a memory item (soft-delete marks is_deleted=1, hard deletes row)."""
        pass

    @abstractmethod
    def search_memories(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search and rank memories using keyword / FTS / vector search."""
        pass

    @abstractmethod
    def list_memories(
        self,
        category: Optional[MemoryCategory] = None,
        include_deleted: bool = False,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List stored memories with optional filtering."""
        pass

    @abstractmethod
    def record_access(self, memory_id: str) -> None:
        """Increment access count and touch last_accessed_at timestamp."""
        pass

    @abstractmethod
    def save_episode(self, episode: EpisodicRecord) -> EpisodicRecord:
        """Persist an episodic conversation/event record."""
        pass

    @abstractmethod
    def get_episodes(self, session_id: str, limit: int = 50) -> List[EpisodicRecord]:
        """Retrieve recent chronological episodes for a given session."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Wipe all memories and episodes (primarily for test teardown)."""
        pass
