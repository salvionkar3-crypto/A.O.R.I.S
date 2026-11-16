"""Unified MemoryService orchestrating ingestion, retrieval, and lifecycle management."""

from pathlib import Path
from typing import Dict, List, Optional

from jarvis.config import Settings, get_settings
from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType, Role
from jarvis.core.exceptions import MemoryNotFoundError, StorageError
from jarvis.core.types import generate_id, utc_now
from jarvis.memory.base import BaseMemoryStore
from jarvis.memory.conflict import ConflictResolver
from jarvis.memory.extractors import BaseMemoryExtractor, RuleBasedMemoryExtractor
from jarvis.memory.models import (
    EpisodicRecord,
    MemoryItem,
    MemoryQuery,
    WorkingMemoryState,
)
from jarvis.memory.sqlite_store import SQLiteMemoryStore
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


class MemoryService:
    """High-level service coordinating working, episodic, semantic, and procedural memory."""

    def __init__(
        self,
        store: Optional[BaseMemoryStore] = None,
        extractor: Optional[BaseMemoryExtractor] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or SQLiteMemoryStore(self.settings.jarvis_db_path)
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.resolver = ConflictResolver(self.store)
        self._working_memory: Dict[str, WorkingMemoryState] = {}

    def initialize(self) -> None:
        """Initialize the underlying storage engine."""
        self.store.initialize()

    def close(self) -> None:
        """Safely close storage connections."""
        self.store.close()

    def remember(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        fact_status: FactStatus = FactStatus.USER_CLAIM,
        tags: Optional[List[str]] = None,
        source_context: Optional[str] = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        """Explicitly record a persistent memory item with conflict resolution."""
        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            fact_status=fact_status,
            category=category,
            content=content.strip(),
            source_context=source_context,
            confidence=confidence,
            tags=tags or [],
        )

        item_to_save, superseded_id = self.resolver.resolve(item)
        if item_to_save is None:
            # Duplicate was found and access recorded
            existing = self.store.list_memories(category=category)
            for m in existing:
                if m.content.strip().lower() == content.strip().lower():
                    return m
            return item

        saved = self.store.save_memory(item_to_save)
        logger.info(
            f"Saved memory {saved.id} (Category: {saved.category.value})."
            + (f" Superseded {superseded_id}." if superseded_id else "")
        )
        return saved

    def ingest_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: Optional[str] = None,
    ) -> List[MemoryItem]:
        """Process a conversational turn: log episodes and extract new memories."""
        # 1. Save user episodic record
        user_ep = EpisodicRecord(
            session_id=session_id,
            role=Role.USER,
            content=user_message,
            source_context=f"session:{session_id}",
        )
        self.store.save_episode(user_ep)

        # 2. Save assistant episodic record if provided
        if assistant_response:
            asst_ep = EpisodicRecord(
                session_id=session_id,
                role=Role.ASSISTANT,
                content=assistant_response,
                source_context=f"session:{session_id}",
            )
            self.store.save_episode(asst_ep)

        # 3. Extract candidate memories from user message
        candidates = self.extractor.extract(
            user_message, source_context=f"session:{session_id}:{user_ep.id}"
        )
        saved_items: List[MemoryItem] = []

        for cand in candidates:
            item_to_save, superseded_id = self.resolver.resolve(cand)
            if item_to_save:
                saved = self.store.save_memory(item_to_save)
                saved_items.append(saved)
                logger.info(
                    f"Ingested memory '{saved.content}' [{saved.category.value}]."
                    + (f" Superseded: {superseded_id}" if superseded_id else "")
                )

        return saved_items

    def retrieve_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """
        Bounded context retrieval: searches memories relevant to query,
        ranks them, and updates access statistics.
        """
        max_limit = limit or self.settings.jarvis_memory_max_retrieval_results
        structured_query = MemoryQuery(
            query_text=query,
            limit=max_limit,
            include_deleted=False,
            include_superseded=False,
        )

        search_results = self.store.search_memories(structured_query)
        items: List[MemoryItem] = []

        for res in search_results:
            # Filter low relevance scores if configured
            if res.score >= self.settings.jarvis_memory_similarity_threshold:
                self.store.record_access(res.item.id)
                items.append(res.item)

        # Update working memory state if session_id provided
        if session_id:
            wm = self.get_working_memory(session_id)
            wm.retrieved_memories = items
            self._working_memory[session_id] = wm

        return items

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Fetch memory item by ID."""
        return self.store.get_memory(memory_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        new_category: Optional[MemoryCategory] = None,
        new_tags: Optional[List[str]] = None,
    ) -> MemoryItem:
        """Update an existing memory's content, category, or tags."""
        existing = self.store.get_memory(memory_id)
        if not existing:
            raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

        existing.content = new_content.strip()
        if new_category:
            existing.category = new_category
        if new_tags is not None:
            existing.tags = new_tags
        existing.updated_at = utc_now()

        return self.store.update_memory(existing)

    def delete_memory(self, memory_id: str, soft: Optional[bool] = None) -> bool:
        """Delete a memory item (defaults to soft delete as configured)."""
        use_soft = soft if soft is not None else self.settings.jarvis_enable_soft_delete
        return self.store.delete_memory(memory_id, soft=use_soft)

    def list_memories(
        self,
        category: Optional[MemoryCategory] = None,
        include_deleted: bool = False,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List active memories."""
        return self.store.list_memories(
            category=category,
            include_deleted=include_deleted,
            include_superseded=include_superseded,
            limit=limit,
        )

    def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> List[EpisodicRecord]:
        """Fetch chronological episode turns for a conversation session."""
        return self.store.get_episodes(session_id=session_id, limit=limit)

    def get_working_memory(self, session_id: str) -> WorkingMemoryState:
        """Retrieve current in-flight working memory state for a session."""
        if session_id not in self._working_memory:
            recent_episodes = self.get_conversation_history(session_id, limit=10)
            self._working_memory[session_id] = WorkingMemoryState(
                session_id=session_id,
                recent_episodes=recent_episodes,
            )
        return self._working_memory[session_id]

    def clear_all(self) -> None:
        """Clear all memories, episodes, and in-memory caches."""
        self._working_memory.clear()
        self.store.clear_all()
