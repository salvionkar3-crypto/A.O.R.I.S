"""Conflict detection and deduplication logic for persistent memory."""

import re
from typing import List, Optional, Tuple
from jarvis.memory.base import BaseMemoryStore
from jarvis.memory.models import MemoryItem
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


def _extract_subject_prefix(text: str) -> Optional[str]:
    """Extract a subject/attribute prefix from memory content (e.g., 'favorite color')."""
    lower = text.lower().strip()
    patterns = [
        r"^(?:my\s+)?(favorite\s+[a-z0-9_-]+)\s+is\s+(.+)$",
        r"^(?:my\s+)?(preferred\s+[a-z0-9_-]+)\s+is\s+(.+)$",
        r"^(?:my\s+)?(preference\s+for\s+[a-z0-9_-]+)\s+is\s+(.+)$",
        r"^(?:the\s+)?(project\s+[a-z0-9_-]+)\s+is\s+(.+)$",
        r"^(?:the\s+)?(database|backend|framework|language)\s+is\s+(.+)$",
    ]
    for p in patterns:
        m = re.match(p, lower)
        if m:
            return m.group(1).strip()
    return None


class ConflictResolver:
    """Detects duplicates and contradictions, updating or superseding existing memories."""

    def __init__(self, store: BaseMemoryStore) -> None:
        self.store = store

    def resolve(self, candidate: MemoryItem) -> Tuple[Optional[MemoryItem], Optional[str]]:
        """
        Evaluate candidate against existing memories.
        
        Returns:
            (memory_to_save, superseded_memory_id)
            If candidate is an exact duplicate, returns (None, None).
            If candidate supersedes an existing memory, returns (candidate, old_id).
            If candidate is completely novel, returns (candidate, None).
        """
        # Fetch active memories in the same category
        existing_memories = self.store.list_memories(
            category=candidate.category,
            include_deleted=False,
            include_superseded=False,
            limit=100,
        )

        candidate_content_norm = candidate.content.strip().lower()
        candidate_subject = _extract_subject_prefix(candidate.content)

        for existing in existing_memories:
            existing_content_norm = existing.content.strip().lower()

            # 1. Exact Duplicate
            if candidate_content_norm == existing_content_norm:
                logger.info(f"Duplicate memory detected: '{candidate.content}'. Updating access stats.")
                self.store.record_access(existing.id)
                return None, None

            # 2. Contradiction / Superseding Check
            # Check if both have a matching subject prefix but different values
            existing_subject = _extract_subject_prefix(existing.content)
            if (
                candidate_subject
                and existing_subject
                and candidate_subject == existing_subject
            ):
                logger.info(
                    f"Contradiction detected on '{candidate_subject}'. "
                    f"Existing: '{existing.content}' superseded by New: '{candidate.content}'"
                )
                existing.superseded_by = candidate.id
                self.store.update_memory(existing)
                return candidate, existing.id

            # 3. High tag overlap and matching subject in unstructured content
            if candidate.tags and existing.tags:
                common_tags = set(candidate.tags).intersection(set(existing.tags))
                if len(common_tags) >= 2 and candidate_subject and candidate_subject in existing_content_norm:
                    logger.info(
                        f"Superseding memory based on high tag overlap {common_tags}. "
                        f"Old: {existing.id} -> New: {candidate.id}"
                    )
                    existing.superseded_by = candidate.id
                    self.store.update_memory(existing)
                    return candidate, existing.id

        return candidate, None
