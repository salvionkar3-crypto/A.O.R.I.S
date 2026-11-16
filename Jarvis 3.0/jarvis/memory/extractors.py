"""Memory extraction mechanisms (deterministic rule-based and LLM-assisted)."""

from abc import ABC, abstractmethod
import re
from typing import List, Optional, Tuple

from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType
from jarvis.memory.models import MemoryItem


class BaseMemoryExtractor(ABC):
    """Abstract interface for extracting candidate memories from conversation text."""

    @abstractmethod
    def extract(
        self, text: str, source_context: Optional[str] = None
    ) -> List[MemoryItem]:
        """Parse text and extract structured memory items."""
        pass


class RuleBasedMemoryExtractor(BaseMemoryExtractor):
    """Deterministic, pattern-matching memory extractor for explicit statements."""

    # Patterns matching user preferences
    PREFERENCE_PATTERNS: List[Tuple[re.Pattern, MemoryCategory]] = [
        (
            re.compile(
                r"(?i)\b(?:please\s+)?remember\s+(?:that\s+)?(.+)",
                re.DOTALL,
            ),
            MemoryCategory.GENERAL,
        ),
        (
            re.compile(
                r"(?i)\bmy\s+preference\s+is\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.PREFERENCE,
        ),
        (
            re.compile(
                r"(?i)\bi\s+prefer\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.PREFERENCE,
        ),
        (
            re.compile(
                r"(?i)\bmy\s+favorite\s+([a-zA-Z0-9_\s]+?)\s+is\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.PREFERENCE,
        ),
        (
            re.compile(
                r"(?i)\bi\s+(?:like|love)\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.PREFERENCE,
        ),
        (
            re.compile(
                r"(?i)\b(?:always|never)\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.RULE,
        ),
        (
            re.compile(
                r"(?i)\bthe\s+project\s+(?:is|name\s+is|architecture\s+is)\s+(.+)",
                re.DOTALL,
            ),
            MemoryCategory.PROJECT,
        ),
    ]

    def _extract_tags(self, content: str) -> List[str]:
        """Extract key entities and alphanumeric words as tags."""
        words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", content.lower())
        stop_words = {
            "that", "this", "with", "from", "have", "been", "were", "what",
            "when", "where", "which", "your", "them", "some", "more", "like",
            "just", "into", "also", "about", "please", "remember", "prefer",
        }
        filtered = [w for w in words if w not in stop_words]
        return list(dict.fromkeys(filtered))[:6]

    def extract(
        self, text: str, source_context: Optional[str] = None
    ) -> List[MemoryItem]:
        candidates: List[MemoryItem] = []
        cleaned_text = text.strip()

        for pattern, category in self.PREFERENCE_PATTERNS:
            match = pattern.search(cleaned_text)
            if match:
                extracted_content = match.group(1).strip().rstrip(".").strip()
                if not extracted_content:
                    continue

                # Refine category for explicit "remember that my favorite..."
                final_category = category
                lower_content = extracted_content.lower()
                if any(k in lower_content for k in ["prefer", "favorite", "like"]):
                    final_category = MemoryCategory.PREFERENCE
                elif any(k in lower_content for k in ["project", "codebase", "repo", "architecture"]):
                    final_category = MemoryCategory.PROJECT
                elif any(k in lower_content for k in ["rule", "never", "always", "must not"]):
                    final_category = MemoryCategory.RULE

                tags = self._extract_tags(extracted_content)

                candidates.append(
                    MemoryItem(
                        memory_type=MemoryType.SEMANTIC,
                        fact_status=FactStatus.USER_CLAIM,
                        category=final_category,
                        content=extracted_content,
                        source_context=source_context,
                        confidence=1.0,
                        tags=tags,
                    )
                )
                # Break on first strong pattern match to avoid duplicates from multiple regexes
                break

        return candidates
