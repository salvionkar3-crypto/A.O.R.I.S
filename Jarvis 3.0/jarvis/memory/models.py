"""Data models for JARVIS Memory and Knowledge Layer."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field

from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType, Role
from jarvis.core.types import JarvisBaseModel, generate_id, utc_now


class MemoryItem(JarvisBaseModel):
    """A persistent unit of semantic or procedural memory."""

    id: str = Field(default_factory=lambda: generate_id("mem"))
    memory_type: MemoryType = Field(
        default=MemoryType.SEMANTIC, description="Logical memory dimension"
    )
    fact_status: FactStatus = Field(
        default=FactStatus.USER_CLAIM,
        description="Epistemic status of the memory",
    )
    category: MemoryCategory = Field(
        default=MemoryCategory.GENERAL,
        description="Categorical domain of this memory",
    )
    content: str = Field(..., description="The factual or instructional statement")
    source_context: Optional[str] = Field(
        default=None, description="Provenance reference (e.g. session_id:msg_id)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]"
    )
    tags: List[str] = Field(
        default_factory=list, description="Keywords or entity tags"
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    is_deleted: bool = False
    superseded_by: Optional[str] = Field(
        default=None, description="ID of memory that supersedes this entry"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary structured metadata"
    )


class EpisodicRecord(JarvisBaseModel):
    """An interaction turn or executed event in conversational history."""

    id: str = Field(default_factory=lambda: generate_id("ep"))
    session_id: str = Field(..., description="Unique conversation session ID")
    role: Role = Field(..., description="Participant role")
    content: str = Field(..., description="Message text or tool output")
    source_context: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryState(JarvisBaseModel):
    """Active in-flight context and scratchpad for the current session."""

    session_id: str
    current_task: Optional[str] = None
    active_plan: Optional[Dict[str, Any]] = None
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    retrieved_memories: List[MemoryItem] = Field(default_factory=list)
    recent_episodes: List[EpisodicRecord] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryQuery(JarvisBaseModel):
    """Structured query for searching persistent memories."""

    query_text: str = Field(..., description="Free-text search terms or natural query")
    memory_type: Optional[MemoryType] = None
    categories: Optional[List[MemoryCategory]] = None
    tags: Optional[List[str]] = None
    min_confidence: float = 0.0
    limit: int = 5
    include_deleted: bool = False
    include_superseded: bool = False


class MemorySearchResult(JarvisBaseModel):
    """Ranked memory result with relevance scoring and match explanation."""

    item: MemoryItem
    score: float = Field(..., description="Relevance rank score")
    match_reason: str = Field(default="keyword_match")
