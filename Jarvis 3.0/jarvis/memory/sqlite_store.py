"""SQLite implementation of BaseMemoryStore with FTS5 keyword search."""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from jarvis.core.enums import FactStatus, MemoryCategory, MemoryType, Role
from jarvis.core.exceptions import MemoryNotFoundError, StorageError
from jarvis.core.types import utc_now
from jarvis.memory.base import BaseMemoryStore
from jarvis.memory.models import (
    EpisodicRecord,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
)
from jarvis.utils.logging import setup_logger

logger = setup_logger(__name__)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize user query for safe FTS5 query execution."""
    # Strip any characters that could break FTS5 syntax
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    if not tokens:
        return ""
    # Search each token with prefix matching
    return " OR ".join(f'"{t}"*' for t in tokens)


class SQLiteMemoryStore(BaseMemoryStore):
    """Production SQLite storage engine with FTS5 ranking and transactional guarantees."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                if self.db_path != ":memory:":
                    parent = Path(self.db_path).parent
                    parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=20.0,
                    check_same_thread=False,
                    detect_types=sqlite3.PARSE_DECLTYPES,
                )
                conn.row_factory = sqlite3.Row
                # Optimize SQLite for concurrent reads and writes
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                self._conn = conn
            except Exception as e:
                raise StorageError(f"Failed to connect to SQLite database at '{self.db_path}': {e}") from e
        return self._conn

    def initialize(self) -> None:
        """Create tables, FTS5 virtual tables, and indices."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        fact_status TEXT NOT NULL,
                        category TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_context TEXT,
                        confidence REAL NOT NULL,
                        tags TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_accessed_at TEXT,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        is_deleted INTEGER NOT NULL DEFAULT 0,
                        superseded_by TEXT,
                        metadata TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_category 
                    ON memories(category);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_is_deleted 
                    ON memories(is_deleted);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_superseded 
                    ON memories(superseded_by);
                    """
                )

                # Full-Text Search virtual table with porter stemming
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id UNINDEXED,
                        content,
                        tags,
                        category,
                        tokenize = 'porter unicode61'
                    );
                    """
                )

                # Episodic history table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodes (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_context TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_episodes_session 
                    ON episodes(session_id, timestamp);
                    """
                )
            logger.info("SQLite memory store initialized successfully.")
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite schema: {e}") from e

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"Error while closing SQLite connection: {e}")
            finally:
                self._conn = None

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            fact_status=FactStatus(row["fact_status"]),
            category=MemoryCategory(row["category"]),
            content=row["content"],
            source_context=row["source_context"],
            confidence=float(row["confidence"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
            access_count=int(row["access_count"]),
            is_deleted=bool(row["is_deleted"]),
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def save_memory(self, item: MemoryItem) -> MemoryItem:
        conn = self._get_connection()
        tags_json = json.dumps(item.tags)
        meta_json = json.dumps(item.metadata)
        tags_text = " ".join(item.tags)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, memory_type, fact_status, category, content,
                        source_context, confidence, tags, created_at, updated_at,
                        last_accessed_at, access_count, is_deleted, superseded_by, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        item.id,
                        item.memory_type.value,
                        item.fact_status.value,
                        item.category.value,
                        item.content,
                        item.source_context,
                        item.confidence,
                        tags_json,
                        item.created_at.isoformat(),
                        item.updated_at.isoformat(),
                        item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                        item.access_count,
                        1 if item.is_deleted else 0,
                        item.superseded_by,
                        meta_json,
                    ),
                )
                # Keep FTS in sync
                conn.execute(
                    """
                    INSERT INTO memories_fts (id, content, tags, category)
                    VALUES (?, ?, ?, ?);
                    """,
                    (item.id, item.content, tags_text, item.category.value),
                )
            return item
        except Exception as e:
            raise StorageError(f"Failed to persist memory {item.id}: {e}") from e

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?;", (memory_id,)
            )
            row = cursor.fetchone()
            return self._row_to_memory(row) if row else None
        except Exception as e:
            raise StorageError(f"Failed to retrieve memory {memory_id}: {e}") from e

    def update_memory(self, item: MemoryItem) -> MemoryItem:
        conn = self._get_connection()
        item.updated_at = utc_now()
        tags_json = json.dumps(item.tags)
        meta_json = json.dumps(item.metadata)
        tags_text = " ".join(item.tags)
        try:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE memories SET
                        memory_type = ?,
                        fact_status = ?,
                        category = ?,
                        content = ?,
                        source_context = ?,
                        confidence = ?,
                        tags = ?,
                        updated_at = ?,
                        last_accessed_at = ?,
                        access_count = ?,
                        is_deleted = ?,
                        superseded_by = ?,
                        metadata = ?
                    WHERE id = ?;
                    """,
                    (
                        item.memory_type.value,
                        item.fact_status.value,
                        item.category.value,
                        item.content,
                        item.source_context,
                        item.confidence,
                        tags_json,
                        item.updated_at.isoformat(),
                        item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                        item.access_count,
                        1 if item.is_deleted else 0,
                        item.superseded_by,
                        meta_json,
                        item.id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise MemoryNotFoundError(f"Memory with ID '{item.id}' does not exist.")

                # Re-index in FTS
                conn.execute("DELETE FROM memories_fts WHERE id = ?;", (item.id,))
                conn.execute(
                    """
                    INSERT INTO memories_fts (id, content, tags, category)
                    VALUES (?, ?, ?, ?);
                    """,
                    (item.id, item.content, tags_text, item.category.value),
                )
            return item
        except MemoryNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update memory {item.id}: {e}") from e

    def delete_memory(self, memory_id: str, soft: bool = True) -> bool:
        conn = self._get_connection()
        try:
            with conn:
                if soft:
                    cursor = conn.execute(
                        """
                        UPDATE memories 
                        SET is_deleted = 1, updated_at = ? 
                        WHERE id = ?;
                        """,
                        (utc_now().isoformat(), memory_id),
                    )
                    # Remove from search index so soft-deleted memories aren't returned in searches
                    conn.execute("DELETE FROM memories_fts WHERE id = ?;", (memory_id,))
                    return cursor.rowcount > 0
                else:
                    cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
                    conn.execute("DELETE FROM memories_fts WHERE id = ?;", (memory_id,))
                    return cursor.rowcount > 0
        except Exception as e:
            raise StorageError(f"Failed to delete memory {memory_id}: {e}") from e

    def search_memories(self, query: MemoryQuery) -> List[MemorySearchResult]:
        conn = self._get_connection()
        sanitized_fts = _sanitize_fts_query(query.query_text)

        results: List[MemorySearchResult] = []
        try:
            if sanitized_fts:
                # FTS5 search with BM25 score
                sql = """
                    SELECT m.*, bm25(memories_fts) AS bm25_rank
                    FROM memories_fts f
                    JOIN memories m ON f.id = m.id
                    WHERE memories_fts MATCH ?
                """
                params: List[Any] = [sanitized_fts]

                if not query.include_deleted:
                    sql += " AND m.is_deleted = 0"
                if not query.include_superseded:
                    sql += " AND m.superseded_by IS NULL"
                if query.memory_type:
                    sql += " AND m.memory_type = ?"
                    params.append(query.memory_type.value)
                if query.categories:
                    cat_placeholders = ",".join("?" for _ in query.categories)
                    sql += f" AND m.category IN ({cat_placeholders})"
                    params.extend([c.value for c in query.categories])
                if query.min_confidence > 0.0:
                    sql += " AND m.confidence >= ?"
                    params.append(query.min_confidence)

                # Order by BM25 rank (more negative is better match)
                sql += " ORDER BY bm25_rank ASC LIMIT ?;"
                params.append(query.limit)

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                for row in rows:
                    item = self._row_to_memory(row)
                    bm25_score = float(row["bm25_rank"])
                    # Transform negative bm25 score into a positive relevance score
                    norm_score = max(0.1, 1.0 / (1.0 + abs(bm25_score)))
                    results.append(
                        MemorySearchResult(
                            item=item,
                            score=round(norm_score, 4),
                            match_reason="fts5_bm25_keyword_match",
                        )
                    )

            # Fallback to LIKE substring search if FTS returned nothing or query had no tokens
            if not results:
                sql = "SELECT * FROM memories WHERE is_deleted = 0"
                params = []
                if not query.include_superseded:
                    sql += " AND superseded_by IS NULL"
                if query.query_text.strip():
                    sql += " AND (content LIKE ? OR tags LIKE ?)"
                    wildcard = f"%{query.query_text.strip()}%"
                    params.extend([wildcard, wildcard])
                if query.memory_type:
                    sql += " AND memory_type = ?"
                    params.append(query.memory_type.value)
                if query.categories:
                    cat_placeholders = ",".join("?" for _ in query.categories)
                    sql += f" AND category IN ({cat_placeholders})"
                    params.extend([c.value for c in query.categories])

                sql += " ORDER BY created_at DESC LIMIT ?;"
                params.append(query.limit)

                cursor = conn.execute(sql, params)
                for row in cursor.fetchall():
                    item = self._row_to_memory(row)
                    results.append(
                        MemorySearchResult(
                            item=item,
                            score=0.5,
                            match_reason="substring_fallback_match",
                        )
                    )

            return results
        except Exception as e:
            raise StorageError(f"Search failed for query '{query.query_text}': {e}") from e

    def list_memories(
        self,
        category: Optional[MemoryCategory] = None,
        include_deleted: bool = False,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> List[MemoryItem]:
        conn = self._get_connection()
        try:
            sql = "SELECT * FROM memories WHERE 1=1"
            params: List[Any] = []
            if not include_deleted:
                sql += " AND is_deleted = 0"
            if not include_superseded:
                sql += " AND superseded_by IS NULL"
            if category:
                sql += " AND category = ?"
                params.append(category.value)
            sql += " ORDER BY created_at DESC LIMIT ?;"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_memory(row) for row in cursor.fetchall()]
        except Exception as e:
            raise StorageError(f"Failed to list memories: {e}") from e

    def record_access(self, memory_id: str) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE memories 
                    SET access_count = access_count + 1,
                        last_accessed_at = ?
                    WHERE id = ?;
                    """,
                    (utc_now().isoformat(), memory_id),
                )
        except Exception as e:
            logger.warning(f"Failed to record access for memory {memory_id}: {e}")

    def save_episode(self, episode: EpisodicRecord) -> EpisodicRecord:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO episodes (
                        id, session_id, role, content, source_context, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        episode.id,
                        episode.session_id,
                        episode.role.value,
                        episode.content,
                        episode.source_context,
                        episode.timestamp.isoformat(),
                        json.dumps(episode.metadata),
                    ),
                )
            return episode
        except Exception as e:
            raise StorageError(f"Failed to save episode {episode.id}: {e}") from e

    def get_episodes(self, session_id: str, limit: int = 50) -> List[EpisodicRecord]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM episodes 
                WHERE session_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?;
                """,
                (session_id, limit),
            )
            episodes: List[EpisodicRecord] = []
            for row in cursor.fetchall():
                episodes.append(
                    EpisodicRecord(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=Role(row["role"]),
                        content=row["content"],
                        source_context=row["source_context"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                )
            return episodes
        except Exception as e:
            raise StorageError(f"Failed to get episodes for session {session_id}: {e}") from e

    def clear_all(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM memories;")
                conn.execute("DELETE FROM memories_fts;")
                conn.execute("DELETE FROM episodes;")
        except Exception as e:
            raise StorageError(f"Failed to clear database: {e}") from e
