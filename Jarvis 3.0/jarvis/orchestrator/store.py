"""Task persistence layer for tracking task execution history in SQLite."""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import List, Optional

from jarvis.agents.verifier import VerificationResult
from jarvis.core.exceptions import StorageError
from jarvis.orchestrator.models import TaskRecord, TaskState, TaskStep


class TaskStore:
    """Manages persistent SQLite storage of task records and execution state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, timeout=20.0, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def initialize(self) -> None:
        """Create tasks table and indices."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_request TEXT NOT NULL,
                        state TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        current_step_index INTEGER NOT NULL,
                        pending_confirmation_json TEXT,
                        retrieved_memory_ids_json TEXT,
                        final_response TEXT,
                        verification_json TEXT,
                        error TEXT,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT
                    );
                    """
                )
                cols = [r["name"] for r in conn.execute("PRAGMA table_info(tasks);").fetchall()]
                if "metadata_json" not in cols:
                    conn.execute("ALTER TABLE tasks ADD COLUMN metadata_json TEXT DEFAULT '{}';")
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_session 
                    ON tasks(session_id, created_at);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_state 
                    ON tasks(state);
                    """
                )
        except Exception as e:
            raise StorageError(f"Failed to initialize tasks table: {e}") from e

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None

    def save_task(self, task: TaskRecord) -> TaskRecord:
        """Insert or update a task record."""
        conn = self._get_connection()
        try:
            meta_payload = dict(task.metadata)
            if task.workflow_type:
                meta_payload["workflow_type"] = task.workflow_type
            if task.intent:
                meta_payload["intent"] = task.intent
            if task.complexity:
                meta_payload["complexity"] = task.complexity
            if task.capabilities:
                meta_payload["capabilities"] = task.capabilities
            if task.replanned:
                meta_payload["replanned"] = task.replanned
            if task.replan_history:
                meta_payload["replan_history"] = task.replan_history

            with conn:
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, session_id, user_request, state, plan_json,
                        current_step_index, pending_confirmation_json,
                        retrieved_memory_ids_json, final_response,
                        verification_json, error, metadata_json, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        state = excluded.state,
                        plan_json = excluded.plan_json,
                        current_step_index = excluded.current_step_index,
                        pending_confirmation_json = excluded.pending_confirmation_json,
                        retrieved_memory_ids_json = excluded.retrieved_memory_ids_json,
                        final_response = excluded.final_response,
                        verification_json = excluded.verification_json,
                        error = excluded.error,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        task.task_id,
                        task.session_id,
                        task.user_request,
                        task.state.value,
                        json.dumps([s.model_dump() for s in task.plan]),
                        task.current_step_index,
                        json.dumps(task.pending_confirmation) if task.pending_confirmation else None,
                        json.dumps(task.retrieved_memory_ids),
                        task.final_response,
                        json.dumps(task.verification.model_dump()) if task.verification else None,
                        task.error,
                        json.dumps(meta_payload),
                        task.created_at.isoformat(),
                        task.updated_at.isoformat(),
                        task.completed_at.isoformat() if task.completed_at else None,
                    ),
                )
            return task
        except Exception as e:
            raise StorageError(f"Failed to save task {task.task_id}: {e}") from e

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch task by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_task(row)
        except Exception as e:
            raise StorageError(f"Failed to get task {task_id}: {e}") from e

    def list_tasks(self, session_id: Optional[str] = None, limit: int = 50) -> List[TaskRecord]:
        """List recent tasks, optionally filtered by session."""
        conn = self._get_connection()
        try:
            sql = "SELECT * FROM tasks"
            params = []
            if session_id:
                sql += " WHERE session_id = ?"
                params.append(session_id)
            sql += " ORDER BY created_at DESC LIMIT ?;"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_task(row) for row in cursor.fetchall()]
        except Exception as e:
            raise StorageError(f"Failed to list tasks: {e}") from e

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        plan_raw = json.loads(row["plan_json"]) if row["plan_json"] else []
        plan = [TaskStep(**s) for s in plan_raw]
        verification_raw = json.loads(row["verification_json"]) if row["verification_json"] else None
        verification = VerificationResult(**verification_raw) if verification_raw else None

        metadata_raw = {}
        if "metadata_json" in row.keys() and row["metadata_json"]:
            try:
                metadata_raw = json.loads(row["metadata_json"])
            except Exception:
                metadata_raw = {}

        return TaskRecord(
            task_id=row["task_id"],
            session_id=row["session_id"],
            user_request=row["user_request"],
            state=TaskState(row["state"]),
            plan=plan,
            current_step_index=int(row["current_step_index"]),
            pending_confirmation=json.loads(row["pending_confirmation_json"]) if row["pending_confirmation_json"] else None,
            retrieved_memory_ids=json.loads(row["retrieved_memory_ids_json"]) if row["retrieved_memory_ids_json"] else [],
            final_response=row["final_response"],
            verification=verification,
            error=row["error"],
            workflow_type=metadata_raw.get("workflow_type"),
            intent=metadata_raw.get("intent"),
            complexity=metadata_raw.get("complexity"),
            capabilities=metadata_raw.get("capabilities", []),
            replanned=metadata_raw.get("replanned", False),
            replan_history=metadata_raw.get("replan_history", []),
            metadata=metadata_raw,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )
