from __future__ import annotations
import json
import sqlite3
from typing import Any, Dict, Optional, Tuple
from .state import State
from .messages import Message

"""SQLite-backed checkpoint and thread persistence.

This module stores serialized `State` snapshots after each node execution.
It enables:
- Auditable replay: Inspect state and node at each checkpoint
- Human-in-the-loop: Resume from an interrupt with injected input
- Branching: Although minimal here, the schema allows multiple threads

Serialization
Pydantic message models are converted to plain dicts so that the entire state
is JSON-serializable. On load, we keep them as dicts to avoid tight coupling
to model classes.
"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
  thread_id TEXT,
  checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
  graph_name TEXT,
  node_name TEXT,
  state_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Storage:
    """Thin wrapper around SQLite for saving and loading checkpoints."""

    def __init__(self, db_path: str = ".mini_langgraph.sqlite3") -> None:
        self.db_path = db_path
        self._ensure()

    def _ensure(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def _to_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, Message):
            data = obj.model_dump()
            data["__type__"] = obj.__class__.__name__
            return data
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_jsonable(x) for x in obj]
        return obj

    def _from_jsonable(self, obj: Any) -> Any:
        # Keep messages as dicts for simplicity in this minimal version
        return obj

    def save_checkpoint(
        self, thread_id: str, graph_name: str, node_name: str, state: State
    ) -> int:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            serializable = self._to_jsonable(dict(state))
            cur.execute(
                "INSERT INTO checkpoints(thread_id, graph_name, node_name, state_json) VALUES (?, ?, ?, ?)",
                (thread_id, graph_name, node_name, json.dumps(serializable)),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()

    def latest_checkpoint(
        self, thread_id: str
    ) -> Optional[Tuple[int, str, str, State]]:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT checkpoint_id, graph_name, node_name, state_json FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            checkpoint_id, graph_name, node_name, state_json = row
            restored = self._from_jsonable(json.loads(state_json))
            return int(checkpoint_id), graph_name, node_name, State(restored)
        finally:
            con.close()

    def load_checkpoint(self, checkpoint_id: int) -> Optional[Tuple[str, str, State]]:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT graph_name, node_name, state_json FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            graph_name, node_name, state_json = row
            restored = self._from_jsonable(json.loads(state_json))
            return graph_name, node_name, State(restored)
        finally:
            con.close()
