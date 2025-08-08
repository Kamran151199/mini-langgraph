from __future__ import annotations
from typing import Any, Dict, List, MutableMapping
from copy import deepcopy

"""State container with merge semantics.

This module defines `State`, a dict-like container that supports predictable
merging across steps. The merge rules model LangGraph-like behavior:

- For list keys (e.g., `messages`, `scratchpad`), values are appended.
- For dicts, keys are shallow-merged with the right-hand side overwriting.
- For scalars, the right-hand side overwrites.

Why this is helpful
- Node outputs can be combined without clobbering prior messages.
- Scratchpad/action logs accumulate across steps.
- You can layer additional fields safely (e.g., `config`, `metadata`).
"""


class State(dict):
    """Graph state with merge semantics.

    - list fields append (e.g., messages, scratchpad)
    - dict fields deep-merge
    - scalars overwrite
    """

    list_append_keys = {"messages", "scratchpad"}

    def merge(self, other: MutableMapping[str, Any]) -> "State":
        """Return a new State by merging `other` onto self.

        Examples
        - Append a new message: `state.merge({"messages": [msg]})`
        - Accumulate scratch steps: `state.merge({"scratchpad": [step]})`
        - Overwrite a scalar: `state.merge({"temperature": 0.0})`
        """
        merged: Dict[str, Any] = deepcopy(self)
        for key, value in other.items():
            if key in self.list_append_keys:
                existing = merged.get(key) or []
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(value, list):
                    merged[key] = existing + value
                else:
                    merged[key] = existing + [value]
            else:
                if isinstance(merged.get(key), dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = deepcopy(value)
        return State(merged)

    @classmethod
    def create(cls, **kwargs: Any) -> "State":
        """Create a fresh State with sensible defaults.

        Defaults
        - messages: list, empty
        - scratchpad: list, empty
        """
        defaults = {
            "messages": [],
            "scratchpad": [],
        }
        defaults.update(kwargs)
        return cls(defaults)
