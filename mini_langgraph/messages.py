"""Message primitives used throughout the mini framework.

This module models a simple, explicit chat message schema inspired by LangGraph
and chat-completion APIs. The goal is to make state trajectories auditable and
easy to serialize while supporting agent patterns like ReAct.

Key concepts
- Roles: `system`, `human`, `ai`, `tool`.
- Structured AI actions: An `AIMessage` can propose a tool call via
  `action_tool` and `action_args`, or deliver a terminal answer via
  `final_answer`.
- Tool observations: A `ToolMessage` records the tool result, both as a string
  for human readability and as the raw `observation` for programmatic use.

Typical flow
1) A `SystemMessage` sets behavior.
2) A `HumanMessage` asks something.
3) An `AIMessage` proposes `{tool, args}` or `{final}`.
4) A `ToolMessage` records the observation after tool execution.

All classes are `pydantic.BaseModel`, so they validate and can be serialized.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Base message with a role and content.

    This is intentionally minimal so it can be serialized easily, logged, and
    used by the LLM client to produce chat history.
    """

    role: str
    content: str


class SystemMessage(Message):
    """Out-of-band instruction for the agent or model behavior."""

    role: str = Field(default="system", frozen=True)


class HumanMessage(Message):
    """User input message."""

    role: str = Field(default="human", frozen=True)


class AIMessage(Message):
    """Model/agent output message.

    Attributes
    - action_tool: Optional name of tool the model wants to call.
    - action_args: Optional mapping of arguments for the tool.
    - final_answer: If set, indicates the agent intends to end the turn.
    """

    role: str = Field(default="ai", frozen=True)
    # Optional structured tool call proposed by the model
    action_tool: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    # If final answer is present, agent should end
    final_answer: Optional[str] = None


class ToolMessage(Message):
    """Represents the observation/result from executing a tool.

    Attributes
    - tool_name: The registered tool that was executed.
    - observation: Raw result value (can be any JSON-serializable type).
    """

    role: str = Field(default="tool", frozen=True)
    tool_name: str
    observation: Any


def to_chat_history(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert typed messages to a plain dict format for chat APIs.

    This produces a list of `{"role": ..., "content": ...}` compatible with
    simple chat clients like the Ollama REST API.
    """
    return [{"role": m.role, "content": m.content} for m in messages]
