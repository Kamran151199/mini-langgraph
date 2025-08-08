from __future__ import annotations
from typing import Any, Dict, Optional, Protocol
from .state import State
from .callbacks import CallbackManager, CallbackHandler

"""Graph and node interfaces for building agent workflows.

This module defines the core graph runtime:
- ExecutionContext: Per-run context like thread id, config, and callbacks
- NodeResult: How a node signals state updates, routing, interrupts, or end
- Node protocol: A callable interface `run(state, ctx) -> NodeResult`
- Graph: A registry of nodes, entrypoint, and terminal node set

Routing
Nodes can return `next_node` to route dynamically. If left `None`, the
`Executor` applies default routing (e.g., based on AI messages).
"""


class ExecutionContext:
    """Runtime context shared across node invocations.

    Attributes
    - thread_id: Logical conversation or workflow id
    - config: Arbitrary configuration, e.g., temperature, thresholds
    - callbacks: Hook points for streaming and observability
    """

    def __init__(
        self,
        thread_id: str,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[CallbackManager] = None,
    ) -> None:
        self.thread_id = thread_id
        self.config = config or {}
        self.callbacks = callbacks or CallbackManager()


class NodeResult:
    """Return value of a node execution.

    - updated_state: Merge into the running state
    - next_node: Route to another node name (if None, executor decides)
    - interrupt: Pause execution and request human input
    - interrupt_prompt: Message to present to a user
    - resume_key: Where to inject a future human answer into state
    - end: Signal terminal completion
    """

    def __init__(
        self,
        updated_state: Optional[State] = None,
        next_node: Optional[str] = None,
        interrupt: bool = False,
        interrupt_prompt: Optional[str] = None,
        resume_key: Optional[str] = None,
        end: bool = False,
    ) -> None:
        self.updated_state = updated_state
        self.next_node = next_node
        self.interrupt = interrupt
        self.interrupt_prompt = interrupt_prompt
        self.resume_key = resume_key
        self.end = end


class Node(Protocol):
    """A unit of computation within the graph.

    Contract: Must implement `run(state, ctx) -> NodeResult`.
    """

    def run(self, state: State, ctx: ExecutionContext) -> NodeResult: ...


class Graph:
    """A simple directed graph of named nodes.

    Typical usage
    g = Graph().add_node("llm", LLMNode(...)).add_node("tool", ToolExecNode()).set_entrypoint("llm")
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.entrypoint: Optional[str] = None
        self.terminal_nodes: set[str] = set()

    def add_node(self, name: str, node: Node) -> "Graph":
        """Register a node under a name. The first node becomes entrypoint by default."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = node
        if self.entrypoint is None:
            self.entrypoint = name
        return self

    def set_entrypoint(self, name: str) -> "Graph":
        """Set the entrypoint node name explicitly."""
        if name not in self.nodes:
            raise KeyError(name)
        self.entrypoint = name
        return self

    def add_terminal(self, name: str) -> "Graph":
        """Mark a node as terminal (optional informational use)."""
        if name not in self.nodes:
            raise KeyError(name)
        self.terminal_nodes.add(name)
        return self

    def get(self, name: str) -> Node:
        """Retrieve a node by name."""
        if name not in self.nodes:
            raise KeyError(name)
        return self.nodes[name]
