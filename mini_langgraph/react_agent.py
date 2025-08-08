from __future__ import annotations
from typing import List
from .graph import Graph
from .callbacks import CallbackManager, PrintCallback

"""Factory for a ReAct-style agent graph.

Provides a minimal 2-node graph:
- `llm`: Produces a tool call or final answer
- `tool`: Executes the requested tool and routes back to `llm`

This mirrors the core ReAct control loop in a compact form.
"""

from .nodes import LLMNode, ToolExecNode


def create_react_agent(
    model: str, system_prompt: str, temperature: float = 0.0
) -> Graph:
    """Create a simple ReAct agent graph with LLM and tool nodes.

    Parameters
    - model: LLM model name (for Ollama)
    - system_prompt: System instruction prepended to messages
    - temperature: LLM temperature
    """
    g = Graph()
    g.add_node(
        "llm",
        LLMNode(model=model, system_prompt=system_prompt, temperature=temperature),
    )
    g.add_node("tool", ToolExecNode())
    g.set_entrypoint("llm")
    return g
