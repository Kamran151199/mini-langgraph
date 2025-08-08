from __future__ import annotations
from typing import Any, Iterable, List, Optional

"""Callback interfaces for streaming and observability.

Define your own `CallbackHandler` to capture tokens, trace node execution,
log checkpoints, or notify a UI. Attach handlers via `CallbackManager` and
pass it to `Executor.run(..., config={"callbacks": manager})`.
"""


class CallbackHandler:
    """Base class with optional hooks for various runtime events."""

    def on_node_start(self, node_name: str, state: dict, ctx: Any) -> None:  # noqa: ANN401
        pass

    def on_node_end(self, node_name: str, state: dict, ctx: Any, result: Any) -> None:  # noqa: ANN401
        pass

    def on_llm_start(
        self, model: str, messages: List[dict], temperature: float
    ) -> None:
        pass

    def on_llm_new_token(self, token: str) -> None:
        pass

    def on_llm_end(self, output_text: str) -> None:
        pass

    def on_tool_start(self, tool_name: str, args: dict) -> None:
        pass

    def on_tool_end(self, tool_name: str, observation: Any) -> None:  # noqa: ANN401
        pass

    def on_checkpoint(
        self, thread_id: str, graph_name: str, node_name: str, checkpoint_id: int
    ) -> None:
        pass

    def on_interrupt(self, prompt: Optional[str], resume_key: Optional[str]) -> None:
        pass


class CallbackManager:
    """Aggregates multiple `CallbackHandler`s and fans out events to each."""

    def __init__(self, handlers: Optional[List[CallbackHandler]] = None) -> None:
        self.handlers: List[CallbackHandler] = handlers or []

    def add(self, handler: CallbackHandler) -> None:
        self.handlers.append(handler)

    # Dispatch helpers
    def node_start(self, node_name: str, state: dict, ctx: Any) -> None:
        for h in self.handlers:
            h.on_node_start(node_name, state, ctx)

    def node_end(self, node_name: str, state: dict, ctx: Any, result: Any) -> None:  # noqa: ANN401
        for h in self.handlers:
            h.on_node_end(node_name, state, ctx, result)

    def llm_start(self, model: str, messages: List[dict], temperature: float) -> None:
        for h in self.handlers:
            h.on_llm_start(model, messages, temperature)

    def llm_new_token(self, token: str) -> None:
        for h in self.handlers:
            h.on_llm_new_token(token)

    def llm_end(self, output_text: str) -> None:
        for h in self.handlers:
            h.on_llm_end(output_text)

    def tool_start(self, tool_name: str, args: dict) -> None:
        for h in self.handlers:
            h.on_tool_start(tool_name, args)

    def tool_end(self, tool_name: str, observation: Any) -> None:  # noqa: ANN401
        for h in self.handlers:
            h.on_tool_end(tool_name, observation)

    def checkpoint(
        self, thread_id: str, graph_name: str, node_name: str, checkpoint_id: int
    ) -> None:
        for h in self.handlers:
            h.on_checkpoint(thread_id, graph_name, node_name, checkpoint_id)

    def interrupt(self, prompt: Optional[str], resume_key: Optional[str]) -> None:
        for h in self.handlers:
            h.on_interrupt(prompt, resume_key)


class PrintCallback(CallbackHandler):
    """A trivial handler that prints streamed tokens to stdout."""

    def on_llm_new_token(self, token: str) -> None:
        print(token, end="", flush=True)
