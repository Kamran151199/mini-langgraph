from __future__ import annotations

"""Graph executor: runs nodes, manages routing, interruptions, checkpoints.

This executor applies a simple control loop over a `Graph`:
- Creates an `ExecutionContext` (thread id, config, callbacks)
- Iteratively runs the current node, merges returned state, and checkpoints
- If a node interrupts, execution returns with an interrupt marker in state
- If a node ends, returns the final state
- If no explicit `next_node`, uses default routing based on last AI message

Resuming: Use `resume(thread_id, injected=...)` to continue after a human-in-the-loop
interrupt. The injected value is merged under the node-provided `resume_key`.
"""
from typing import Any, Dict, Optional
from .graph import Graph, ExecutionContext, NodeResult
from .state import State
from .storage import Storage


class Executor:
    """Run a `Graph` to completion with checkpointing and interrupts.

    Parameters
    - graph: The graph to execute
    - graph_name: Used in checkpoint records for traceability
    - storage: Persistence backend for checkpoints (defaults to SQLite file)
    """

    def __init__(
        self, graph: Graph, graph_name: str = "graph", storage: Optional[Storage] = None
    ) -> None:
        self.graph = graph
        self.graph_name = graph_name
        self.storage = storage or Storage()

    def run(
        self,
        initial_state: Optional[State] = None,
        *,
        thread_id: str,
        config: Optional[Dict[str, Any]] = None,
        max_steps: int = 50,
    ) -> State:
        """Run the graph starting from its entrypoint.

        Arguments
        - initial_state: Optional state to resume/continue with
        - thread_id: Logical conversation/workflow id
        - config: Arbitrary configuration; may contain `callbacks`
        - max_steps: Safety cap to prevent infinite loops
        """
        state = initial_state or State.create()
        # Allow callbacks to be passed via config["callbacks"]
        callbacks = None
        if config and isinstance(config, dict):
            callbacks = config.get("callbacks")
        ctx = ExecutionContext(thread_id=thread_id, config=config, callbacks=callbacks)

        current_node_name = self.graph.entrypoint
        if not current_node_name:
            raise ValueError("Graph has no entrypoint")

        steps = 0
        while steps < max_steps:
            node = self.graph.get(current_node_name)
            try:
                ctx.callbacks.node_start(current_node_name, dict(state), ctx)
            except Exception:
                pass

            result: NodeResult = node.run(state, ctx)

            try:
                ctx.callbacks.node_end(current_node_name, dict(state), ctx, result)
            except Exception:
                pass

            # Merge state
            if result.updated_state is not None:
                state = result.updated_state

            # Checkpoint after each node
            checkpoint_id = self.storage.save_checkpoint(
                thread_id, self.graph_name, current_node_name, state
            )
            try:
                ctx.callbacks.checkpoint(
                    thread_id, self.graph_name, current_node_name, checkpoint_id
                )
            except Exception:
                pass

            # Interrupt
            if result.interrupt:
                # Return early with special state marker
                try:
                    ctx.callbacks.interrupt(result.interrupt_prompt, result.resume_key)
                except Exception:
                    pass
                return state.merge(
                    {
                        "_interrupt": {
                            "prompt": result.interrupt_prompt,
                            "resume_key": result.resume_key,
                        }
                    }
                )

            # End or next routing logic
            if result.end:
                return state

            if result.next_node:
                current_node_name = result.next_node
            else:
                # default controller: if last AI message has final_answer -> end; elif has action -> tool; else -> LLM
                def _get_field(obj, name):
                    return (
                        getattr(obj, name, None)
                        if hasattr(obj, name)
                        else obj.get(name)
                    )

                last_ai = None
                for msg in reversed(state.get("messages", [])):
                    role = (
                        getattr(msg, "role", None)
                        if hasattr(msg, "role")
                        else (msg.get("role") if isinstance(msg, dict) else None)
                    )
                    if role == "ai":
                        last_ai = msg
                        break
                if last_ai and _get_field(last_ai, "final_answer"):
                    return state
                if last_ai and _get_field(last_ai, "action_tool"):
                    # go to tool exec if exists
                    if "tool" in self.graph.nodes:
                        current_node_name = "tool"
                        steps += 1
                        continue
                # otherwise, go to llm if exists
                if "llm" in self.graph.nodes:
                    current_node_name = "llm"
                else:
                    return state

            steps += 1

        return state

    def resume(self, *, thread_id: str, injected: Dict[str, Any]) -> State:
        """Resume after an interrupt by injecting a human-provided value.

        Loads the latest checkpoint for `thread_id`, reads the stored
        `resume_key` from `_interrupt`, merges `injected` under that key, and
        continues running the graph.
        """
        latest = self.storage.latest_checkpoint(thread_id)
        if not latest:
            raise ValueError("No checkpoint to resume from")
        _, graph_name, node_name, state = latest
        # Merge injected value (e.g., human input) under resume key
        if "_interrupt" not in state:
            raise ValueError("State is not in interrupt mode")
        resume_key = state["_interrupt"].get("resume_key")
        # prompt kept in checkpoint for audit; not needed at resume time
        del state["_interrupt"]
        state = State(state).merge({resume_key: injected})
        # Continue from same node (or route)
        return self.run(state, thread_id=thread_id)
