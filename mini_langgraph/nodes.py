"""Built-in node implementations (LLM, Tool exec, Human, Router).

LLMNode
- Calls the LLM with system + conversation messages
- Streams tokens via callbacks when available
- Parses structured output into either an action (tool call) or final answer

ToolExecNode
- Locates the last AI action, executes the requested tool, records observation
- Appends an action/observation step to the scratchpad
- Routes back to the LLM for the next decision

HumanNode
- Interrupts execution to request external input (e.g., approval)

RouterNode
- Delegates the decision of the next node to a provided function
"""

from __future__ import annotations
from typing import Optional
from .graph import NodeResult, ExecutionContext
from .state import State
from .llm import OllamaClient, extract_action_or_final
from .tools import call_tool


class LLMNode:
    """Node that interacts with an LLM to produce the next agent step."""

    def __init__(
        self, model: str, system_prompt: str, temperature: float = 0.0
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = OllamaClient()

    def run(self, state: State, ctx: ExecutionContext) -> NodeResult:
        """Call the LLM with the current conversation and return updated state.

        Behavior
        - Ensures a `SystemMessage` is present at the start of `messages`
        - Streams tokens via callbacks (if streaming is available)
        - Parses a simple JSON contract for tool calls or final answers
        """
        # Expect state["messages"] to be list of dicts or similar
        from .messages import SystemMessage, AIMessage, to_chat_history

        messages = state.get("messages", [])
        if not messages or messages[0].role != "system":
            messages = [SystemMessage(content=self.system_prompt)] + messages

        chat_messages = to_chat_history(messages)
        # Attempt streaming via callbacks
        ctx.callbacks.llm_start(self.model, chat_messages, self.temperature)
        output = ""
        try:
            chunks = []
            for tok in self.client.chat_stream(
                model=self.model, messages=chat_messages, temperature=self.temperature
            ):
                chunks.append(tok)
                ctx.callbacks.llm_new_token(tok)
            output = "".join(chunks)
            ctx.callbacks.llm_end(output)
        except Exception:
            # fallback to non-streaming
            output = self.client.chat(
                model=self.model, messages=chat_messages, temperature=self.temperature
            )
        parsed = extract_action_or_final(output)

        ai_content = output
        ai_message = AIMessage(content=ai_content)
        if "action" in parsed and isinstance(parsed["action"], dict):
            ai_message.action_tool = parsed["action"].get("tool")
            ai_message.action_args = parsed["action"].get("args", {})
        if "final" in parsed:
            ai_message.final_answer = str(parsed["final"])

        new_state = state.merge({"messages": [ai_message]})
        return NodeResult(
            updated_state=new_state, next_node=None
        )  # next decided by controller


class ToolExecNode:
    """Node that runs the tool requested by the latest AI message."""

    def run(self, state: State, ctx: ExecutionContext) -> NodeResult:
        """Execute the selected tool and append a `ToolMessage` and scratch step."""
        from .messages import ToolMessage, AIMessage

        # Find last AI message with an action
        ai_with_action: Optional[AIMessage] = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.action_tool:
                ai_with_action = msg
                break
        if ai_with_action is None or not ai_with_action.action_tool:
            return NodeResult(updated_state=state, end=True)

        tool_name = ai_with_action.action_tool
        args = ai_with_action.action_args or {}
        try:
            ctx.callbacks.tool_start(tool_name, args)
            result = call_tool(tool_name, args)
        except Exception as e:
            observation = {"error": str(e)}
        else:
            observation = result
        finally:
            try:
                ctx.callbacks.tool_end(tool_name, observation)
            except Exception:
                pass

        tool_msg = ToolMessage(
            content=str(observation), tool_name=tool_name, observation=observation
        )
        new_state = state.merge(
            {
                "messages": [tool_msg],
                "scratchpad": [
                    {"action": tool_name, "args": args, "observation": observation}
                ],
            }
        )
        # After executing a tool, route back to the LLM for the next decision
        return NodeResult(updated_state=new_state, next_node="llm")


class HumanNode:
    """Node that interrupts execution to involve a human operator."""

    def __init__(self, resume_key: str, prompt: str) -> None:
        self.resume_key = resume_key
        self.prompt = prompt

    def run(self, state: State, ctx: ExecutionContext) -> NodeResult:
        """Return an interrupt result with a prompt and where to resume.

        The executor persists a checkpoint and returns. Later, call
        `Executor.resume(thread_id, injected=...)` to continue; the injected
        value will be merged under `resume_key`.
        """
        return NodeResult(
            updated_state=state,
            interrupt=True,
            interrupt_prompt=self.prompt,
            resume_key=self.resume_key,
        )


class RouterNode:
    """Node that chooses the next node name using `route_fn(state, ctx)`.

    Useful for building swarms, meta-controllers, or conditional branches.
    """

    def __init__(self, route_fn) -> None:
        self.route_fn = route_fn  # (state, ctx) -> next_node_name

    def run(self, state: State, ctx: ExecutionContext) -> NodeResult:
        next_node = self.route_fn(state, ctx)
        return NodeResult(updated_state=state, next_node=next_node)
