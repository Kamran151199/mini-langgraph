from typing import Any
from pydantic import BaseModel
from mini_langgraph.graph import Graph
from mini_langgraph.nodes import RouterNode
from mini_langgraph.react_agent import create_react_agent
from mini_langgraph.execution import Executor
from mini_langgraph.state import State
from mini_langgraph.messages import HumanMessage
from mini_langgraph.tools import tool


class CalcArgs(BaseModel):
    x: int
    y: int


@tool(name="multiply", description="Multiply two numbers", args_model=CalcArgs)
def multiply_tool(args: dict):
    return args["x"] * args["y"]


def build_math_agent() -> Graph:
    SYSTEM = (
        "You are a math agent. For math requests, call the 'multiply' tool. "
        "Return JSON only."
    )
    g = create_react_agent(model="llama3", system_prompt=SYSTEM)
    return g


def build_general_agent() -> Graph:
    SYSTEM = (
        "You are a general assistant. Answer succinctly. Return JSON only with 'final'."
    )
    g = create_react_agent(model="llama3", system_prompt=SYSTEM)
    return g


def main():
    swarm = Graph()
    agents = {
        "math": build_math_agent(),
        "general": build_general_agent(),
    }

    def route(state: State, ctx) -> str:
        content = " ".join([m.content for m in state.get("messages", []) if hasattr(m, "content")]).lower()
        if any(k in content for k in ["multiply", "product", "times", "math"]):
            sub = agents["math"]
        else:
            sub = agents["general"]
        # Execute subgraph to completion, then end
        sub_exec = Executor(sub, graph_name="sub")
        result_state = sub_exec.run(state, thread_id=ctx.thread_id)
        # place result back into state and end
        state.update(result_state)
        return None

    swarm.add_node("router", RouterNode(route))
    swarm.set_entrypoint("router")

    ex = Executor(swarm, graph_name="swarm")
    state = State.create(messages=[HumanMessage(content="What is the product of 6 and 7?")])
    out = ex.run(state, thread_id="swarm-thread")
    print("Final messages:")
    for m in out["messages"]:
        print(m.role, "::", getattr(m, "content", None))


if __name__ == "__main__":
    main()
