from pydantic import BaseModel
from mini_langgraph.tools import tool, registry
from mini_langgraph.react_agent import create_react_agent
from mini_langgraph.execution import Executor
from mini_langgraph.state import State
from mini_langgraph.messages import HumanMessage, SystemMessage


class SearchArgs(BaseModel):
    query: str


@tool(name="search", description="Search the web (mock)", args_model=SearchArgs)
def search_tool(args: dict):
    q = args["query"].lower()
    # mock search result
    return {"results": [f"Result about {q}"]}


SYSTEM = (
    "You are a ReAct agent. Use tools if needed. "
    "Return JSON only. Examples:\n"
    "{\"thought\": \"...\", \"action\": {\"tool\": \"search\", \"args\": {\"query\": \"python\"}}}\n"
    "or {\"thought\": \"...\", \"final\": \"answer\"}"
)


def main():
    g = create_react_agent(model="llama3", system_prompt=SYSTEM)
    ex = Executor(g, graph_name="react_agent")

    state = State.create(messages=[HumanMessage(content="Find info about Python dataclasses")])
    out = ex.run(state, thread_id="t1")
    print("Final state messages:")
    for m in out["messages"]:
        print(m.role, "::", getattr(m, "content", None))
    if out.get("scratchpad"):
        print("Scratchpad steps:", out["scratchpad"]) 


if __name__ == "__main__":
    main()
