from pydantic import BaseModel
from mini_langgraph.tools import tool, call_tool, registry


class AddArgs(BaseModel):
    a: int
    b: int


@tool(name="add", description="Add two integers", args_model=AddArgs)
def add_tool(args: dict):
    return args["a"] + args["b"]


if __name__ == "__main__":
    print("Registered tools:", registry.names())
    result = call_tool("add", {"a": 2, "b": 40})
    print("add result:", result)
