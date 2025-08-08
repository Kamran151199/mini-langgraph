from mini_langgraph.react_agent import create_react_agent
from mini_langgraph.execution import Executor
from mini_langgraph.state import State
from mini_langgraph.messages import HumanMessage
from mini_langgraph.callbacks import CallbackManager, PrintCallback


def main():
    SYSTEM = 'You are a helpful assistant. Return JSON only. If you can answer directly, use {"final": "..."}.'
    g = create_react_agent(model="llama3", system_prompt=SYSTEM)
    ex = Executor(g, graph_name="streaming_demo")

    state = State.create(
        messages=[HumanMessage(content="Explain generators in Python in 2 sentences.")]
    )
    # Provide callbacks via config; ExecutionContext will pick it up in this minimal example by direct mutation
    # For simplicity, we inject callbacks by creating the context when calling run (here we mutate after creation)
    # Supply PrintCallback to see streamed tokens
    cbs = CallbackManager([PrintCallback()])
    out = ex.run(state, thread_id="stream-1", config={"callbacks": cbs})
    # Note: In this minimal version, to see tokens, you can modify Executor to accept callbacks or extend ExecutionContext creation.
    print("\n--- Final messages ---")
    for m in out["messages"]:
        print(m.role, "::", getattr(m, "content", None))


if __name__ == "__main__":
    main()
