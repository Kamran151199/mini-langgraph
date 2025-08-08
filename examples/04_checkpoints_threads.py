from mini_langgraph.graph import Graph
from mini_langgraph.nodes import RouterNode
from mini_langgraph.execution import Executor
from mini_langgraph.state import State


def main():
    # trivial graph that just checkpoints on entry, then ends
    def route(state, ctx):
        return None  # will trigger default controller to end

    g = Graph()
    g.add_node("router", RouterNode(route))
    g.set_entrypoint("router")

    ex = Executor(g, graph_name="checkpoints")
    out1 = ex.run(State.create(), thread_id="th1")
    print("After first run, latest checkpoint stored. State size:", len(out1))
    # Run again on same thread to show continued checkpointing
    out2 = ex.run(out1, thread_id="th1")
    print("After second run, state size:", len(out2))


if __name__ == "__main__":
    main()
