from mini_langgraph.graph import Graph
from mini_langgraph.nodes import HumanNode
from mini_langgraph.execution import Executor
from mini_langgraph.state import State


def main():
    g = Graph()
    g.add_node("ask_human", HumanNode(resume_key="approval", prompt="Approve sending email? (yes/no)"))
    g.set_entrypoint("ask_human")

    ex = Executor(g, graph_name="human_loop")
    state = State.create()
    interrupted = ex.run(state, thread_id="thread-human")
    if "_interrupt" in interrupted:
        print("Interrupted:", interrupted["_interrupt"])  # simulate user input
        resumed = ex.resume(thread_id="thread-human", injected="yes")
        print("Resumed state:", resumed)


if __name__ == "__main__":
    main()
