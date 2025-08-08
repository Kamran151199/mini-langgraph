# Mini LangGraph – A didactic reimplementation

This repository contains a compact, readable reimplementation of the core ideas behind LangGraph. It is intentionally small and heavily documented so you can understand how things work internally and adapt them to your own needs.

What you get:
- A directed graph runtime of typed nodes and edges
- A unified state model (messages, scratchpad, arbitrary fields) with merge semantics
- Tools as structured, typed functions with automatic arg validation and mapping
- A ReAct-style agent loop that decides to call tools or produce a final answer
- Human-in-the-loop interrupts and resumable checkpoints (threads/branches)
- Persistent state storage (SQLite) with resumable execution
- Configurables passed through the runtime
- Multi-agent routing (swarm) via a router node
- Ollama-backed LLM interface for local models

The code aims to answer: "How are tools called? How are args mapped to functions? Where and how are actions run? How does state persist? How do human interrupts work? How do we construct a ReAct agent and a swarm of agents?"

The implementation is synchronous and optimized for clarity over raw performance. The total code is small enough to read in one sitting.

## Contents

- `mini_langgraph/`
  - `messages.py`: Message types (system, human, ai, tool) and utilities
  - `state.py`: Shared state container and merge semantics
  - `tools.py`: Tool abstraction, registry, decorators, arg validation (Pydantic)
  - `graph.py`: Graph and Node abstractions; edge routing
  - `execution.py`: Executor that runs graphs, handles interrupts, scratchpad
  - `storage.py`: SQLite-backed checkpoint/threads persistence
  - `llm.py`: Ollama LLM client and a simple tool-call JSON protocol
  - `react_agent.py`: A factory to build a ReAct-style agent graph
  - `nodes.py`: Ready-to-use node types (LLMNode, ToolExecNode, HumanNode, RouterNode)
  - `callbacks.py`: Simple callback system incl. streaming token hooks
- `examples/`
  - `01_tools_basics.py`: Define and call tools with arg mapping
  - `02_react_agent.py`: Basic ReAct agent using tools and Ollama
  - `03_human_in_the_loop.py`: Inject human confirmation/input
  - `04_checkpoints_threads.py`: Persist and resume threads
  - `05_swarm_router.py`: Route across multiple agents

## Install

Requires Python 3.10+.

```
pip install -r requirements.txt
```

You also need an Ollama server running locally for LLM usage:

- Install Ollama from `https://ollama.com`
- Pull a model, e.g.:
```
ollama pull llama3
```

## Quickstart

Run the basic tools example:

```
python examples/01_tools_basics.py
```

Run the ReAct agent with a toy toolset and Ollama:

```
python examples/02_react_agent.py
```

Run with human-in-the-loop:

```
python examples/03_human_in_the_loop.py
```

Run with checkpoints/threads (resume demo):

```
python examples/04_checkpoints_threads.py
```

Run a swarm/router demo:

```
python examples/05_swarm_router.py
```

Run streaming demo (prints streamed tokens via callbacks):

```
python examples/06_streaming.py
```

## Design Overview

### 1) Messages and State

- Messages are structured records: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- State is a dict-like object with merge semantics. Lists (like `messages`, `scratchpad`) append; scalars overwrite. This mirrors LangGraph's typed state merging idea.
- The scratchpad holds action/observation pairs to ground the LLM's next step.

### 2) Tools

- Tools are Python callables registered under a name.
- Args are defined by a Pydantic model (or inferred) for robust validation and helpful errors.
- The runtime maps `{ "tool": "name", "args": {..} }` to the corresponding function and executes it.
- Results flow back into the state as `ToolMessage` and appended to scratchpad.

### 3) Graph, Nodes, and Execution

- A `Graph` is a mapping of node names to `Node` objects plus an entry point and optional terminal nodes.
- Each `Node` implements `run(state, ctx) -> NodeResult` and decides the next node via `NodeResult.next_node`.
- The `Executor` loops until a terminal condition: a node returns `end`, a maximum step limit, or an interrupt.
- Interrupts are explicit requests for human input; execution is checkpointed and can be resumed with user data.
- Callbacks allow observing node start/end, checkpoints, and LLM token streaming. LLM streaming is implemented in `llm.py` with `/api/chat` stream and surfaced via `callbacks.py`.

### 4) ReAct Agent

- The agent maintains a scratchpad of thoughts, actions, and observations.
- The LLM produces either a final answer or a tool call (as JSON). We keep the JSON contract simple and inspectable.
- Tool calls are validated, executed, and their observations added to the scratchpad to inform the next step.

### 5) Persistence: Checkpoints and Threads

- Every step may be checkpointed into SQLite: thread id, checkpoint id, graph, node, and serialized state.
- You can resume from any checkpoint and branch (creating multiple threads).
- This makes retracing, auditing, and replaying executions straightforward.

### 6) Configurables

- Arbitrary config (e.g., temperature, routing thresholds, feature flags) flows through the execution context.
- Nodes can read configuration to alter behavior at runtime.

### 7) Swarm/Router

- A `RouterNode` can select among sub-graphs (agents) based on the state or LLM suggestion.
- This composes multiple agents into a swarm without complicating individual agents.

## How tool-calls are parsed and executed

We instruct the LLM to output one JSON object per step with this shape:

```json
{
  "thought": "...",
  "action": {
    "tool": "search",
    "args": {"query": "weather in sf"}
  }
}
```

Or, for a final answer:

```json
{
  "thought": "...",
  "final": "It will be sunny."
}
```

This is intentionally simple. When an `action` appears, we look up the registered `Tool` by name, validate `args` via Pydantic, call the underlying function, and add a `ToolMessage(observation=...)` to state. The scratchpad is updated, and the loop continues.

## Human-in-the-loop

Nodes can return a `NodeResult` with `interrupt=True` and a `prompt` describing what is needed (e.g., "Approve sending email?"), plus a `resume_key` so you can later resume with the human-provided value. The `Executor` persists a checkpoint and returns control. A later call to `resume(thread_id, resume_key, value)` continues from that point.

## Glossary mapping to LangGraph

- Graph/Node/Edge → `Graph`, `Node`, `NodeResult` and dynamic `next_node`
- Messages → `messages.py` types
- Scratchpad → `state["scratchpad"]`
- Tool nodes → `ToolExecNode` and `tools.py`
- Human-in-the-loop → `HumanNode` and `interrupts` in `execution.py`
- Checkpoints/Threads → `storage.py` and `Executor` checkpointing
- Configurables → `ExecutionContext.config`
- `create_react_agent` → `react_agent.create_react_agent`
- Swarm → `RouterNode` that composes multiple sub-graphs

## Limitations

- Synchronous; no parallel branches in this minimal version
- Simple JSON tool-call contract; no OpenAI-style function-calling schema
- Not optimized for very large states or high throughput

## Reading tips

- Start with `graph.py` and `execution.py` to see how nodes run
- Skim `messages.py` and `state.py` for the state model
- Read `tools.py` to see how tool args are validated and executed
- Then study `react_agent.py` and `nodes.py` to see the agent loop and built-in nodes
- Finally, check `examples/` to see how to use it end-to-end

---

If you want to extend this with async execution, background tasks, retries/backoff, or OpenAI-style tool-calling, the abstractions here are designed to be easily adapted.
