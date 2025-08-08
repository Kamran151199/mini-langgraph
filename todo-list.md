# TODO – Enhancements and parity items

- Command abstraction (LangGraph parity)
  - Implement `commands.py` with typed commands: `AddMessages`, `CallTool`, `UpdateState`, `SetNext`, `Interrupt`, `End`
  - Let nodes return `List[Command]` or `NodeResult`; provide adapter between them
  - Add a command interpreter in the executor (apply in order, validate, trace)
  - Extend callbacks to emit `on_command(cmd)` for auditing

- Async executor + richer streaming
  - Async/await executor and node `run` methods; cooperative cancellation
  - True incremental JSON parsing for tool-calls (partial object state machine)
  - Timeouts per node; graceful fallback from streaming to non-streaming

- Structured outputs & validation
  - Pydantic schema targets for LLM outputs; automatic JSON repair/retry
  - Enforce tool-call arg schema generation in prompts

- Parallel branches (fork/join)
  - Parallel node execution with merge policies (append/first-wins/custom)
  - Deterministic ordering, idempotency and join semantics

- Retries & policies
  - Per-node retries with backoff/jitter; circuit breakers for flaky tools
  - Budgeting (step/time/token/tool-call budgets) with enforcement

- Observability & tracing
  - OpenTelemetry spans for nodes, tools, checkpoints; context propagation
  - Metrics (time, cost, token counts); optional structured logs

- Tool sandboxing & security
  - Subprocess/isolation layer with resource/time limits; tool allowlists
  - Safe serialization for args/results; secret handling best practices

- Planner–executor agent
  - Planner that emits a small plan graph; executor runs it with our runtime
  - Plan validation and replanning on failures

- Memory & RAG
  - Vector store-backed memory node (read/write/query); summarization policies
  - Session-level vs. long-term memory management

- Deterministic testing harness
  - Stub LLM with scripted outputs; golden tests for state transitions
  - Snapshot tests for checkpoints and routing

- Persistence improvements
  - Optional rehydration of messages into Pydantic classes on load
  - Named threads/branches; list and diff checkpoints

- Configurables & typing
  - Typed config object; per-node overrides; feature flags

- Interfaces & UX
  - Simple CLI/server to run graphs and stream tokens via SSE/WebSocket
  - Example UIs for traces and human-in-the-loop prompts
