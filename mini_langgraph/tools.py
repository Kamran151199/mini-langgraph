from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

"""Tool abstraction and registry.

This module provides a minimal, explicit mapping from LLM-proposed actions to
concrete Python functions with validated arguments.

Concepts
- ToolSpec: Metadata and a validated implementation.
- @tool decorator: Registers a callable with a name and optional Pydantic args model.
- call_tool: Looks up by name and executes with validated args.

Scenarios
- Simple tools: No args model, accept a dict.
- Typed tools: Provide a Pydantic model to validate and coerce inputs.
- Registry errors: Duplicate names or unknown tools raise helpful exceptions.
"""


class ToolSpec(BaseModel):
    """Description of a single tool.

    Fields
    - name: Registry key used by the agent.
    - description: Human-readable help text.
    - args_model: Optional Pydantic model type for validating args.
    - implementation: A function that receives a dict of validated args.
    """

    name: str
    description: str = ""
    args_model: Optional[Type[BaseModel]] = None
    implementation: Callable[[Dict[str, Any]], Any]


class ToolRegistry:
    """In-memory tool catalog with uniqueness checks."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool; duplicates are rejected."""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Retrieve a tool by name; raises KeyError if missing."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'")
        return self._tools[name]

    def names(self):
        return list(self._tools.keys())


registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: str = "",
    args_model: Optional[Type[BaseModel]] = None,
):
    """Decorator to convert a function into a registered tool.

    Usage
    @tool(name="search", args_model=SearchArgs)
    def search(args: dict):
        ...
    """

    def decorator(fn: Callable[[Dict[str, Any]], Any]):
        tool_name = name or fn.__name__

        def impl(args: Dict[str, Any]) -> Any:
            if args_model is not None:
                try:
                    validated = args_model(**args)
                except ValidationError as e:
                    raise ValueError(f"Invalid args for tool '{tool_name}': {e}")
                result = fn(validated.model_dump())
            else:
                result = fn(args)
            return result

        registry.register(
            ToolSpec(
                name=tool_name,
                description=description,
                args_model=args_model,
                implementation=impl,
            )
        )
        return fn

    return decorator


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    """Validate args (if model provided) and execute a tool implementation."""
    spec = registry.get(name)
    return spec.implementation(args)
