"""Simplified StructuredTool implementation for local testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


@dataclass
class StructuredTool:
    """Minimal stand-in for LangChain's StructuredTool."""

    name: str
    description: str
    func: Callable[..., Any]
    args_schema: Optional[Type] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @classmethod
    def from_function(
        cls,
        *,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: Optional[Type] = None,
    ) -> "StructuredTool":
        return cls(name=name, description=description, func=func, args_schema=args_schema)


__all__ = ["StructuredTool"]
