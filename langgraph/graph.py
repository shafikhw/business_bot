"""Simple sequential state graph stub used for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

END = "__end__"


@dataclass
class _Node:
    name: str
    fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass
class StateGraph:
    state_type: Any
    nodes: Dict[str, _Node] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)
    entry_point: Optional[str] = None

    def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> None:
        self.nodes[name] = _Node(name=name, fn=fn)

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def compile(self) -> "CompiledGraph":
        if self.entry_point is None:
            raise ValueError("Entry point must be defined before compiling the graph")
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph) -> None:
        self._graph = graph
        self.bayut_tools = []  # attribute used by callers

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self._graph.entry_point
        result_state: Dict[str, Any] = dict(state)
        while current and current != END:
            node = self._graph.nodes[current]
            update = node.fn(dict(result_state))  # provide a copy for isolation
            if update:
                result_state.update(update)
            current = self._graph.edges.get(current, END)
        return result_state


__all__ = ["StateGraph", "END"]
