"""LangGraph orchestration for matching property preferences with Bayut tools."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .tools import BayutToolset


class PropertySearchState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    preferences: Dict[str, Any]
    property_cards: List[Dict[str, Any]]
    raw_payloads: List[Dict[str, Any]]
    available_tools: List[str]
    last_bayut_request: Dict[str, Any]


def _register_tools(state: PropertySearchState, tool_names: List[str]) -> PropertySearchState:
    updated = dict(state)
    updated.setdefault("available_tools", tool_names)
    updated.setdefault("preferences", {})
    return updated  # type: ignore[return-value]


def _run_bayut_search(state: PropertySearchState, toolset: BayutToolset) -> PropertySearchState:
    preferences = state.get("preferences", {})
    result = toolset.search_properties(preferences=preferences)

    raw_payloads: List[Dict[str, Any]] = list(state.get("raw_payloads", []))
    raw_payloads.append(result.raw)

    updated: PropertySearchState = {
        "preferences": preferences,
        "available_tools": state.get("available_tools", []),
        "property_cards": result.cards,
        "raw_payloads": raw_payloads,
        "last_bayut_request": {
            "payload": result.request_payload,
            "params": result.query_params,
        },
    }
    return updated


def build_property_search_graph(toolset: BayutToolset):
    """Compile a LangGraph that ensures Bayut tooling runs for property preferences."""

    graph = StateGraph(PropertySearchState)

    bayut_tools = toolset.get_langchain_tools()
    tool_names = [tool.name for tool in bayut_tools]

    graph.add_node("register_tools", lambda state: _register_tools(state, tool_names))
    graph.add_node("bayut_search", lambda state: _run_bayut_search(state, toolset))

    graph.set_entry_point("register_tools")
    graph.add_edge("register_tools", "bayut_search")
    graph.add_edge("bayut_search", END)

    compiled = graph.compile()
    # Attach the concrete LangChain tools for downstream agents/dispatchers.
    compiled.bayut_tools = bayut_tools
    return compiled


__all__ = ["PropertySearchState", "build_property_search_graph"]
