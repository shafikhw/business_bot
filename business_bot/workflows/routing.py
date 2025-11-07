"""LangGraph routing for enriching Bayut recommendations with map context."""
from __future__ import annotations

from typing import Any, Callable, List, MutableMapping, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ..tools.maps import (
    MapEnrichmentResult,
    PointOfInterest,
    enrich_recommendations_with_maps,
)


class RecommendationState(TypedDict, total=False):
    """Shared state flowing through the property recommendation workflow."""

    user_query: str
    bayut_results: List[MutableMapping[str, Any]]
    points_of_interest: Sequence[Any]
    recommendations: List[MutableMapping[str, Any]]
    map_enrichment_notice: Optional[str]
    response: str


StateHandler = Callable[[RecommendationState], RecommendationState]
PointsProvider = Callable[[RecommendationState], Sequence[Any]]


def _ensure_point(point: Any) -> PointOfInterest:
    if isinstance(point, PointOfInterest):
        return point

    if isinstance(point, MutableMapping):
        try:
            return PointOfInterest(
                name=str(point.get("name")),
                latitude=float(point.get("latitude")),
                longitude=float(point.get("longitude")),
                profile=point.get("profile"),
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            pass
    raise ValueError(f"Invalid point of interest: {point!r}")


def _resolve_points(state: RecommendationState, provider: Optional[PointsProvider]) -> List[PointOfInterest]:
    if provider is not None:
        points = provider(state)
    else:
        points = state.get("points_of_interest", []) or []

    resolved: List[PointOfInterest] = []
    for point in points:
        try:
            resolved.append(_ensure_point(point))
        except ValueError:
            continue
    return resolved


def build_recommendation_workflow(
    *,
    fetch_bayut: StateHandler,
    compose_response: StateHandler,
    points_provider: Optional[PointsProvider] = None,
    fallback_message: Optional[str] = None,
) -> Any:
    """Build a LangGraph workflow that enriches Bayut results with mapping details."""

    def routing_node(state: RecommendationState) -> RecommendationState:
        points = _resolve_points(state, points_provider)
        enrichment: List[MapEnrichmentResult] = enrich_recommendations_with_maps(
            state.get("bayut_results", []),
            points,
            fallback_message=fallback_message
            or "We're working on the map information and will update it shortly.",
        )

        state["recommendations"] = [result.listing for result in enrichment]
        notices = [result.fallback_message for result in enrichment if result.fallback_message]
        state["map_enrichment_notice"] = notices[0] if notices else None
        return state

    graph: StateGraph = StateGraph(RecommendationState)
    graph.add_node("bayut", fetch_bayut)
    graph.add_node("routing", routing_node)
    graph.add_node("respond", compose_response)

    graph.add_edge(START, "bayut")
    graph.add_edge("bayut", "routing")
    graph.add_edge("routing", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


__all__ = ["RecommendationState", "build_recommendation_workflow"]
