"""LangGraph orchestration for the property matcher personas.

This module wires together a preference elicitation agent, a Bayut search agent,
and a concierge follow-up agent using LangGraph. The graph is designed to run as
a ReAct-style loop where each agent updates a shared state payload. The entry
point, :func:`run_property_matcher`, exposes configuration hooks for the model
and routing to specific personas.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

PersonaKey = Literal["preference_elicitation", "bayut_search", "concierge_follow_up"]


class PropertyMatcherState(TypedDict, total=False):
    """Shared state that flows between LangGraph nodes."""

    messages: List[Dict[str, str]]
    preferences: Dict[str, Any]
    listings: List[Dict[str, Any]]
    lead: Optional[Dict[str, Any]]
    feedback: Optional[str]
    next_action: PersonaKey | Literal["end"]
    context: Optional[str]


@dataclass
class Persona:
    """Configuration for a persona node in the graph."""

    key: PersonaKey
    name: str
    description: str
    system_prompt: str


DEFAULT_PERSONAS: Dict[PersonaKey, Persona] = {
    "preference_elicitation": Persona(
        key="preference_elicitation",
        name="Preference Specialist",
        description="Understands the client's goals and collects property search preferences.",
        system_prompt=(
            "You are NeuraEstate's preference specialist. Ask targeted questions "
            "to clarify the client's needs for Dubai properties (location, budget, "
            "bedrooms, amenities). Summarise what you know in bullet points and end "
            "with one clarifying question if more detail is required."
        ),
    ),
    "bayut_search": Persona(
        key="bayut_search",
        name="Bayut Scout",
        description="Searches Bayut listings and prepares a shortlist based on preferences.",
        system_prompt=(
            "You are a research analyst with real-time knowledge of Bayut listings. "
            "Propose 2-3 Dubai properties that match the client's stated preferences. "
            "Return concise listing summaries with location, price, bedrooms, and "
            "a compelling reason they match. If data is missing, clearly state the "
            "assumptions made."
        ),
    ),
    "concierge_follow_up": Persona(
        key="concierge_follow_up",
        name="Concierge",
        description="Schedules follow-ups, captures leads, and keeps the loop going.",
        system_prompt=(
            "You are a concierge who nurtures the lead. Provide next steps, capture "
            "contact information if offered, and encourage continued engagement. "
            "Close with an invitation for further details or another preference update."
        ),
    ),
}

def _append_message(state: PropertyMatcherState, role: str, content: str) -> None:
    history = list(state.get("messages", []))
    history.append({"role": role, "content": content})
    state["messages"] = history


def _extract_preferences(text: str, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simple heuristic extraction for common property preferences."""

    preferences: Dict[str, Any] = dict(existing or {})
    text_lower = text.lower()

    # Bedrooms
    bedroom_match = re.search(r"(\d+)\s*(bed|bedroom)", text_lower)
    if bedroom_match:
        preferences["bedrooms"] = int(bedroom_match.group(1))

    # Budget detection (accept formats like 2.5M, 2500000, 200k)
    budget_match = re.search(r"(\d+[\d,.]*\s*(?:m|million|k|aed)?)\s*(?:aed|dirham|dhs|million|m|k)?", text_lower)
    if budget_match:
        value = budget_match.group(1)
        value = value.replace(",", "")
        if value.endswith("m") or "million" in value:
            numeric = float(re.sub(r"[^0-9.]", "", value)) * 1_000_000
        elif value.endswith("k"):
            numeric = float(re.sub(r"[^0-9.]", "", value)) * 1_000
        else:
            numeric = float(re.sub(r"[^0-9.]", "", value))
        preferences["budget_aed"] = numeric

    # Location cues
    locations = [
        loc for loc in [
            "dubai marina",
            "downtown",
            "jlt",
            "business bay",
            "palm",
            "dubai hills",
            "jumeirah",
            "arabian ranches",
        ]
        if loc in text_lower
    ]
    if locations:
        preferences["locations"] = sorted(set(locations))

    # Property type
    for keyword, property_type in {
        "apartment": "apartment",
        "villa": "villa",
        "townhouse": "townhouse",
        "penthouse": "penthouse",
    }.items():
        if keyword in text_lower:
            preferences["property_type"] = property_type
            break

    return preferences


def _preferences_complete(preferences: Dict[str, Any]) -> bool:
    return bool(
        preferences
        and preferences.get("bedrooms")
        and preferences.get("budget_aed")
        and preferences.get("locations")
    )


def _invoke_persona(
    persona: Persona,
    state: PropertyMatcherState,
    llm: Optional[BaseChatModel],
    extra_system_context: Optional[str] = None,
) -> str:
    """Invoke an LLM with persona-specific prompting, falling back if needed."""

    messages = state.get("messages", [])
    prompt_messages: List[Tuple[str, str]] = [
        ("system", persona.system_prompt),
    ]
    if extra_system_context:
        prompt_messages.append(("system", extra_system_context))
    prompt_messages.extend((msg["role"], msg["content"]) for msg in messages if msg["role"] != "system")
    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    if llm is None:
        # Fallback deterministic response
        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        return (
            f"[{persona.name}] Unable to reach the language model. Based on the latest "
            f"message, here is a heuristic response: {last_user[:280]}"
        )

    try:
        response = llm.invoke(prompt.format_messages())
    except Exception as exc:  # pragma: no cover - defensive fallback
        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        return (
            f"[{persona.name}] Encountered an error reaching the model ({exc}). "
            f"Here's a heuristic echo of your last message: {last_user[:280]}"
        )

    return response.content if isinstance(response, BaseMessage) else str(response)


def _preference_elicitation_node(
    state: PropertyMatcherState,
    persona: Persona,
    llm: Optional[BaseChatModel],
) -> PropertyMatcherState:
    last_user_message = next((msg for msg in reversed(state.get("messages", [])) if msg["role"] == "user"), None)
    if last_user_message:
        state["preferences"] = _extract_preferences(last_user_message["content"], state.get("preferences"))

    summary = _invoke_persona(persona, state, llm, state.get("context"))
    _append_message(state, "assistant", summary)

    if _preferences_complete(state.get("preferences", {})):
        state["next_action"] = "bayut_search"
    else:
        state["next_action"] = "end"
    return state


def _synthesise_listings(preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    locations = preferences.get("locations") or ["Dubai Marina"]
    bedrooms = preferences.get("bedrooms", 2)
    budget = preferences.get("budget_aed", 2_000_000)
    property_type = preferences.get("property_type", "apartment")

    suggestions: List[Dict[str, Any]] = []
    step = max(int(budget // len(locations)) if locations else int(budget), 250_000)
    for idx, location in enumerate(locations[:3]):
        suggestions.append(
            {
                "title": f"{bedrooms}-bed {property_type.title()} in {location.title()}",
                "location": location.title(),
                "price_aed": int(min(budget, budget - idx * 0.1 * budget)),
                "bedrooms": bedrooms,
                "highlights": [
                    "Developer incentives available",
                    "Proximity to metro",
                    "Community amenities included",
                ][:2],
            }
        )
    if not suggestions:
        suggestions.append(
            {
                "title": f"{bedrooms}-bed {property_type.title()} in Dubai Marina",
                "location": "Dubai Marina",
                "price_aed": int(budget),
                "bedrooms": bedrooms,
                "highlights": ["Sea view", "Walkable lifestyle"],
            }
        )
    return suggestions


def _bayut_search_node(
    state: PropertyMatcherState,
    persona: Persona,
    llm: Optional[BaseChatModel],
) -> PropertyMatcherState:
    preferences = state.get("preferences", {})
    listings = _synthesise_listings(preferences)
    state["listings"] = listings

    listings_json = json.dumps(listings, ensure_ascii=False, indent=2)
    extra_context = (
        "Summaries should reference these structured Bayut-inspired listings:\n" + listings_json
    )
    summary = _invoke_persona(persona, state, llm, extra_context)
    _append_message(state, "assistant", summary)

    state["next_action"] = "concierge_follow_up" if listings else "end"
    return state


EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_REGEX = re.compile(r"\b\+?97[01]\d{7,9}|\b\d{7,12}\b")


def _extract_lead(state: PropertyMatcherState) -> Optional[Dict[str, Any]]:
    """Identify contact information from the conversation."""

    for message in reversed(state.get("messages", [])):
        if message["role"] != "user":
            continue
        email_match = EMAIL_REGEX.search(message["content"])
        phone_match = PHONE_REGEX.search(message["content"])
        if email_match or phone_match:
            lead: Dict[str, Any] = {"source": "chat", "notes": message["content"]}
            if email_match:
                lead["email"] = email_match.group(0)
            if phone_match:
                lead["phone"] = phone_match.group(0)
            return lead
    return None


def _concierge_node(
    state: PropertyMatcherState,
    persona: Persona,
    llm: Optional[BaseChatModel],
) -> PropertyMatcherState:
    listings = state.get("listings", [])
    preferences = state.get("preferences", {})
    highlights = json.dumps({"preferences": preferences, "listings": listings}, ensure_ascii=False)
    summary = _invoke_persona(
        persona,
        state,
        llm,
        extra_system_context=(
            "Use this structured context to guide your concierge follow-up: " + highlights
        ),
    )
    _append_message(state, "assistant", summary)

    lead = _extract_lead(state)
    if lead:
        state["lead"] = lead

    if not listings:
        state["feedback"] = "No listings available for the provided preferences."

    state["next_action"] = "end"
    return state


def _route_next(state: PropertyMatcherState) -> Literal[
    "preference_elicitation", "bayut_search", "concierge_follow_up", "end"
]:
    return state.get("next_action", "end")  # type: ignore[return-value]


def build_property_matcher_graph(
    *,
    personas: Optional[Dict[PersonaKey, Persona]] = None,
    agent_selection: Optional[Sequence[PersonaKey]] = None,
    llm: Optional[BaseChatModel] = None,
) -> StateGraph:
    """Build the LangGraph graph for the property matcher."""

    persona_map = dict(DEFAULT_PERSONAS)
    if personas:
        persona_map.update(personas)

    if agent_selection:
        missing = set(agent_selection) - set(persona_map)
        if missing:
            raise ValueError(f"Unknown personas requested: {', '.join(sorted(missing))}")
        selected = [persona_map[key] for key in agent_selection]
    else:
        selected = [persona_map[key] for key in ("preference_elicitation", "bayut_search", "concierge_follow_up")]

    graph = StateGraph(PropertyMatcherState)

    for persona in selected:
        if persona.key == "preference_elicitation":
            graph.add_node(
                persona.key,
                lambda state, persona=persona: _preference_elicitation_node(state, persona, llm),
            )
        elif persona.key == "bayut_search":
            graph.add_node(
                persona.key,
                lambda state, persona=persona: _bayut_search_node(state, persona, llm),
            )
        elif persona.key == "concierge_follow_up":
            graph.add_node(
                persona.key,
                lambda state, persona=persona: _concierge_node(state, persona, llm),
            )

    if not selected:
        raise ValueError("At least one persona must be selected for the property matcher graph.")

    entry_key = selected[0].key
    graph.set_entry_point(entry_key)

    for persona in selected:
        graph.add_conditional_edges(
            persona.key,
            _route_next,
            {
                "preference_elicitation": "preference_elicitation"
                if any(p.key == "preference_elicitation" for p in selected)
                else END,
                "bayut_search": "bayut_search"
                if any(p.key == "bayut_search" for p in selected)
                else END,
                "concierge_follow_up": "concierge_follow_up"
                if any(p.key == "concierge_follow_up" for p in selected)
                else END,
                "end": END,
            },
        )

    return graph


def _ensure_messages(conversation_state: PropertyMatcherState, context: Optional[str]) -> None:
    messages = list(conversation_state.get("messages", []))
    if not any(msg["role"] == "system" for msg in messages) and context:
        messages.insert(0, {"role": "system", "content": context})
    conversation_state["messages"] = messages


def run_property_matcher(
    conversation_state: PropertyMatcherState,
    *,
    model: Optional[str | BaseChatModel] = None,
    temperature: float = 0.3,
    agent_selection: Optional[Sequence[PersonaKey]] = None,
    personas: Optional[Dict[PersonaKey, Persona]] = None,
    config: Optional[RunnableConfig] = None,
) -> PropertyMatcherState:
    """Execute the LangGraph workflow for the property matcher personas."""

    state = PropertyMatcherState(**conversation_state)
    context = state.get("context")
    _ensure_messages(state, context)
    state["next_action"] = state.get("next_action") or "preference_elicitation"

    llm: Optional[BaseChatModel]
    if isinstance(model, BaseChatModel):
        llm = model
    elif isinstance(model, str):
        llm = ChatOpenAI(model=model, temperature=temperature)
    elif model is None:
        try:
            llm = ChatOpenAI(temperature=temperature)
        except Exception:  # pragma: no cover - fallback if OpenAI creds missing
            llm = None
    else:
        raise TypeError("model must be a string model name, a BaseChatModel, or None")

    graph = build_property_matcher_graph(
        personas=personas,
        agent_selection=agent_selection,
        llm=llm,
    )

    compiled = graph.compile()
    result_state: PropertyMatcherState = compiled.invoke(state, config=config)
    result_state.setdefault("messages", state.get("messages", []))
    return result_state


__all__ = [
    "Persona",
    "PersonaKey",
    "PropertyMatcherState",
    "DEFAULT_PERSONAS",
    "build_property_matcher_graph",
    "run_property_matcher",
]
