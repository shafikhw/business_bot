"""Bayut OpenAPI tooling for property search and recommendations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class BayutSearchResult:
    """Container for Bayut search results and audit metadata."""

    cards: List[Dict[str, Any]]
    raw: Dict[str, Any]
    request_payload: Dict[str, Any]
    query_params: Dict[str, Any]
    audit_path: Path


class BayutPropertySearchInput(BaseModel):
    """Schema for property search tool invocations."""

    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request body filters to forward to the Bayut /properties_search endpoint.",
    )
    page: Optional[int] = Field(
        default=None,
        ge=0,
        description="Zero-based pagination index forwarded as the `page` query parameter.",
    )
    language: Optional[str] = Field(
        default="en",
        description="Preferred response language (passed as the `langs` query parameter).",
    )


class BayutRecommendationInput(BayutPropertySearchInput):
    """Schema for Bayut recommendation tool invocations."""

    anchor_property_id: int = Field(
        ..., description="Property identifier used as the anchor for recommendation filtering."
    )


class BayutToolset:
    """Utility wrapper around the Bayut RapidAPI OpenAPI definition."""

    def __init__(
        self,
        api_key: str,
        *,
        spec_path: Path | str = "openapi.json",
        base_url: str = "https://bayut-api1.p.rapidapi.com",
        api_host: str = "bayut-api1.p.rapidapi.com",
        audit_log_path: Path | str = Path("logs/bayut_raw.jsonl"),
        requests_wrapper: Optional[RequestsWrapper] = None,
    ) -> None:
        self.spec_path = Path(spec_path)
        if not self.spec_path.exists():
            raise FileNotFoundError(f"OpenAPI specification not found at {self.spec_path}")

        with self.spec_path.open("r", encoding="utf-8") as handle:
            self.spec: Dict[str, Any] = json.load(handle)

        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_url = base_url.rstrip("/")
        self.api_host = api_host
        self.api_key = api_key

        self._requests = requests_wrapper or RequestsWrapper(
            headers={
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": api_host,
                "content-type": "application/json",
            },
            base_url=self.base_url,
        )

        self._allowed_filters = self._extract_allowed_filters()
        self._query_parameters = self._extract_query_parameters()

    # ---------------------------------------------------------------------
    # Public LangChain tool exposure
    # ---------------------------------------------------------------------
    def get_langchain_tools(self) -> List[StructuredTool]:
        """Return StructuredTool instances for LangGraph/LLM orchestration."""

        return [
            StructuredTool.from_function(
                name="bayut_property_search",
                description=(
                    "Search Bayut listings filtered by the caller's property preferences. "
                    "Accepts the raw Bayut JSON body under `preferences` so the agent can "
                    "forward amenities, location IDs, price ranges, etc. Returns structured "
                    "property cards summarising the results."
                ),
                func=self._property_search_tool,
                args_schema=BayutPropertySearchInput,
            ),
            StructuredTool.from_function(
                name="bayut_recommend_similar",
                description=(
                    "Recommend Bayut listings similar to a specific property. The caller must "
                    "provide an `anchor_property_id` and optional additional preferences to "
                    "further refine the result set."
                ),
                func=self._recommendation_tool,
                args_schema=BayutRecommendationInput,
            ),
        ]

    # ------------------------------------------------------------------
    # Core search / recommendation methods
    # ------------------------------------------------------------------
    def search_properties(
        self,
        *,
        preferences: Optional[Dict[str, Any]] = None,
        page: Optional[int] = None,
        language: Optional[str] = None,
    ) -> BayutSearchResult:
        """Call the `/properties_search` endpoint with provided filters."""

        body = self._prepare_payload(preferences or {})
        params = self._prepare_query_parameters(page=page, language=language)

        logger.debug("Calling Bayut properties_search", extra={"body": body, "params": params})
        response = self._requests.post("/properties_search", params=params or None, json=body)
        data = self._coerce_json(response)

        result = BayutSearchResult(
            cards=self._normalise_cards(data),
            raw=data,
            request_payload=body,
            query_params=params,
            audit_path=self.audit_log_path,
        )
        self._persist_raw("properties_search", result)
        return result

    def recommend_similar(
        self,
        *,
        anchor_property_id: int,
        preferences: Optional[Dict[str, Any]] = None,
        page: Optional[int] = None,
        language: Optional[str] = None,
    ) -> BayutSearchResult:
        """Recommend properties by re-using the search endpoint with an anchor property."""

        combined: Dict[str, Any] = {}
        if preferences:
            combined.update(preferences)
        # The public Bayut API accepts `reference_ids` arrays to target specific listings.
        # We fall back to a vendor-neutral key if the agent supplied one already.
        combined.setdefault("reference_ids", [anchor_property_id])
        combined.setdefault("similar_property_id", anchor_property_id)

        return self.search_properties(
            preferences=combined,
            page=page,
            language=language,
        )

    # ------------------------------------------------------------------
    # Internal LangChain tool entry-points
    # ------------------------------------------------------------------
    def _property_search_tool(self, *, preferences: Dict[str, Any], page: Optional[int] = None, language: Optional[str] = "en") -> Dict[str, Any]:
        """LangChain tool function for property search."""

        result = self.search_properties(preferences=preferences, page=page, language=language)
        return {
            "cards": result.cards,
            "count": len(result.cards),
            "audit_log": str(result.audit_path),
        }

    def _recommendation_tool(
        self,
        *,
        anchor_property_id: int,
        preferences: Dict[str, Any] | None = None,
        page: Optional[int] = None,
        language: Optional[str] = "en",
    ) -> Dict[str, Any]:
        """LangChain tool function for recommendation requests."""

        result = self.recommend_similar(
            anchor_property_id=anchor_property_id,
            preferences=preferences,
            page=page,
            language=language,
        )
        return {
            "cards": result.cards,
            "count": len(result.cards),
            "audit_log": str(result.audit_path),
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _extract_allowed_filters(self) -> set[str]:
        try:
            properties = (
                self.spec["paths"]["/properties_search"]["post"]["requestBody"]["content"]["application/json"][
                    "schema"
                ]["properties"]
            )
            return set(properties.keys())
        except KeyError:  # pragma: no cover - defensive guard if spec changes
            logger.warning("Unable to extract allowed filters from OpenAPI spec")
            return set()

    def _extract_query_parameters(self) -> set[str]:
        params: set[str] = set()
        try:
            for item in self.spec["paths"]["/properties_search"]["post"].get("parameters", []):
                if "name" in item:
                    params.add(item["name"])
        except KeyError:  # pragma: no cover - defensive guard if spec changes
            logger.warning("Unable to extract query parameters from OpenAPI spec")
        return params

    def _prepare_payload(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in preferences.items():
            if value is None:
                continue
            if self._allowed_filters and key not in self._allowed_filters:
                logger.debug("Ignoring unsupported Bayut filter", extra={"key": key})
                continue
            payload[key] = value
        return payload

    def _prepare_query_parameters(self, *, page: Optional[int], language: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if page is not None and (not self._query_parameters or "page" in self._query_parameters):
            params["page"] = page
        if language and (not self._query_parameters or "langs" in self._query_parameters):
            params["langs"] = language
        return params

    def _normalise_cards(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        data_section = payload.get("data") or {}
        listings: Iterable[Dict[str, Any]]
        if isinstance(data_section, dict) and "results" in data_section:
            listings = data_section.get("results", []) or []
        else:
            listings = payload.get("results") or payload.get("hits") or []

        cards: List[Dict[str, Any]] = []
        for listing in listings:
            if not isinstance(listing, dict):
                continue
            cards.append(self._create_property_card(listing))
        return cards

    def _create_property_card(self, listing: Dict[str, Any]) -> Dict[str, Any]:
        price_value = listing.get("price") or listing.get("price_value") or listing.get("list_price")
        currency = (
            listing.get("price_currency")
            or listing.get("currency")
            or listing.get("price_detail", {}).get("currency")
        )
        frequency = listing.get("rent_frequency") or listing.get("frequency")
        price_text = self._format_price(price_value, currency, frequency)

        location = self._format_location(listing)
        amenities = self._extract_amenities(listing)
        trucheck = self._extract_trucheck_status(listing)

        return {
            "id": listing.get("id") or listing.get("external_id") or listing.get("reference"),
            "title": listing.get("title") or listing.get("name"),
            "price": price_text,
            "location": location,
            "bedrooms": listing.get("rooms") or listing.get("bedrooms"),
            "bathrooms": listing.get("baths") or listing.get("bathrooms"),
            "size_sqft": listing.get("size") or listing.get("area") or listing.get("builtup_area"),
            "amenities": amenities,
            "is_trucheck": trucheck,
            "url": (listing.get("meta", {}) or {}).get("url") or listing.get("url"),
            "raw_reference": listing.get("reference") or listing.get("reference_number"),
        }

    def _format_price(self, value: Any, currency: Optional[str], frequency: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        currency_code = currency or "AED"
        price = f"{currency_code} {numeric:,.0f}"
        if frequency:
            price = f"{price} / {frequency.replace('_', ' ')}"
        return price

    def _format_location(self, listing: Dict[str, Any]) -> Optional[str]:
        location_tree = listing.get("location_tree") or listing.get("location") or []
        parts: List[str] = []
        if isinstance(location_tree, list):
            for item in location_tree:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("location")
                    if name:
                        parts.append(str(name))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(location_tree, dict):
            name = location_tree.get("name") or location_tree.get("location")
            if name:
                parts.append(str(name))
        if not parts:
            textual = listing.get("location_title") or listing.get("display_location")
            if textual:
                parts.append(str(textual))
        return " • ".join(parts) if parts else None

    def _extract_amenities(self, listing: Dict[str, Any]) -> List[str]:
        amenities_raw = listing.get("amenities") or listing.get("amenity_labels") or []
        amenities: List[str] = []
        if isinstance(amenities_raw, list):
            for item in amenities_raw:
                if isinstance(item, dict):
                    label = item.get("name") or item.get("label") or item.get("title")
                    if label:
                        amenities.append(str(label))
                elif isinstance(item, str):
                    amenities.append(item)
        return amenities

    def _extract_trucheck_status(self, listing: Dict[str, Any]) -> bool:
        verification = listing.get("verification") or {}
        if isinstance(verification, dict):
            status = verification.get("status") or verification.get("state")
            if isinstance(status, str):
                return status.lower() in {"truchecked", "approved", "verified", "true"}
            if isinstance(status, bool):
                return status
        for key in ("is_trucheck", "isTruChecked", "is_truchecked"):
            value = listing.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "yes", "approved", "verified"}
        return False

    def _coerce_json(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "json"):
            try:
                return response.json()  # type: ignore[return-value]
            except Exception:  # pragma: no cover - fallback to text handling
                text = getattr(response, "text", "")
                return json.loads(text) if text else {}
        if isinstance(response, str):
            return json.loads(response)
        raise ValueError("Unable to coerce Bayut response into JSON")

    def _persist_raw(self, endpoint: str, result: BayutSearchResult) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "request": {
                "payload": result.request_payload,
                "params": result.query_params,
            },
            "response": result.raw,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = ["BayutToolset", "BayutSearchResult", "BayutPropertySearchInput", "BayutRecommendationInput"]
