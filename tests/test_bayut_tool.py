import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_bot.graph import build_property_search_graph
from business_bot.tools import BayutToolset


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class DummyRequestsWrapper:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, params=None, json=None):  # noqa: A002 - signature mirrors RequestsWrapper
        self.calls.append({"url": url, "params": params, "json": json})
        return DummyResponse(self._payload)


SAMPLE_LISTING = {
    "id": 123,
    "title": "Modern Apartment",
    "price": 2500000,
    "price_currency": "AED",
    "location_tree": [{"name": "Dubai"}, {"name": "Downtown"}],
    "rooms": 2,
    "baths": 3,
    "size": 1200,
    "amenities": [{"label": "Balcony"}, {"label": "Shared Pool"}],
    "verification": {"status": "truchecked"},
    "meta": {"url": "https://www.bayut.com/property/details-123.html"},
    "reference": "ABC123",
}

SAMPLE_PAYLOAD = {"data": {"results": [SAMPLE_LISTING]}, "page": 0, "count": 1}


@pytest.fixture()
def audit_log_path(tmp_path: Path) -> Path:
    return tmp_path / "bayut_audit.jsonl"


def test_search_properties_formats_request_and_cards(audit_log_path: Path):
    wrapper = DummyRequestsWrapper(SAMPLE_PAYLOAD)
    toolset = BayutToolset(
        api_key="test-key",
        spec_path="openapi.json",
        audit_log_path=audit_log_path,
        requests_wrapper=wrapper,
    )

    preferences = {"purpose": "for-sale", "price_min": 1000000, "unused": "skip"}
    result = toolset.search_properties(preferences=preferences, page=2, language="ar")

    assert wrapper.calls == [
        {
            "url": "/properties_search",
            "params": {"page": 2, "langs": "ar"},
            "json": {"purpose": "for-sale", "price_min": 1000000},
        }
    ]

    assert result.cards == [
        {
            "id": 123,
            "title": "Modern Apartment",
            "price": "AED 2,500,000",
            "location": "Dubai • Downtown",
            "bedrooms": 2,
            "bathrooms": 3,
            "size_sqft": 1200,
            "amenities": ["Balcony", "Shared Pool"],
            "is_trucheck": True,
            "url": "https://www.bayut.com/property/details-123.html",
            "raw_reference": "ABC123",
        }
    ]

    assert audit_log_path.exists()
    logged = audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert logged
    payload_entry = json.loads(logged[-1])
    assert payload_entry["request"]["payload"] == {"purpose": "for-sale", "price_min": 1000000}
    assert payload_entry["response"] == SAMPLE_PAYLOAD


def test_graph_invokes_bayut_tool_with_preferences(audit_log_path: Path):
    wrapper = DummyRequestsWrapper(SAMPLE_PAYLOAD)
    toolset = BayutToolset(
        api_key="test-key",
        spec_path="openapi.json",
        audit_log_path=audit_log_path,
        requests_wrapper=wrapper,
    )

    graph = build_property_search_graph(toolset)
    result_state = graph.invoke({"preferences": {"purpose": "for-rent", "price_max": 5000}})

    assert wrapper.calls[0]["json"] == {"purpose": "for-rent", "price_max": 5000}
    assert "bayut_property_search" in result_state["available_tools"]
    assert result_state["property_cards"]
    assert result_state["raw_payloads"][-1] == SAMPLE_PAYLOAD
    assert result_state["last_bayut_request"]["payload"] == {"purpose": "for-rent", "price_max": 5000}
