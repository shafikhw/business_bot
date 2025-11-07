"""Utility helpers for enriching property listings with map context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from urllib.parse import quote_plus

Coordinate = Tuple[float, float]


def _coerce_coordinate(value: Any) -> Optional[float]:
    """Attempt to coerce a latitude or longitude value to ``float``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _extract_coordinates(payload: Mapping[str, Any]) -> Optional[Coordinate]:
    """Extract ``(latitude, longitude)`` pairs from known Bayut payload shapes."""

    # Direct keys on the payload.
    direct_lat = _coerce_coordinate(payload.get("latitude") or payload.get("lat"))
    direct_lon = _coerce_coordinate(payload.get("longitude") or payload.get("lng") or payload.get("lon"))
    if direct_lat is not None and direct_lon is not None:
        return direct_lat, direct_lon

    for key in ("geography", "geolocation", "geo", "location", "coordinates"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            lat = _coerce_coordinate(nested.get("lat") or nested.get("latitude"))
            lon = _coerce_coordinate(nested.get("lng") or nested.get("lon") or nested.get("longitude"))
            if lat is not None and lon is not None:
                return lat, lon

    return None


@dataclass(frozen=True)
class PointOfInterest:
    """Simple representation of a nearby amenity or landmark."""

    name: str
    category: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def coordinates(self) -> Optional[Coordinate]:
        """Return a coordinate tuple when both latitude and longitude are set."""

        if self.latitude is None or self.longitude is None:
            return None
        return float(self.latitude), float(self.longitude)


@dataclass(frozen=True)
class TravelTimeEstimate:
    """Estimated travel time and distance for a given transport mode."""

    destination: str
    mode: str
    distance_km: float
    duration_minutes: float


@dataclass(frozen=True)
class MapEnrichmentResult:
    """Aggregate map context for a specific property listing."""

    listing_id: Any
    coordinates: Optional[Coordinate]
    static_map_url: Optional[str]
    travel_times: Dict[str, List[TravelTimeEstimate]]


def geocode_listing_location(listing: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive the best-effort geocode context for a Bayut listing payload."""

    coordinates = _extract_coordinates(listing)

    location_tree = listing.get("location_tree") if isinstance(listing, Mapping) else None
    query: Optional[str] = None
    if isinstance(location_tree, Sequence):
        names = [node.get("name") for node in location_tree if isinstance(node, Mapping) and node.get("name")]
        if names:
            query = ", ".join(str(name) for name in names)

    return {
        "latitude": coordinates[0] if coordinates else None,
        "longitude": coordinates[1] if coordinates else None,
        "query": query,
    }


def generate_static_map_url(listing: Mapping[str, Any]) -> Optional[str]:
    """Generate a shareable Google Maps link based on listing coordinates."""

    geocode = geocode_listing_location(listing)
    lat = geocode.get("latitude")
    lon = geocode.get("longitude")
    if lat is not None and lon is not None:
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

    if geocode.get("query"):
        return f"https://www.google.com/maps/search/?api=1&query={quote_plus(str(geocode['query']))}"

    return None


def _haversine_distance_km(origin: Coordinate, destination: Coordinate) -> float:
    """Compute great-circle distance between two WGS84 coordinates."""

    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _normalise_poi(poi: Union[PointOfInterest, Mapping[str, Any]]) -> PointOfInterest:
    """Coerce dictionary payloads into :class:`PointOfInterest` instances."""

    if isinstance(poi, PointOfInterest):
        return poi

    name = str(poi.get("name", "POI"))
    category = str(poi.get("category", "unknown"))
    latitude = _coerce_coordinate(poi.get("latitude") or poi.get("lat"))
    longitude = _coerce_coordinate(poi.get("longitude") or poi.get("lng") or poi.get("lon"))
    metadata = {key: value for key, value in poi.items() if key not in {"name", "category", "latitude", "lat", "longitude", "lng", "lon"}}
    return PointOfInterest(name=name, category=category, latitude=latitude, longitude=longitude, metadata=metadata or None)


DEFAULT_SPEEDS_KMH = {
    "walking": 5.0,
    "cycling": 15.0,
    "driving": 40.0,
    "transit": 25.0,
}


def estimate_travel_times(
    origin: Optional[Coordinate],
    destinations: Iterable[Union[PointOfInterest, Mapping[str, Any]]],
    *,
    modes: Optional[Iterable[str]] = None,
) -> Dict[str, List[TravelTimeEstimate]]:
    """Estimate travel time between an origin and a sequence of destinations."""

    if origin is None:
        return {}

    modes = list(modes or DEFAULT_SPEEDS_KMH.keys())
    results: Dict[str, List[TravelTimeEstimate]] = {}

    for index, raw_destination in enumerate(destinations):
        poi = _normalise_poi(raw_destination)
        coords = poi.coordinates()
        if coords is None:
            continue

        distance = _haversine_distance_km(origin, coords)
        destination_key = poi.name or f"destination_{index}"
        estimates: List[TravelTimeEstimate] = []

        for mode in modes:
            speed = DEFAULT_SPEEDS_KMH.get(mode)
            if not speed or speed <= 0:
                continue
            duration_hours = distance / speed
            estimates.append(
                TravelTimeEstimate(
                    destination=destination_key,
                    mode=mode,
                    distance_km=distance,
                    duration_minutes=duration_hours * 60.0,
                )
            )

        if estimates:
            results[destination_key] = estimates

    return results


def enrich_recommendations_with_maps(
    listings: Iterable[Mapping[str, Any]],
    *,
    points_of_interest: Optional[Iterable[Union[PointOfInterest, Mapping[str, Any]]]] = None,
    travel_modes: Optional[Iterable[str]] = None,
) -> List[MapEnrichmentResult]:
    """Augment property recommendations with derived map metadata."""

    poi_list: List[PointOfInterest] = []
    if points_of_interest:
        poi_list = [_normalise_poi(poi) for poi in points_of_interest]

    results: List[MapEnrichmentResult] = []

    for listing in listings:
        coordinates = _extract_coordinates(listing)
        travel = {}
        if coordinates and poi_list:
            travel = estimate_travel_times(coordinates, poi_list, modes=travel_modes)

        static_map = generate_static_map_url(listing)

        results.append(
            MapEnrichmentResult(
                listing_id=listing.get("id"),
                coordinates=coordinates,
                static_map_url=static_map,
                travel_times=travel,
            )
        )

    return results


__all__ = [
    "Coordinate",
    "MapEnrichmentResult",
    "PointOfInterest",
    "TravelTimeEstimate",
    "enrich_recommendations_with_maps",
    "estimate_travel_times",
    "generate_static_map_url",
    "geocode_listing_location",
]

