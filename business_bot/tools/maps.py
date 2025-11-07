"""Map and geocoding helpers built around the Mapbox APIs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import requests

from ..config import MissingMapCredentialError, settings

LOGGER = logging.getLogger(__name__)

MAPBOX_BASE_URL = "https://api.mapbox.com"
MAPBOX_GEOCODING_URL = f"{MAPBOX_BASE_URL}/geocoding/v5/mapbox.places"
MAPBOX_STATIC_URL = f"{MAPBOX_BASE_URL}/styles/v1"
MAPBOX_MATRIX_URL = f"{MAPBOX_BASE_URL}/directions-matrix/v1"


class MapServiceUnavailable(RuntimeError):
    """Raised when the configured maps provider cannot service a request."""


@dataclass(slots=True)
class PointOfInterest:
    """Representation of a user provided point of interest."""

    name: str
    latitude: float
    longitude: float
    profile: Optional[str] = None  # Allows overriding the default routing profile


@dataclass(slots=True)
class TravelTimeEstimate:
    """A travel time estimation between an origin and a POI."""

    name: str
    distance_meters: Optional[float]
    duration_minutes: Optional[float]


@dataclass(slots=True)
class MapEnrichmentResult:
    """Container describing how a recommendation was enriched with map data."""

    listing: MutableMapping[str, Any]
    geocoding: Optional[Dict[str, Any]] = None
    static_map_url: Optional[str] = None
    travel_times: List[TravelTimeEstimate] = field(default_factory=list)
    fallback_message: Optional[str] = None
    error: Optional[str] = None


def _is_mapbox_enabled() -> bool:
    return settings.maps_provider.lower() == "mapbox"


def _extract_coordinates(listing: MutableMapping[str, Any]) -> Optional[Tuple[float, float]]:
    """Attempt to extract latitude/longitude pairs from a Bayut response payload."""

    if not listing:
        return None

    lat = None
    lon = None

    # Known Bayut response shapes
    geography = listing.get("geography") if isinstance(listing, MutableMapping) else None
    if isinstance(geography, MutableMapping):
        lat = geography.get("lat") or geography.get("latitude")
        lon = geography.get("lng") or geography.get("lon") or geography.get("longitude")

    if lat is None or lon is None:
        location = listing.get("location") if isinstance(listing, MutableMapping) else None
        if isinstance(location, MutableMapping):
            lat = lat or location.get("lat") or location.get("latitude")
            lon = lon or location.get("lon") or location.get("lng") or location.get("longitude")

    if lat is None or lon is None:
        coords = listing.get("coordinates") if isinstance(listing, MutableMapping) else None
        if isinstance(coords, (tuple, list)) and len(coords) >= 2:
            lon = lon or coords[0]
            lat = lat or coords[1]
        elif isinstance(coords, MutableMapping):
            lat = lat or coords.get("lat") or coords.get("latitude")
            lon = lon or coords.get("lng") or coords.get("lon") or coords.get("longitude")

    if lat is None or lon is None:
        return None

    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        LOGGER.debug("Invalid coordinates found on listing: %s", listing)
        return None


def _require_mapbox_token() -> str:
    try:
        return settings.require_mapbox_token()
    except MissingMapCredentialError as exc:  # pragma: no cover - defensive guard
        raise MapServiceUnavailable(str(exc)) from exc


def _request_mapbox(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _is_mapbox_enabled():
        raise MapServiceUnavailable("Mapbox provider is disabled in configuration.")

    token = _require_mapbox_token()
    params = dict(params or {})
    params.setdefault("access_token", token)

    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover - network failure guard
        LOGGER.warning("Mapbox request failed: %s", exc)
        raise MapServiceUnavailable("Unable to reach the map service at the moment.") from exc


def geocode_listing_location(
    listing: MutableMapping[str, Any], *, language: str = "en"
) -> Optional[Dict[str, Any]]:
    """Reverse geocode a Bayut listing using its coordinates."""

    if not _is_mapbox_enabled():
        LOGGER.debug("Maps provider %s disabled, skipping geocoding", settings.maps_provider)
        return None

    coordinates = _extract_coordinates(listing)
    if not coordinates:
        LOGGER.debug("Listing does not contain coordinates, skipping geocoding")
        return None

    lat, lon = coordinates
    query_url = f"{MAPBOX_GEOCODING_URL}/{lon},{lat}.json"
    try:
        payload = _request_mapbox(
            query_url,
            {
                "types": "address,place,locality,region",
                "language": language,
                "limit": 1,
            },
        )
    except MapServiceUnavailable:
        return None

    features = payload.get("features") or []
    return features[0] if features else None


def generate_static_map_url(
    latitude: float,
    longitude: float,
    *,
    zoom: int = 14,
    width: int = 640,
    height: int = 360,
    marker_color: str = "#2F855A",
    marker_label: str = "A",
) -> Optional[str]:
    """Return a Mapbox static map image URL for the supplied coordinates."""

    if not _is_mapbox_enabled():
        return None

    token = _require_mapbox_token()
    style = settings.mapbox_static_style
    marker = f"pin-s-{marker_label}+{marker_color.replace('#', '')}({longitude},{latitude})"
    dimensions = f"{max(1, width)}x{max(1, height)}"

    return (
        f"{MAPBOX_STATIC_URL}/{style}/static/{marker}/"
        f"{longitude},{latitude},{zoom}/{dimensions}@2x?access_token={token}"
    )


def estimate_travel_times(
    latitude: float,
    longitude: float,
    points_of_interest: Sequence[PointOfInterest],
    *,
    profile: Optional[str] = None,
) -> List[TravelTimeEstimate]:
    """Estimate travel times from the origin to each point of interest."""

    if not _is_mapbox_enabled() or not points_of_interest:
        return []

    profile = profile or settings.mapbox_directions_profile
    coordinates = [f"{longitude},{latitude}"] + [
        f"{poi.longitude},{poi.latitude}" for poi in points_of_interest
    ]
    url = f"{MAPBOX_MATRIX_URL}/{profile}/" + ";".join(coordinates)
    try:
        payload = _request_mapbox(url, {"annotations": "duration,distance"})
    except MapServiceUnavailable:
        return []

    durations = payload.get("durations") or []
    distances = payload.get("distances") or []

    estimates: List[TravelTimeEstimate] = []
    for index, poi in enumerate(points_of_interest, start=1):
        duration = None
        distance = None

        if len(durations) > 0 and len(durations[0]) > index:
            raw_duration = durations[0][index]
            if raw_duration is not None:
                duration = round(raw_duration / 60, 1)

        if len(distances) > 0 and len(distances[0]) > index:
            raw_distance = distances[0][index]
            if raw_distance is not None:
                distance = round(raw_distance, 1)

        estimates.append(
            TravelTimeEstimate(
                name=poi.name,
                distance_meters=distance,
                duration_minutes=duration,
            )
        )

    return estimates


def enrich_recommendations_with_maps(
    recommendations: Iterable[MutableMapping[str, Any]],
    points_of_interest: Sequence[PointOfInterest],
    *,
    fallback_message: str = (
        "We're temporarily unable to load live map details. You'll still see the "
        "property information while we reconnect to the map service."
    ),
) -> List[MapEnrichmentResult]:
    """Augment recommendation dictionaries with geocoding, maps, and travel time data."""

    enriched: List[MapEnrichmentResult] = []

    for item in recommendations:
        listing = item if isinstance(item, MutableMapping) else {}
        coordinates = _extract_coordinates(listing)

        if not coordinates:
            enriched.append(
                MapEnrichmentResult(
                    listing=listing,
                    fallback_message=fallback_message,
                    error="Missing coordinates on listing; unable to plot on map.",
                )
            )
            continue

        lat, lon = coordinates
        geocoding = geocode_listing_location(listing)
        static_map = None
        travel_times: List[TravelTimeEstimate] = []
        error: Optional[str] = None
        fallback: Optional[str] = None

        if geocoding is None:
            fallback = fallback_message

        try:
            static_map = generate_static_map_url(lat, lon)
            travel_times = estimate_travel_times(lat, lon, points_of_interest)
        except MapServiceUnavailable as exc:
            LOGGER.debug("Map service unavailable while enriching listing: %s", exc)
            error = str(exc)
            fallback = fallback_message

        listing_map: Dict[str, Any] = listing.setdefault("map", {})
        listing_map.update(
            {
                "latitude": lat,
                "longitude": lon,
                "geocoding": geocoding,
                "static_map_url": static_map,
                "travel_times": [
                    {
                        "name": estimate.name,
                        "distance_meters": estimate.distance_meters,
                        "duration_minutes": estimate.duration_minutes,
                    }
                    for estimate in travel_times
                ],
                "fallback_message": fallback,
            }
        )

        enriched.append(
            MapEnrichmentResult(
                listing=listing,
                geocoding=geocoding,
                static_map_url=static_map,
                travel_times=travel_times,
                fallback_message=fallback,
                error=error,
            )
        )

    return enriched


__all__ = [
    "MapEnrichmentResult",
    "MapServiceUnavailable",
    "PointOfInterest",
    "TravelTimeEstimate",
    "enrich_recommendations_with_maps",
    "estimate_travel_times",
    "generate_static_map_url",
    "geocode_listing_location",
]
