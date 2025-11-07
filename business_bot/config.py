"""Configuration helpers for the business bot application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class MissingMapCredentialError(RuntimeError):
    """Raised when a map provider requires credentials that are not configured."""


@dataclass(slots=True)
class Settings:
    """Centralised application configuration loaded from environment variables."""

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    provider: str = os.getenv("PROVIDER", "openai")
    maps_provider: str = os.getenv("MAPS_PROVIDER", "mapbox")
    mapbox_access_token: Optional[str] = os.getenv("MAPBOX_ACCESS_TOKEN")
    mapbox_static_style: str = os.getenv("MAPBOX_STATIC_STYLE", "mapbox/streets-v12")
    mapbox_directions_profile: str = os.getenv("MAPBOX_PROFILE", "mapbox/driving")

    def require_mapbox_token(self) -> str:
        """Return the Mapbox token or raise if it is not set."""

        if not self.mapbox_access_token:
            raise MissingMapCredentialError(
                "Mapbox access token is required but was not provided. "
                "Set the MAPBOX_ACCESS_TOKEN environment variable."
            )
        return self.mapbox_access_token


settings = Settings()


__all__ = ["Settings", "settings", "MissingMapCredentialError"]
