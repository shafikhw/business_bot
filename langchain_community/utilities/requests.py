"""Simplified RequestsWrapper compatible with the tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests


@dataclass
class RequestsWrapper:
    """Lightweight drop-in replacement for LangChain's RequestsWrapper."""

    headers: Optional[Dict[str, str]] = None
    base_url: str = ""
    session: Optional[requests.Session] = None

    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = self.session or requests.Session()
        self.headers = self.headers or {}
        self.base_url = self.base_url.rstrip("/")

    def _prepare_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if not self.base_url:
            return url
        if url.startswith("/"):
            return f"{self.base_url}{url}"
        return f"{self.base_url}/{url}"

    def post(self, url: str, params: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None):
        full_url = self._prepare_url(url)
        response = self._session.post(full_url, params=params, json=json, headers=self.headers)
        return response


__all__ = ["RequestsWrapper"]
