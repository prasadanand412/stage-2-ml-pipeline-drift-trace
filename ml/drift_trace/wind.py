"""Wind forcing via OpenWeatherMap.

CLAUDE.md mandates OpenWeatherMap (NOAA NOMADS OpenDAP is retired). This
module fetches current wind and caches the raw JSON response locally so
repeated calls for the same location/hour do not re-hit the rate-limited
API.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import requests

from .config import DEFAULT_CACHE_DIR, EnvironmentCredentials

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# OpenWeatherMap's free-tier "current weather" endpoint has no meaningful
# sub-hourly resolution, so cache entries are bucketed to the hour: this
# also naturally deduplicates repeated calls within the same hour.
_CACHE_BUCKET_SECONDS = 3600


class WindFetchError(RuntimeError):
    """Raised when wind data cannot be retrieved or parsed."""


def _cache_path(cache_dir: Path, lat: float, lon: float, bucket: int) -> Path:
    key = f"{round(lat, 3)}_{round(lon, 3)}_{bucket}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"wind_{digest}.json"


def get_wind_uv(
    lat: float,
    lon: float,
    credentials: EnvironmentCredentials | None = None,
    cache_dir: Path | None = None,
    timeout_seconds: float = 10.0,
) -> tuple[float, float]:
    """Fetch current wind at (lat, lon) as (u, v) m/s components.

    u is eastward, v is northward — the convention OpenDrift's
    ``x_wind``/``y_wind`` fallback config expects.

    Raises WindFetchError (never silently substitutes fallback values) if
    the API key is missing or the request fails. Caller decides whether a
    fallback is acceptable (e.g. explicit demo mode); this function never
    makes that decision itself.
    """
    credentials = credentials or EnvironmentCredentials.from_env()
    api_key = credentials.require_openweather()
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    bucket = int(time.time() // _CACHE_BUCKET_SECONDS)
    cache_file = _cache_path(cache_dir, lat, lon, bucket)

    if cache_file.exists():
        payload = json.loads(cache_file.read_text())
    else:
        try:
            response = requests.get(
                OPENWEATHER_URL,
                params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WindFetchError(f"OpenWeatherMap request failed: {exc}") from exc

        if response.status_code != 200:
            # Never include the API key (it is a query param, so the raw
            # response/URL is not echoed here).
            raise WindFetchError(
                f"OpenWeatherMap returned HTTP {response.status_code} for "
                f"lat={lat}, lon={lon}."
            )

        payload = response.json()
        if "wind" not in payload:
            raise WindFetchError(
                f"OpenWeatherMap response missing 'wind' field for lat={lat}, "
                f"lon={lon}: keys={list(payload.keys())}"
            )
        cache_file.write_text(json.dumps(payload))

    speed = payload["wind"]["speed"]
    deg = payload["wind"].get("deg", 0)

    # Meteorological convention: `deg` is the direction the wind blows
    # FROM, measured clockwise from north. Convert to the (u, v) vector the
    # wind blows TOWARD.
    u = -speed * math.sin(math.radians(deg))
    v = -speed * math.cos(math.radians(deg))
    return u, v
