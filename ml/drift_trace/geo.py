"""Small geospatial helpers used for origin/uncertainty computation.

Kept deliberately simple (planar-on-sphere haversine) — sufficient for the
scale of a single spill's backward-trace particle cloud (tens to a few
hundred km). Not valid across the antimeridian; documented as a known
limitation rather than silently handled.
"""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
