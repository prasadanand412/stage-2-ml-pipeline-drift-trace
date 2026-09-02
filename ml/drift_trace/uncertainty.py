"""Origin-point and uncertainty-radius estimation from a particle cloud.

These are simulation-derived dispersion statistics (distance from the
particle centroid within which 50%/95% of *valid* particles fall) — not a
statistical confidence interval in the inferential-statistics sense, since
particles share correlated forcing rather than being independent samples.
Labeled accordingly in the result contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import haversine_km


@dataclass
class OriginEstimate:
    origin_lon: float
    origin_lat: float
    uncertainty_50_km: float
    uncertainty_95_km: float
    valid_particle_count: int


def estimate_origin(lons: np.ndarray, lats: np.ndarray) -> OriginEstimate:
    """Compute a robust origin centroid and dispersion radii from particle
    end positions.

    ``lons``/``lats`` may contain NaN for particles with no valid final
    position at all (e.g. simulation produced no output for them); these
    are excluded from the estimate, and the caller is expected to already
    have accounted for them in the particle-outcome statistics (this
    function only handles geometry, not accounting).

    Raises ValueError if there are zero valid particles — the origin
    estimate must never be silently reported from an empty set.
    """
    valid_mask = ~(np.isnan(lons) | np.isnan(lats))
    valid_lons = lons[valid_mask]
    valid_lats = lats[valid_mask]
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise ValueError(
            "No valid particle positions available to estimate origin "
            "(all particles were invalid/NaN with no recorded position)."
        )

    # Circular mean for longitude would matter near the antimeridian; not
    # handled here (documented limitation, see geo.py).
    origin_lon = float(np.mean(valid_lons))
    origin_lat = float(np.mean(valid_lats))

    distances = np.array(
        [haversine_km(origin_lon, origin_lat, lo, la) for lo, la in zip(valid_lons, valid_lats)]
    )
    distances.sort()

    uncertainty_50 = float(np.percentile(distances, 50)) if n_valid > 0 else 0.0
    uncertainty_95 = float(np.percentile(distances, 95)) if n_valid > 0 else 0.0

    return OriginEstimate(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        uncertainty_50_km=uncertainty_50,
        uncertainty_95_km=uncertainty_95,
        valid_particle_count=n_valid,
    )
