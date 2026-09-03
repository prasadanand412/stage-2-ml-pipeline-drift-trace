"""Particle-result extraction and statistics from an OpenDrift simulation.

Centralizes reading of ``o.result`` so origin estimation and forward-trace
serialization agree on how status/NaN are interpreted. Based on
empirically-verified OpenDrift 1.14.x behavior (see module docstring in
trace.py for what was verified and how).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParticleOutcome:
    """Per-particle summary of how a simulation ended for that particle."""

    total: int
    successful: int  # reached the full requested duration, still active
    stranded: int  # deactivated due to coastline contact
    invalid: int  # NaN position with no recorded status (e.g. missing data)
    other_terminated: int  # deactivated for any other reason (outside domain, etc.)
    reason_counts: dict = field(default_factory=dict)


def _status_name(status_categories: list[str], code: float) -> str | None:
    if np.isnan(code):
        return None
    idx = int(code)
    if 0 <= idx < len(status_categories):
        return status_categories[idx]
    return f"unknown_status_{idx}"


def summarize_outcomes(o) -> ParticleOutcome:
    """Classify every seeded particle by how its simulation ended.

    Uses the *last non-NaN* status value recorded for each particle
    (particles keep their deactivation status frozen after the timestep
    they were deactivated in; positions go to NaN in later timesteps —
    verified empirically against OpenDrift 1.14.12).
    """
    status = o.result.status.values  # shape (trajectory, time)
    categories = list(o.status_categories)
    total = status.shape[0]

    successful = stranded = invalid = other_terminated = 0
    reason_counts: dict[str, int] = {}

    for i in range(total):
        row = status[i]
        valid_idx = np.where(~np.isnan(row))[0]
        if len(valid_idx) == 0:
            invalid += 1
            reason_counts["invalid"] = reason_counts.get("invalid", 0) + 1
            continue

        last_code = row[valid_idx[-1]]
        reason = _status_name(categories, last_code)

        if reason == "active":
            # Still active at its last recorded step. This covers both a
            # particle that reached the full requested duration and one
            # whose simulation as a whole ended early for a global reason
            # (e.g. reader time coverage exhausted) while this particle
            # itself never stranded/terminated.
            successful += 1
        elif reason == "stranded":
            stranded += 1
        else:
            other_terminated += 1
            reason_counts[reason or "unknown"] = reason_counts.get(reason or "unknown", 0) + 1

    return ParticleOutcome(
        total=total,
        successful=successful,
        stranded=stranded,
        invalid=invalid,
        other_terminated=other_terminated,
        reason_counts=reason_counts,
    )


def last_valid_positions(o) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lons, lats, ages_seconds) — one entry per particle — using
    each particle's own last valid (non-NaN) recorded position.

    This is the key correctness fix over the original notebook, which took
    the mean of ``lon.isel(time=-1)`` and therefore silently dropped every
    particle that had gone to NaN (stranded/deactivated) before the final
    timestep — those particles were excluded from the mean entirely rather
    than being counted or represented by their last real position.
    """
    lon = o.result.lon.values
    lat = o.result.lat.values
    age = o.result.age_seconds.values
    n = lon.shape[0]

    out_lon = np.full(n, np.nan, dtype=np.float64)
    out_lat = np.full(n, np.nan, dtype=np.float64)
    out_age = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        valid_idx = np.where(~np.isnan(lon[i]))[0]
        if len(valid_idx) == 0:
            continue
        last = valid_idx[-1]
        out_lon[i] = lon[i, last]
        out_lat[i] = lat[i, last]
        out_age[i] = age[i, last]

    return out_lon, out_lat, out_age
