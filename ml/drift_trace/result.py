"""Assembles the Stage 2 result contract (Phase 7) from trace_origin() and
simulate_forward() outputs, for the backend/frontend to consume."""
from __future__ import annotations

from typing import Any

from .trace import ForwardSimulationResult, TraceOriginResult


def build_result_contract(
    trace_result: TraceOriginResult,
    forward_result: ForwardSimulationResult | None = None,
) -> dict[str, Any]:
    """Combine a backward trace (required) and an optional forward
    simulation into the clean JSON-serializable contract consumed by the
    backend/frontend.

    Particle cloud coordinates are included so the frontend can render the
    backward-traced particle cloud, per Phase 4 of the Stage 2 hardening
    spec.
    """
    contract: dict[str, Any] = {
        "detection": trace_result.detection,
        "origin": {
            "lat": trace_result.origin["lat"],
            "lon": trace_result.origin["lon"],
            "timestamp": trace_result.origin["timestamp"],
            "uncertainty_50_km": trace_result.origin["uncertainty_50_km"],
            "uncertainty_95_km": trace_result.origin["uncertainty_95_km"],
            "uncertainty_note": trace_result.origin["uncertainty_note"],
            "stopped_early": trace_result.origin["stopped_early"],
        },
        "particles": {
            "total": trace_result.particle_stats["total"],
            "successful": trace_result.particle_stats["successful"],
            "stranded": trace_result.particle_stats["stranded"],
            "invalid": trace_result.particle_stats["invalid"],
            "other_terminated": trace_result.particle_stats["other_terminated"],
            "backward_cloud_lons": trace_result.particle_cloud.lons,
            "backward_cloud_lats": trace_result.particle_cloud.lats,
        },
        "metadata": trace_result.metadata,
    }

    if forward_result is not None:
        contract["forward_trace"] = {
            "trajectory": forward_result.trajectory,
            "particle_stats": forward_result.particle_stats,
        }
        contract["metadata"] = {
            **trace_result.metadata,
            "forward_metadata": forward_result.metadata,
        }

    return contract
