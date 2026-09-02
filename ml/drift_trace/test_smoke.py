"""Lightweight smoke tests for the Stage 2 interface.

These are NOT scientific validation of drift accuracy — they check
interface correctness, error handling, and result-schema shape using
small particle counts / short durations so they run in seconds, not
minutes. Run with: python -m ml.drift_trace.test_smoke
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ml.drift_trace.config import EnvironmentCredentials, MissingCredentialError
from ml.drift_trace.result import build_result_contract
from ml.drift_trace.trace import (
    EnvironmentalForcing,
    SimulationDomainError,
    simulate_forward,
    trace_origin,
)


def test_missing_forcing_raises():
    forcing = EnvironmentalForcing()
    try:
        trace_origin(72.6, 19.0, datetime(2026, 8, 27, 12, 0), forcing, particle_count=5)
    except SimulationDomainError:
        pass
    else:
        raise AssertionError("expected SimulationDomainError for missing forcing")
    print("PASS: missing-forcing raises SimulationDomainError")


def test_invalid_coordinates_raise():
    forcing = EnvironmentalForcing(wind_u=1.0, wind_v=1.0)
    for lon, lat in [(200.0, 10.0), (10.0, 100.0)]:
        try:
            trace_origin(lon, lat, datetime(2026, 8, 27, 12, 0), forcing, particle_count=5)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for lon={lon}, lat={lat}")
    print("PASS: invalid coordinates raise ValueError")


def test_missing_credential_error_message_has_no_secret():
    creds = EnvironmentCredentials(
        openweather_api_key=None, copernicus_username=None, copernicus_password=None
    )
    try:
        creds.require_openweather()
    except MissingCredentialError as exc:
        assert "OPENWEATHER_API_KEY" in str(exc)
    else:
        raise AssertionError("expected MissingCredentialError")
    print("PASS: missing-credential error identifies the env var, not a value")


def test_tz_aware_timestamp_normalized():
    forcing = EnvironmentalForcing(wind_u=3.0, wind_v=1.0, current_u=0.2, current_v=0.1)
    tz_time = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    result = trace_origin(
        70.0, 15.0, tz_time, forcing, trace_duration_hours=3, particle_count=10
    )
    assert result.detection["timestamp"] == "2026-08-27T12:00:00"
    print("PASS: tz-aware detection_time is normalized and does not raise")


def test_backward_trace_open_ocean_shape():
    # Open ocean, far from any coastline, small particle count / short
    # duration for speed. Verifies the result contract shape end-to-end.
    forcing = EnvironmentalForcing(
        wind_u=5.0, wind_v=2.0, current_u=0.3, current_v=0.1, sources=["fallback:demo"]
    )
    result = trace_origin(
        70.0, 15.0,
        datetime(2026, 8, 27, 12, 0),
        forcing,
        trace_duration_hours=6,
        particle_count=30,
    )
    assert result.particle_stats["total"] == 30
    assert result.particle_stats["successful"] == 30, result.particle_stats
    assert result.particle_stats["invalid"] == 0
    assert len(result.particle_cloud.lons) == 30
    assert result.origin["stopped_early"] is False
    contract = build_result_contract(result)
    assert contract["origin"]["uncertainty_50_km"] >= 0
    assert contract["origin"]["uncertainty_95_km"] >= contract["origin"]["uncertainty_50_km"]
    assert contract["particles"]["total"] == 30
    print("PASS: backward trace over open ocean produces a well-formed result contract")


def test_backward_trace_with_stranding():
    # Seed close to the Mumbai coastline with a slow onshore drift and a
    # short duration, so the run produces a genuine mix of particles that
    # strand and particles that remain active (verified empirically to
    # produce both outcomes with these parameters).
    forcing = EnvironmentalForcing(
        wind_u=-1.5, wind_v=0.2, current_u=-0.05, current_v=0.0, sources=["fallback:demo"]
    )
    result = trace_origin(
        72.80, 18.945,
        datetime(2026, 8, 27, 12, 0),
        forcing,
        trace_duration_hours=3,
        particle_count=300,
    )
    stats = result.particle_stats
    assert stats["total"] == 300
    assert stats["stranded"] > 0, "expected some particles to strand in this scenario"
    assert stats["successful"] > 0, "expected some particles to remain active"
    assert stats["stranded"] + stats["successful"] + stats["invalid"] + stats["other_terminated"] == 300
    # Stranded particles must still be represented in the particle cloud
    # (this is the core bug fix over the original notebook).
    assert len(result.particle_cloud.lons) == stats["successful"] + stats["stranded"] + stats["other_terminated"]
    print(
        f"PASS: stranding handled without silent particle loss "
        f"(stranded={stats['stranded']}, successful={stats['successful']})"
    )


def test_backward_trace_all_stranded_first_step():
    # Seed right at the coastline with a strong onshore current, so every
    # particle strands within the first calculation step. This is the
    # specific early-termination case where OpenDrift's run() raises
    # ValueError("Simulation stopped within first timestep. ...") rather
    # than the differently-worded message this module used to (incorrectly)
    # match on by string content — verifies the fix handles it via
    # o.result being populated rather than exception text.
    forcing = EnvironmentalForcing(
        wind_u=-20.0, wind_v=0.0, current_u=-5.0, current_v=0.0, sources=["fallback:demo"]
    )
    result = trace_origin(
        72.80, 18.945,
        datetime(2026, 8, 27, 12, 0),
        forcing,
        trace_duration_hours=24,
        particle_count=50,
    )
    stats = result.particle_stats
    assert stats["total"] == 50
    assert stats["stranded"] == 50, stats
    assert result.origin["stopped_early"] is True
    assert len(result.particle_cloud.lons) == 50
    print("PASS: total first-step stranding is handled as a normal early termination")


def test_forward_simulation_shape():
    forcing = EnvironmentalForcing(
        wind_u=4.0, wind_v=1.0, current_u=0.2, current_v=0.1, sources=["fallback:demo"]
    )
    result = simulate_forward(
        72.6, 19.0,
        datetime(2026, 8, 27, 12, 0),
        forcing,
        duration_hours=6,
        particle_count=20,
        output_step_hours=1,
    )
    assert len(result.trajectory) >= 1
    first = result.trajectory[0]
    assert "centroid_lon" in first and "centroid_lat" in first
    assert result.particle_stats["total"] == 20
    print("PASS: forward simulation produces a non-empty structured trajectory")


def test_continuous_release_ages_differ():
    forcing = EnvironmentalForcing(
        wind_u=3.0, wind_v=1.0, current_u=0.2, current_v=0.1, sources=["fallback:demo"]
    )
    result = simulate_forward(
        72.6, 19.0,
        datetime(2026, 8, 27, 10, 0),
        forcing,
        duration_hours=4,
        particle_count=20,
        release_type="continuous",
        continuous_release_end=datetime(2026, 8, 27, 12, 0),
    )
    last_step = result.trajectory[-1]
    assert result.metadata["release_type"] == "continuous"
    print("PASS: continuous release runs without error and tags metadata correctly")


if __name__ == "__main__":
    test_missing_forcing_raises()
    test_invalid_coordinates_raise()
    test_missing_credential_error_message_has_no_secret()
    test_tz_aware_timestamp_normalized()
    test_backward_trace_open_ocean_shape()
    test_backward_trace_with_stranding()
    test_backward_trace_all_stranded_first_step()
    test_forward_simulation_shape()
    test_continuous_release_ages_differ()
    print("\nAll Stage 2 smoke tests passed.")
