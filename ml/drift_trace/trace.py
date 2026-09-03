"""Stage 2 core interface: trace_origin() and simulate_forward().

=== Verified OpenDrift 1.14.x semantics this module relies on ===
(checked empirically against opendrift==1.14.12 in this environment, not
assumed from documentation alone)

1. Backward integration must use ``run(..., time_step=-N)`` (a *negative*
   time step on the normal, non-negated environment). This is OpenDrift's
   documented and only physically-correct way to run backward — see
   https://opendrift.github.io/gallery/example_long_leeway_backtrack.html.
   The original notebook instead ran forward-in-time with *negated*
   fallback wind/current vectors. For simple constant fallback fields the
   two happen to reach the same mean end position (verified: both landed
   within ~1e-5 degrees of the analytically expected origin in a
   round-trip test), BUT they are not equivalent in general:
     - OpenOil explicitly disables oil weathering (evaporation,
       dispersion, emulsification) when ``time_step.days < 0``
       (see ``openoil.py``, `oil_weathering()` and the mass-balance
       methods). Negating the environment instead of negating time keeps
       weathering *on*, so mass/viscosity/oil-budget outputs from the
       "negated env" approach are not physically meaningful for a
       backward run (confirmed: fraction_evaporated grew from 0 to ~0.14
       over 24h under the negated-env method, vs staying exactly 0 under
       true negative-time_step backward integration).
     - Negating a *real* (non-constant) current/wind reader field is not
       equivalent to running backward through it at all — reversing a
       time-varying vector field's sign is not the same operation as
       reversing the direction of time through it.
   This module therefore always uses negative time_step for backward
   traces and never negates environmental fallback/reader values.

2. ``o.result`` is an xarray Dataset indexed by (trajectory, time). For a
   backward run, ``o.result.time`` is already in descending order, so
   ``isel(time=-1)`` is the earliest simulated instant (the origin) with
   no manual reversal needed (verified).

3. When a particle strands or is otherwise deactivated, its ``status``
   value freezes at the deactivation code for that one timestep and then
   becomes NaN in every subsequent timestep; its lon/lat similarly go to
   NaN from the following timestep onward (verified: last real position
   is preserved at the timestep of deactivation, not at the final
   timestep). Any origin/trajectory extraction that blindly reads
   ``isel(time=-1)`` therefore silently drops every particle that
   stranded or otherwise terminated before the run's final step — this
   was the core bug in the original notebook's origin estimator. This
   module instead reads each particle's own last valid (non-NaN) sample
   via ``particles.last_valid_positions``.

4. ``o.run()`` raises ``ValueError`` when every particle deactivates
   before the requested duration elapses (e.g. all stranded, or all left
   reader coverage with no fallback). The message text is NOT stable:
   OpenDrift's internal loop raises ``'No more active or scheduled
   elements, quitting.'`` but that is caught by an inner
   ``except Exception`` and rewrapped — as ``'Simulation stopped within
   first timestep. ...'`` if termination happens on/before the first
   calculation step, or (verified) not re-raised at all (``run()``
   returns normally) if it happens later. Matching on message text is
   therefore unreliable and was a bug in an earlier version of this
   module. What IS reliable (verified empirically for both the
   first-step and later-step early-termination cases): ``o.result`` is
   always populated with the partial trajectory data once elements have
   been seeded, whether or not ``run()`` raises. This module therefore
   catches ``ValueError`` from ``run()`` and treats it as a normal early
   termination whenever ``o.result is not None``, re-raising only if
   ``o.result`` is unset (which would indicate a pre-run failure, e.g.
   bad seeding, unrelated to early deactivation).

5. OpenDrift compares seed/reader times as naive datetimes and raises
   ``TypeError: Cannot compare tz-naive and tz-aware datetime-like
   objects`` if given a timezone-aware ``datetime`` (verified). All
   timestamps are normalized to naive UTC before being passed to
   OpenDrift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import particles as particle_utils
from .uncertainty import estimate_origin

DEFAULT_OIL_TYPE = "GENERIC BUNKER C"
DEFAULT_PARTICLE_COUNT = 1000
DEFAULT_TIME_STEP_SECONDS = 3600

ReleaseType = Literal["instantaneous", "continuous"]


class SimulationDomainError(RuntimeError):
    """Raised when the simulation cannot proceed for a domain/config reason
    (e.g. missing environmental forcing, invalid coordinates) rather than
    an OpenDrift-internal bug."""


def _to_naive_utc(value: datetime) -> datetime:
    """Normalize a datetime to naive UTC (OpenDrift cannot compare
    tz-aware and tz-naive datetimes internally)."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _validate_coordinates(lon: float, lat: float) -> None:
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"longitude {lon} out of valid range [-180, 180]")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude {lat} out of valid range [-90, 90]")


@dataclass
class EnvironmentalForcing:
    """Environmental inputs for a single trace_origin()/simulate_forward()
    call.

    ``wind_u``/``wind_v`` and either ``currents_file`` (a real,
    spatially/temporally resolved NetCDF reader — production path, from
    Copernicus Marine via currents.fetch_currents) or ``current_u``/
    ``current_v`` (constant fallback — demo/smoke-test path only) are
    required. This dataclass makes no network calls itself and does not
    decide "real vs demo" for you: the production call sites
    (wind.get_wind_uv / currents.fetch_currents) raise rather than
    substitute fallback values, so a caller only ends up with constant
    fallback currents by explicitly passing ``current_u``/``current_v``
    themselves.
    """

    wind_u: float | None = None
    wind_v: float | None = None
    currents_file: str | Path | None = None
    current_u: float | None = None
    current_v: float | None = None
    sources: list[str] = field(default_factory=list)

    def require_present(self) -> None:
        if self.wind_u is None or self.wind_v is None:
            raise SimulationDomainError(
                "No wind forcing supplied (wind_u/wind_v are None). The "
                "production pipeline must not silently substitute fallback "
                "wind values — pass real wind from wind.get_wind_uv() or "
                "explicit demo constants."
            )
        if self.currents_file is None and (
            self.current_u is None or self.current_v is None
        ):
            raise SimulationDomainError(
                "No current forcing supplied (no currents_file and no "
                "current_u/current_v). The production pipeline must not "
                "silently substitute fallback current values — pass a real "
                "currents NetCDF from currents.fetch_currents() or explicit "
                "demo constants."
            )


def _build_simulation(
    forcing: EnvironmentalForcing,
    loglevel: int,
):
    # Imported lazily so importing this module does not require opendrift
    # to be installed for callers that only need e.g. the dataclasses.
    from opendrift.models.openoil import OpenOil
    from opendrift.readers import reader_netCDF_CF_generic

    o = OpenOil(loglevel=loglevel)

    if forcing.currents_file is not None:
        reader = reader_netCDF_CF_generic.Reader(str(forcing.currents_file))
        o.add_reader([reader])
    else:
        o.set_config("environment:fallback:x_sea_water_velocity", forcing.current_u)
        o.set_config("environment:fallback:y_sea_water_velocity", forcing.current_v)

    o.set_config("environment:fallback:x_wind", forcing.wind_u)
    o.set_config("environment:fallback:y_wind", forcing.wind_v)

    return o


@dataclass
class ParticleCloud:
    lons: list[float]
    lats: list[float]
    ages_seconds: list[float]


@dataclass
class TraceOriginResult:
    detection: dict[str, Any]
    origin: dict[str, Any]
    particle_stats: dict[str, Any]
    particle_cloud: ParticleCloud
    metadata: dict[str, Any]
    # The live OpenDrift simulation object, kept only so callers (e.g. the
    # demo notebook) can still use OpenDrift's own plotting (`o.plot(...)`)
    # without this module reimplementing it. Not JSON-serializable and
    # deliberately excluded from build_result_contract().
    simulation: Any = None


def trace_origin(
    detection_lon: float,
    detection_lat: float,
    detection_time: datetime,
    forcing: EnvironmentalForcing,
    oil_type: str = DEFAULT_OIL_TYPE,
    trace_duration_hours: float = 24,
    particle_count: int = DEFAULT_PARTICLE_COUNT,
    time_step_seconds: int = DEFAULT_TIME_STEP_SECONDS,
    loglevel: int = 50,
) -> TraceOriginResult:
    """Trace a detected spill backward to estimate its probable origin.

    Runs a physically-correct backward integration (negative time_step —
    see module docstring) seeded at the detection point/time, then derives
    a probable-origin centroid with 50%/95% simulation-derived uncertainty
    radii from the particles' own last valid positions (not a plain mean
    over all particles, which would misrepresent stranded/terminated
    particles as if they never existed).

    Does not silently substitute dummy environmental data: callers must
    supply an ``EnvironmentalForcing`` with real wind/current data, or
    explicitly construct one with fallback constants for a documented
    demo/smoke-test run.
    """
    _validate_coordinates(detection_lon, detection_lat)
    if trace_duration_hours <= 0:
        raise ValueError("trace_duration_hours must be positive")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    forcing.require_present()

    detection_time = _to_naive_utc(detection_time)

    o = _build_simulation(forcing, loglevel)
    o.seed_elements(
        lon=detection_lon,
        lat=detection_lat,
        number=particle_count,
        time=detection_time,
        oil_type=oil_type,
    )

    requested_duration = timedelta(hours=trace_duration_hours)
    try:
        o.run(duration=requested_duration, time_step=-abs(time_step_seconds))
    except ValueError:
        # OpenDrift raises when every particle has been deactivated before
        # reaching the requested duration. The exact message is not stable
        # (see module docstring point 4) so we do not match on text: a
        # populated o.result is what actually proves this was a legitimate
        # early termination (e.g. everything stranded quickly near the
        # coast) rather than a pre-run failure.
        if o.result is None:
            raise

    outcome = particle_utils.summarize_outcomes(o)
    lons, lats, ages = particle_utils.last_valid_positions(o)

    origin = estimate_origin(lons, lats)

    # age_seconds is negative during a backward run (OpenDrift's age grows
    # in the direction of integration); the magnitude is the elapsed trace
    # duration for each particle.
    simulated_seconds = float(np.nanmax(np.abs(ages))) if np.any(~np.isnan(ages)) else 0.0
    actual_hours_traced = simulated_seconds / 3600.0
    origin_time = detection_time - timedelta(hours=actual_hours_traced)
    stopped_early = actual_hours_traced < trace_duration_hours - 1e-6

    valid_lons = lons[~np.isnan(lons)]
    valid_lats = lats[~np.isnan(lats)]

    return TraceOriginResult(
        detection={
            "lat": detection_lat,
            "lon": detection_lon,
            "timestamp": detection_time.isoformat(),
        },
        origin={
            "lat": origin.origin_lat,
            "lon": origin.origin_lon,
            "timestamp": origin_time.isoformat(),
            "uncertainty_50_km": origin.uncertainty_50_km,
            "uncertainty_95_km": origin.uncertainty_95_km,
            "uncertainty_note": (
                "Radii are simulation-derived dispersion statistics over the "
                "backward-traced particle cloud (distance from the particle "
                "centroid containing 50%/95% of successfully-traced "
                "particles), not a statistical confidence interval."
            ),
            "stopped_early": stopped_early,
            "actual_hours_traced": actual_hours_traced,
        },
        particle_stats={
            "total": outcome.total,
            "successful": outcome.successful,
            "stranded": outcome.stranded,
            "invalid": outcome.invalid,
            "other_terminated": outcome.other_terminated,
            "reason_counts": outcome.reason_counts,
        },
        particle_cloud=ParticleCloud(
            lons=valid_lons.tolist(),
            lats=valid_lats.tolist(),
            ages_seconds=ages[~np.isnan(ages)].tolist(),
        ),
        metadata={
            "environmental_sources": forcing.sources,
            "simulation_duration_hours": trace_duration_hours,
            "particle_count": particle_count,
            "oil_type": oil_type,
            "time_step_seconds": time_step_seconds,
            "direction": "backward",
        },
        simulation=o,
    )


@dataclass
class ForwardSimulationResult:
    trajectory: list[dict[str, Any]]
    particle_stats: dict[str, Any]
    metadata: dict[str, Any]
    # See TraceOriginResult.simulation — same purpose (notebook plotting).
    simulation: Any = None


def simulate_forward(
    origin_lon: float,
    origin_lat: float,
    origin_time: datetime,
    forcing: EnvironmentalForcing,
    oil_type: str = DEFAULT_OIL_TYPE,
    duration_hours: float = 24,
    particle_count: int = DEFAULT_PARTICLE_COUNT,
    time_step_seconds: int = DEFAULT_TIME_STEP_SECONDS,
    release_type: ReleaseType = "instantaneous",
    continuous_release_end: datetime | None = None,
    output_step_hours: float = 1,
    loglevel: int = 50,
) -> ForwardSimulationResult:
    """Forward-simulate spread from an estimated origin/spill state.

    Reuses the same environmental-forcing plumbing as ``trace_origin`` and
    returns a per-output-timestep centroid + spread trajectory, suitable
    for frontend map overlays and downstream AIS-correlation windowing.

    ``release_type='continuous'`` models an ongoing leak (e.g. a pipeline)
    by seeding elements continuously between ``origin_time`` and
    ``continuous_release_end`` rather than all at once — each increment
    then ages from its own release moment (OpenDrift native support via
    seeding with a ``time=[start, end]`` pair, verified).
    """
    _validate_coordinates(origin_lon, origin_lat)
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    if release_type == "continuous" and continuous_release_end is None:
        raise ValueError("continuous_release_end is required for release_type='continuous'")
    forcing.require_present()

    origin_time = _to_naive_utc(origin_time)
    seed_time: datetime | list[datetime]
    if release_type == "continuous":
        seed_time = [origin_time, _to_naive_utc(continuous_release_end)]
    else:
        seed_time = origin_time

    o = _build_simulation(forcing, loglevel)
    o.seed_elements(
        lon=origin_lon,
        lat=origin_lat,
        number=particle_count,
        time=seed_time,
        oil_type=oil_type,
    )

    requested_duration = timedelta(hours=duration_hours)
    try:
        o.run(
            duration=requested_duration,
            time_step=abs(time_step_seconds),
            time_step_output=int(output_step_hours * 3600),
        )
    except ValueError:
        # See trace_origin() / module docstring point 4: message text is
        # not stable, a populated o.result is what proves this was a
        # legitimate early termination rather than a pre-run failure.
        if o.result is None:
            raise

    outcome = particle_utils.summarize_outcomes(o)

    lon_all = o.result.lon.values
    lat_all = o.result.lat.values
    times = o.result.time.values

    trajectory: list[dict[str, Any]] = []
    for t_idx in range(lon_all.shape[1]):
        lon_t = lon_all[:, t_idx]
        lat_t = lat_all[:, t_idx]
        valid = ~(np.isnan(lon_t) | np.isnan(lat_t))
        if not np.any(valid):
            continue
        trajectory.append(
            {
                "timestamp": np.datetime_as_string(times[t_idx], unit="s"),
                "centroid_lon": float(np.mean(lon_t[valid])),
                "centroid_lat": float(np.mean(lat_t[valid])),
                "active_particles": int(valid.sum()),
                "particle_lons": lon_t[valid].tolist(),
                "particle_lats": lat_t[valid].tolist(),
            }
        )

    return ForwardSimulationResult(
        trajectory=trajectory,
        particle_stats={
            "total": outcome.total,
            "successful": outcome.successful,
            "stranded": outcome.stranded,
            "invalid": outcome.invalid,
            "other_terminated": outcome.other_terminated,
        },
        metadata={
            "environmental_sources": forcing.sources,
            "simulation_duration_hours": duration_hours,
            "particle_count": particle_count,
            "oil_type": oil_type,
            "release_type": release_type,
            "direction": "forward",
        },
        simulation=o,
    )
