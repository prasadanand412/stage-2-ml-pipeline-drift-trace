"""Ocean current forcing via Copernicus Marine Service.

Downloads are cached to disk by bounding-box + time-range key so repeated
calls for the same request reuse the existing NetCDF file instead of
re-downloading (Copernicus Marine is rate-limited/slow, per CLAUDE.md).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CACHE_DIR, EnvironmentCredentials

DEFAULT_DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"


class CurrentsFetchError(RuntimeError):
    """Raised when ocean current data cannot be retrieved."""


def _cache_filename(
    dataset_id: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    start: datetime,
    end: datetime,
) -> str:
    key = (
        f"{dataset_id}_{min_lon}_{max_lon}_{min_lat}_{max_lat}_"
        f"{start.isoformat()}_{end.isoformat()}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"currents_{digest}.nc"


def fetch_currents(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    start: datetime,
    end: datetime,
    dataset_id: str = DEFAULT_DATASET_ID,
    credentials: EnvironmentCredentials | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download (or reuse cached) ocean current NetCDF for the given domain/time.

    Returns the local path to a NetCDF file with ``uo``/``vo`` variables,
    suitable for ``opendrift.readers.reader_netCDF_CF_generic.Reader``.

    Raises CurrentsFetchError if credentials are missing or the download
    fails — this function never fabricates current data.
    """
    credentials = credentials or EnvironmentCredentials.from_env()
    username, password = credentials.require_copernicus()

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_filename = _cache_filename(
        dataset_id, min_lon, max_lon, min_lat, max_lat, start, end
    )
    output_path = cache_dir / output_filename

    if output_path.exists():
        return output_path

    try:
        import copernicusmarine
    except ImportError as exc:
        raise CurrentsFetchError(
            "copernicusmarine package is not installed; cannot fetch ocean "
            "current forcing."
        ) from exc

    try:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            username=username,
            password=password,
            variables=["uo", "vo"],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_depth=0,
            maximum_depth=1,
            start_datetime=start.isoformat(),
            end_datetime=end.isoformat(),
            output_directory=str(cache_dir),
            output_filename=output_filename,
            disable_progress_bar=True,
        )
    except Exception as exc:  # copernicusmarine raises assorted exception types
        raise CurrentsFetchError(
            f"Copernicus Marine subset() failed for dataset {dataset_id}: {exc}"
        ) from exc

    if not output_path.exists():
        raise CurrentsFetchError(
            f"Copernicus Marine subset() reported success but {output_path} "
            "was not created."
        )
    return output_path
