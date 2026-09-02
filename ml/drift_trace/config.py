"""Configuration and credential handling for Stage 2 environmental forcing.

Credentials are read from environment variables ONLY. Nothing here ever
prints, logs, or persists a credential value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Local cache for downloaded environmental forcing (currents/wind). Kept out
# of the OpenDrift readers module so both the wind and current fetchers share
# one on-disk cache location and convention.
DEFAULT_CACHE_DIR = Path(
    os.environ.get("OILTRACE_ENV_CACHE_DIR", "ml/drift_trace/cache")
)

OPENWEATHER_API_KEY_ENV = "OPENWEATHER_API_KEY"
COPERNICUS_USERNAME_ENV = "COPERNICUSMARINE_SERVICE_USERNAME"
COPERNICUS_PASSWORD_ENV = "COPERNICUSMARINE_SERVICE_PASSWORD"


class MissingCredentialError(RuntimeError):
    """Raised when a required credential env var is not set.

    Never include the credential value itself in the message.
    """


@dataclass(frozen=True)
class EnvironmentCredentials:
    openweather_api_key: str | None
    copernicus_username: str | None
    copernicus_password: str | None

    @classmethod
    def from_env(cls) -> "EnvironmentCredentials":
        return cls(
            openweather_api_key=os.environ.get(OPENWEATHER_API_KEY_ENV) or None,
            copernicus_username=os.environ.get(COPERNICUS_USERNAME_ENV) or None,
            copernicus_password=os.environ.get(COPERNICUS_PASSWORD_ENV) or None,
        )

    def require_openweather(self) -> str:
        if not self.openweather_api_key:
            raise MissingCredentialError(
                f"{OPENWEATHER_API_KEY_ENV} is not set. Wind forcing cannot be "
                "fetched from OpenWeatherMap. Set this environment variable "
                "with a valid API key; the pipeline will not fall back to "
                "dummy wind data in production use."
            )
        return self.openweather_api_key

    def require_copernicus(self) -> tuple[str, str]:
        if not self.copernicus_username or not self.copernicus_password:
            raise MissingCredentialError(
                f"{COPERNICUS_USERNAME_ENV} / {COPERNICUS_PASSWORD_ENV} are not "
                "both set. Ocean current forcing cannot be fetched from "
                "Copernicus Marine Service. Set both environment variables "
                "(free account at data.marine.copernicus.eu); the pipeline "
                "will not fall back to dummy current data in production use."
            )
        return self.copernicus_username, self.copernicus_password
