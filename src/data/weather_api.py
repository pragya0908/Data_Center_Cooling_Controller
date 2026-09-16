"""NASA POWER Meteorological API client for CoolRL.

Retrieves hourly near-surface air temperature (T2M) from NASA's Prediction
of Worldwide Energy Resources (POWER) API to serve as an external ambient
temperature scenario for the CoolRL data-center simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import requests
import pandas as pd

# Default coordinates for initial scenario: Bengaluru, Karnataka, India
DEFAULT_LATITUDE: float = 12.9716
DEFAULT_LONGITUDE: float = 77.5946
DEFAULT_COMMUNITY: str = "RE"  # Renewable Energy community
NASA_POWER_HOURLY_API_URL: str = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_nasa_power_hourly_t2m(
    start_date: str = "20180501",
    end_date: str = "20180530",
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    output_raw_dir: Path | str = "data/raw/weather",
) -> dict[str, Any]:
    """Fetch hourly 2-meter air temperature (T2M) from NASA POWER REST API.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYYMMDD' format (e.g. '20180501').
    end_date : str
        End date in 'YYYYMMDD' format (e.g. '20180530').
    latitude : float
        Latitude in decimal degrees (default: 12.9716 for Bengaluru).
    longitude : float
        Longitude in decimal degrees (default: 77.5946 for Bengaluru).
    output_raw_dir : Path | str
        Directory to store the raw API response and metadata.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from the NASA POWER API.
    """
    output_dir = Path(output_raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, str | float] = {
        "parameters": "T2M",
        "community": DEFAULT_COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    print(
        f"[NASA POWER API] Requesting hourly T2M for ({latitude}, {longitude}) "
        f"from {start_date} to {end_date}..."
    )

    response = requests.get(NASA_POWER_HOURLY_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    # Save exact raw response payload
    raw_json_file = output_dir / f"nasa_power_hourly_T2M_{start_date}_{end_date}.json"
    with raw_json_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[NASA POWER API] Raw JSON response saved to: {raw_json_file}")

    # Save exact query metadata and parameters
    meta_file = output_dir / "api_request_params.json"
    metadata = {
        "api_endpoint": NASA_POWER_HOURLY_API_URL,
        "query_parameters": params,
        "location_name": "Bengaluru, India",
        "parameter_description": "T2M: Near-surface air temperature at 2 meters above ground level",
        "parameter_units": "degrees Celsius (°C)",
        "temporal_resolution": "hourly",
        "saved_raw_file": str(raw_json_file),
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[NASA POWER API] Query parameters saved to: {meta_file}")

    return payload


def parse_nasa_power_t2m_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse NASA POWER JSON response into a structured pandas DataFrame.

    Parameters
    ----------
    payload : dict[str, Any]
        Raw JSON payload from NASA POWER.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp_raw', 'timestamp_utc', 'ambient_temp_c'].
    """
    t2m_dict = payload.get("properties", {}).get("parameter", {}).get("T2M", {})
    if not t2m_dict:
        raise ValueError("NASA POWER response contains no 'T2M' parameter data.")

    records = []
    for raw_time, temp_val in t2m_dict.items():
        # raw_time format is 'YYYYMMDDHH'
        dt = pd.to_datetime(raw_time, format="%Y%m%d%H", utc=True)
        records.append({
            "timestamp_raw": raw_time,
            "timestamp_utc": dt,
            "ambient_temp_c": float(temp_val),
        })

    df = pd.DataFrame(records)
    df.sort_values("timestamp_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    payload = fetch_nasa_power_hourly_t2m()
    df = parse_nasa_power_t2m_to_dataframe(payload)
    print(f"Parsed {len(df)} hourly temperature observations.")
    print("Sample:\n", df.head())
