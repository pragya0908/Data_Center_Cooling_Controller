"""Download pipeline for CoolRL real-world datasets.

Safely acquires:
1. Alibaba Cluster Trace 2018 (downsampled 300s machine usage from Zenodo)
2. Google Cluster Trace 2019 (downsampled 300s instance usage from Zenodo)
3. NASA POWER Hourly Meteorological Data (T2M air temperature for Bengaluru, India)

Uses streaming HTTP requests, prevents accidental overwriting, and validates file sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path
import requests

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.weather_api import fetch_nasa_power_hourly_t2m

# Verified download URLs from Zenodo record 14564847
ALIBABA_ZENODO_URL: str = (
    "https://zenodo.org/api/records/14564847/files/"
    "machine_usage_days_1_to_8_grouped_300_seconds.csv/content"
)
GOOGLE_ZENODO_URL: str = (
    "https://zenodo.org/api/records/14564847/files/"
    "instance_usage_grouped_300_seconds_month.csv/content"
)

RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
ALIBABA_RAW_DIR = RAW_DATA_DIR / "alibaba"
GOOGLE_RAW_DIR = RAW_DATA_DIR / "google"
WEATHER_RAW_DIR = RAW_DATA_DIR / "weather"


def download_file(
    url: str,
    destination: Path | str,
    description: str = "file",
    force: bool = False,
    chunk_size: int = 65536,
) -> Path:
    """Download a file via streaming HTTP GET with progress tracking.

    Parameters
    ----------
    url : str
        Source download URL.
    destination : Path | str
        Destination filepath.
    description : str
        Human-readable description for console logs.
    force : bool
        If True, re-download and overwrite existing file.
    chunk_size : int
        Download buffer size in bytes.

    Returns
    -------
    Path
        Destination Path object.
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        size_kb = dest_path.stat().st_size / 1024
        print(f"[CACHE] {description} already exists at {dest_path} ({size_kb:.1f} KB). Skipping download.")
        return dest_path

    print(f"[DOWNLOAD] Fetching {description} from:\n  {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    downloaded = 0

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with temp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        temp_path.replace(dest_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {description}: {exc}") from exc

    final_size_kb = dest_path.stat().st_size / 1024
    print(f"[SUCCESS] Downloaded {description} ({final_size_kb:.1f} KB) -> {dest_path}")
    return dest_path


def download_all_datasets(force: bool = False) -> dict[str, Path]:
    """Acquire all required real-world datasets for CoolRL.

    Parameters
    ----------
    force : bool
        If True, overwrite existing cached datasets.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset identifiers to their local file paths.
    """
    results: dict[str, Path] = {}

    # 1. Alibaba Cluster Trace 2018
    alibaba_dest = ALIBABA_RAW_DIR / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
    results["alibaba"] = download_file(
        url=ALIBABA_ZENODO_URL,
        destination=alibaba_dest,
        description="Alibaba 2018 Machine Usage Trace (300s resolution)",
        force=force,
    )

    # 2. Google Cluster Workload Traces 2019
    google_dest = GOOGLE_RAW_DIR / "instance_usage_grouped_300_seconds_month.csv"
    results["google"] = download_file(
        url=GOOGLE_ZENODO_URL,
        destination=google_dest,
        description="Google 2019 Instance Usage Trace (300s resolution)",
        force=force,
    )

    # 3. NASA POWER Hourly Weather (Bengaluru, India)
    # Fetch 30-day continuous period (2018-05-01 to 2018-05-30)
    weather_json_dest = WEATHER_RAW_DIR / "nasa_power_hourly_T2M_20180501_20180530.json"
    if not weather_json_dest.exists() or force:
        fetch_nasa_power_hourly_t2m(
            start_date="20180501",
            end_date="20180530",
            output_raw_dir=WEATHER_RAW_DIR,
        )
    else:
        print(f"[CACHE] Weather data already exists at {weather_json_dest}. Skipping API call.")
    results["weather"] = weather_json_dest

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("CoolRL Dataset Acquisition Pipeline")
    print("=" * 70)
    downloaded_paths = download_all_datasets()
    print("\nAll datasets acquired successfully:")
    for k, p in downloaded_paths.items():
        print(f"  - {k.upper()}: {p} ({p.stat().st_size / 1024:.1f} KB)")
