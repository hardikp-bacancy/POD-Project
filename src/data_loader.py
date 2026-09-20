"""
data_loader.py
--------------
Responsible for getting the raw dataset onto disk and into a Pandas DataFrame.

Design notes
------------
* The dataset is the public "Spotify Tracks Dataset" on Hugging Face
  (maharshipandya/spotify-tracks-dataset, BSD licence). No API key is needed.
* We download the Parquet conversion once and cache it in data/. Every later
  run reads the local file, so the app works offline after the first launch.
* We never assume the schema is correct -- validate_columns() checks what
  actually arrived and reports what is missing, so the rest of the app can
  adapt instead of crashing with a KeyError.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pandas as pd

# Where the dataset comes from and where we keep it.
HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/maharshipandya/"
    "spotify-tracks-dataset/parquet/default/train/0.parquet"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOCAL_PARQUET = os.path.join(DATA_DIR, "spotify_tracks.parquet")

# Columns we genuinely need. If one of these is missing the app cannot work.
REQUIRED_COLUMNS = ["track_name", "artists"]

# Columns we would like to have. Missing ones degrade features but are survivable.
OPTIONAL_COLUMNS = [
    "album_name", "track_genre", "popularity", "duration_ms", "explicit",
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "track_id",
]


class DatasetError(Exception):
    """Raised when the dataset cannot be loaded or is unusable."""


def download_dataset(url: str = HF_PARQUET_URL, dest: str = LOCAL_PARQUET) -> str:
    """Download the Parquet file to `dest` if it is not already cached."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # Downloaded to a .part file first so an interrupted download never
        # leaves a corrupt "valid-looking" cache behind.
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetError(
            "Could not download the dataset. Check your internet connection, "
            f"or manually place the Parquet/CSV file at: {dest}\n"
            f"Original error: {exc}"
        ) from exc
    return dest


def validate_columns(df: pd.DataFrame) -> dict:
    """
    Compare the real schema against what we expect.

    Returns a small report instead of raising, so callers (and the notebook)
    can print it. Raises only when a truly required column is absent.
    """
    present = set(df.columns)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in present]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in present]

    if missing_required:
        raise DatasetError(
            f"Dataset is missing required column(s): {missing_required}. "
            f"Columns found: {sorted(present)}"
        )

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_optional": missing_optional,
        "missing_values": {
            col: int(n) for col, n in df.isna().sum().items() if n > 0
        },
    }


def load_raw_data(path: str | None = None, download_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the raw dataset into a DataFrame.

    Accepts .parquet or .csv so a user can drop in their own copy of the data.
    """
    path = path or LOCAL_PARQUET

    if not os.path.exists(path):
        if not download_if_missing:
            raise DatasetError(f"Dataset not found at {path} and downloading is disabled.")
        path = download_dataset(dest=path)

    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise DatasetError(f"Unsupported file type: {path} (expected .parquet or .csv)")
    except DatasetError:
        raise
    except Exception as exc:  # corrupt file, unreadable, wrong format...
        raise DatasetError(
            f"Failed to read the dataset at {path}. The file may be corrupt -- "
            f"delete it and let the app re-download.\nOriginal error: {exc}"
        ) from exc

    if df.empty:
        raise DatasetError(f"The dataset at {path} loaded successfully but contains 0 rows.")

    # Drop the unnamed pandas index column the CSV export leaves behind.
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    return df


def describe_dataset(df: pd.DataFrame) -> str:
    """Human-readable summary, used by the notebook and the --inspect CLI."""
    report = validate_columns(df)
    lines = [
        f"Rows:    {report['n_rows']:,}",
        f"Columns: {report['n_columns']}",
        "",
        "Column names:",
        "  " + ", ".join(report["columns"]),
    ]
    if report["missing_optional"]:
        lines += ["", "Optional columns NOT present:", "  " + ", ".join(report["missing_optional"])]
    if report["missing_values"]:
        lines += ["", "Columns containing missing values:"]
        lines += [f"  {c}: {n}" for c, n in report["missing_values"].items()]
    else:
        lines += ["", "No missing values."]
    return "\n".join(lines)


if __name__ == "__main__":
    # `python -m src.data_loader` prints the Step-1 inspection report.
    data = load_raw_data()
    print(describe_dataset(data))
    print("\nFirst 3 rows:\n")
    print(data.head(3).to_string())
