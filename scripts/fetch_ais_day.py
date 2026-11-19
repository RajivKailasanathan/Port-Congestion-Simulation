"""Download one day of Danish AIS data and keep only a small port slice.

The daily DMA zip is several hundred MB. This script streams it to a
temporary file, reads the CSV in chunks straight from the zip (never
unzipped to disk), keeps rows inside a bounding box and for cargo/tanker
ship types, writes a small Parquet file, then deletes the zip.

Usage:
    python scripts/fetch_ais_day.py 2026-10-01 \
        --lat 55.40 55.50 --lon 10.30 10.45 --out data/

Check the DMA download listing for the current base URL and file naming
before running; it has changed over the years.
"""
import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://web.ais.dk/aisdata"  # confirm on the DMA download page
KEEP_TYPES = {"Cargo", "Tanker"}
COLUMNS = [
    "# Timestamp", "MMSI", "Latitude", "Longitude", "Navigational status",
    "SOG", "COG", "Heading", "IMO", "Name", "Ship type", "Length", "Width",
    "Draught", "Destination",
]


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def filter_zip(zip_path: Path, lat, lon) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(member) as fh:
            for chunk in pd.read_csv(fh, usecols=lambda c: c in COLUMNS,
                                     chunksize=500_000, low_memory=False):
                m = (chunk["Latitude"].between(*lat)
                     & chunk["Longitude"].between(*lon)
                     & chunk["Ship type"].isin(KEEP_TYPES))
                if m.any():
                    parts.append(chunk[m])
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    df["# Timestamp"] = pd.to_datetime(df["# Timestamp"],
                                       format="%d/%m/%Y %H:%M:%S")
    return df.rename(columns={"# Timestamp": "Timestamp"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("date", help="YYYY-MM-DD")
    p.add_argument("--lat", nargs=2, type=float, required=True)
    p.add_argument("--lon", nargs=2, type=float, required=True)
    p.add_argument("--out", default="data")
    p.add_argument("--base-url", default=BASE_URL)
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{a.base_url}/aisdk-{a.date}.zip"

    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    tmp = Path(tmp)
    try:
        print(f"Downloading {url} ...")
        download(url, tmp)
        df = filter_zip(tmp, a.lat, a.lon)
    finally:
        tmp.unlink(missing_ok=True)  # never keep the large zip

    out = out_dir / f"port_{a.date}.parquet"
    df.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"Kept {len(df):,} rows, {df['MMSI'].nunique()} vessels -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
