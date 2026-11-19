"""Summarise one raw DMA AIS day file and cut small port slices.

Reads the multi-GB CSV in chunks (never all in memory) and writes small
outputs to data/inspect/:
  head_sample.csv      first 20,000 rows, all columns (a few MB)
  summary.json         row count, time range, value counts, data-quality checks
  ports.csv            per candidate port: rows, vessels, stationary vessels
  slices/<port>.parquet  cargo + tanker rows inside each port box

Usage: put the day CSV files directly in data/, then run
    python scripts/inspect_day.py

Every data/*.csv is processed; days already inspected are skipped.
Outputs go to data/inspect/<day>/.

Takes roughly 3-10 minutes for a 3 GB file.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Approximate boxes (lat_min, lat_max, lon_min, lon_max) covering port + nearby anchorage.
# Placeholders for screening only; refine with polygons in Phase 0.
PORTS = {
    "aarhus":     (56.08, 56.22, 10.18, 10.45),
    "fredericia": (55.50, 55.62, 9.62, 9.85),
    "esbjerg":    (55.38, 55.52, 8.25, 8.50),
    "aalborg":    (57.00, 57.12, 9.80, 10.35),
    "kalundborg": (55.62, 55.72, 10.90, 11.15),
    "copenhagen": (55.62, 55.75, 12.55, 12.75),
    "skagen":     (57.65, 57.85, 10.45, 10.80),
}
CARGO_TANKER = {"Cargo", "Tanker"}
USECOLS = ["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude",
           "Navigational status", "SOG", "Ship type", "Length", "Draught",
           "Destination"]

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"


def day_files() -> list:
    """All CSV files directly in data/ that have not been inspected yet."""
    files = sorted(DATA.glob("*.csv"))
    return [f for f in files
            if not (DATA / "inspect" / f.stem / "summary.json").exists()]


def main(csv_path) -> None:
    src = Path(csv_path)
    day = src.name.split(".")[0]          # e.g. aisdk-2026-10-30
    out = DATA / "inspect" / day
    print(f"== {src.name} -> {out}", flush=True)
    (out / "slices").mkdir(parents=True, exist_ok=True)

    pd.read_csv(src, nrows=20_000, low_memory=False).to_csv(out / "head_sample.csv", index=False)

    n_rows = 0
    mmsi_all, mmsi_ct = set(), set()
    ship_type, nav_status, mobile = Counter(), Counter(), Counter()
    bad_pos = sog_missing = 0
    t_min = t_max = None
    port_rows = {p: [] for p in PORTS}

    reader = pd.read_csv(src, usecols=lambda c: c in USECOLS,
                         chunksize=1_000_000, low_memory=False)
    for i, ch in enumerate(reader):
        n_rows += len(ch)
        ts = pd.to_datetime(ch["# Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        lo, hi = ts.min(), ts.max()
        t_min = lo if t_min is None or lo < t_min else t_min
        t_max = hi if t_max is None or hi > t_max else t_max

        mmsi_all.update(ch["MMSI"].unique())
        ct = ch["Ship type"].isin(CARGO_TANKER)
        mmsi_ct.update(ch.loc[ct, "MMSI"].unique())
        ship_type.update(ch["Ship type"].fillna("NA").value_counts().to_dict())
        nav_status.update(ch["Navigational status"].fillna("NA").value_counts().to_dict())
        mobile.update(ch["Type of mobile"].fillna("NA").value_counts().to_dict())
        bad_pos += int((~ch["Latitude"].between(-90, 90) | ~ch["Longitude"].between(-180, 180)).sum())
        sog_missing += int(ch["SOG"].isna().sum())

        sub = ch[ct].copy()
        sub["Timestamp"] = ts[ct]
        for p, (la0, la1, lo0, lo1) in PORTS.items():
            m = sub["Latitude"].between(la0, la1) & sub["Longitude"].between(lo0, lo1)
            if m.any():
                port_rows[p].append(sub[m].drop(columns=["# Timestamp"]))
        print(f"chunk {i + 1}: {n_rows:,} rows read", flush=True)

    port_stats = []
    for p, parts in port_rows.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=USECOLS)
        if len(df):
            df = df.sort_values(["MMSI", "Timestamp"])
            try:
                df.to_parquet(out / "slices" / f"{p}.parquet", index=False)
            except ImportError:  # no pyarrow: fall back to compressed CSV
                df.to_csv(out / "slices" / f"{p}.csv.gz", index=False)
        stat = df[df["SOG"] < 0.5] if len(df) else df
        port_stats.append({
            "port": p,
            "rows": len(df),
            "vessels": df["MMSI"].nunique() if len(df) else 0,
            "vessels_stationary": stat["MMSI"].nunique() if len(stat) else 0,
            "vessels_at_anchor_status": df.loc[df["Navigational status"] == "At anchor", "MMSI"].nunique() if len(df) else 0,
            "vessels_moored_status": df.loc[df["Navigational status"] == "Moored", "MMSI"].nunique() if len(df) else 0,
        })
    pd.DataFrame(port_stats).to_csv(out / "ports.csv", index=False)

    summary = {
        "file": src.name, "rows": n_rows,
        "time_min": str(t_min), "time_max": str(t_max),
        "unique_mmsi": len(mmsi_all), "unique_mmsi_cargo_tanker": len(mmsi_ct),
        "rows_bad_position": bad_pos, "rows_sog_missing": sog_missing,
        "type_of_mobile": dict(mobile.most_common()),
        "ship_type": dict(ship_type.most_common(30)),
        "navigational_status": dict(nav_status.most_common()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    files = day_files()
    print(f"Found {len(files)} CSV file(s) to inspect in {DATA}: " + ", ".join(f.name for f in files))
    for f in files:
        main(f)
