"""Clean the daily port slices, stitch the days together, and detect stays.

A stay is a continuous period in which a large ship is stationary inside a
port box. Each stay is labelled berth or anchorage, and flagged as censored
when it touches the start or end of the data window.

Run from anywhere:
    python src/stays.py
Writes data/stays.csv and data/tracks.parquet (or tracks.csv.gz).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
PORTS = ["aalborg", "aarhus", "esbjerg", "fredericia", "kalundborg"]

MIN_LENGTH_M = 80        # drop harbour craft, bunker barges, tugs
STATIONARY_KN = 0.5      # speed over ground below this = stationary
MAX_GAP_MIN = 30         # a longer silence breaks a stay
MERGE_GAP_MIN = 30       # rejoin stays split by a short drift ...
MERGE_DIST_M = 300       # ... if the ship is still in the same place
MIN_STAY_H = 1.0
CENSOR_MARGIN_MIN = 30   # stays this close to the data edges are censored


def load_port(port: str) -> pd.DataFrame:
    parts = []
    for day_dir in sorted((DATA / "inspect").glob("aisdk-*")):
        f = next(iter((day_dir / "slices").glob(f"{port}.*")), None)
        if f is None:
            continue
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["port"] = port
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Type of mobile"] == "Class A"]
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df = df.dropna(subset=["SOG"])
    lengths = df.groupby("MMSI")["Length"].median()
    big = lengths[lengths >= MIN_LENGTH_M].index
    df = df[df["MMSI"].isin(big)]
    df = df.drop_duplicates(["MMSI", "Timestamp"]).sort_values(["MMSI", "Timestamp"])
    return df.reset_index(drop=True)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def detect_stays(df: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
    rows = []
    for (port, mmsi), g in df.groupby(["port", "MMSI"]):
        stat = (g["SOG"] < STATIONARY_KN).to_numpy()
        t = g["Timestamp"].to_numpy()
        gap = np.diff(t, prepend=t[0]) > np.timedelta64(MAX_GAP_MIN, "m")
        # new segment when the stationary flag flips or the signal drops out
        seg = np.cumsum(np.r_[True, (stat[1:] != stat[:-1]) | gap[1:]])
        g = g.assign(seg=seg, stat=stat)
        eps = (g[g["stat"]].groupby("seg")
               .agg(start=("Timestamp", "min"), end=("Timestamp", "max"),
                    lat=("Latitude", "median"), lon=("Longitude", "median"),
                    anchor_frac=("Navigational status", lambda s: (s == "At anchor").mean()))
               .reset_index(drop=True))
        # merge stays split by a short drift in the same place
        merged = []
        for e in eps.itertuples(index=False):
            if merged:
                p = merged[-1]
                close = haversine_m(p["lat"], p["lon"], e.lat, e.lon) < MERGE_DIST_M
                if close and (e.start - p["end"]) < pd.Timedelta(minutes=MERGE_GAP_MIN):
                    p["end"] = e.end
                    continue
            merged.append(e._asdict())
        for e in merged:
            hours = (e["end"] - e["start"]).total_seconds() / 3600
            if hours < MIN_STAY_H:
                continue
            rows.append({
                "port": port, "MMSI": mmsi,
                "ship_type": g["Ship type"].mode().iat[0],
                "length_m": g["Length"].median(),
                "start": e["start"], "end": e["end"], "hours": round(hours, 2),
                "lat": e["lat"], "lon": e["lon"],
                "kind": "anchorage" if e["anchor_frac"] > 0.5 else "berth",
                "left_censored": e["start"] - t_start < pd.Timedelta(minutes=CENSOR_MARGIN_MIN),
                "right_censored": t_end - e["end"] < pd.Timedelta(minutes=CENSOR_MARGIN_MIN),
            })
    return pd.DataFrame(rows)


def main() -> None:
    tracks = pd.concat([clean(load_port(p)) for p in PORTS], ignore_index=True)
    t_start, t_end = tracks["Timestamp"].min(), tracks["Timestamp"].max()
    stays = detect_stays(tracks, t_start, t_end).sort_values(["port", "start"])
    stays.to_csv(DATA / "stays.csv", index=False)
    keep = ["port", "MMSI", "Timestamp", "Latitude", "Longitude", "SOG", "Ship type", "Length"]
    try:
        tracks[keep].to_parquet(DATA / "tracks.parquet", index=False)
    except ImportError:
        tracks[keep].to_csv(DATA / "tracks.csv.gz", index=False)
    days = (t_end - t_start).total_seconds() / 86400
    print(f"{len(tracks):,} clean rows, {tracks['MMSI'].nunique()} ships, {days:.1f} days")
    print(stays.groupby(["port", "kind"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
