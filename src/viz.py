"""Exploratory figures for the port week. Run after src/stays.py:
    python src/viz.py
Writes PNGs to figures/.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parent.parent
DATA, FIG = PROJECT / "data", PROJECT / "figures"
FIG.mkdir(exist_ok=True)

# Palette: reference categorical slots 1-2, light mode
SURFACE, INK, INK2, GRID, TRACK = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
KIND_COLOR = {"berth": "#2a78d6", "anchorage": "#eb6834"}
PORT_ORDER = ["aalborg", "aarhus", "esbjerg", "fredericia", "kalundborg"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def load():
    stays = pd.read_csv(DATA / "stays.csv", parse_dates=["start", "end"])
    p = DATA / "tracks.parquet"
    tracks = pd.read_parquet(p) if p.exists() else pd.read_csv(DATA / "tracks.csv.gz")
    tracks["Timestamp"] = pd.to_datetime(tracks["Timestamp"])
    return stays, tracks


def title(fig, main, sub):
    """Title block at a fixed size in inches; returns the top of the plot area."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, main, fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.40 / h, sub, fontsize=9, color=INK2, va="top")
    return 1 - 0.65 / h


def kind_legend(ax, loc="upper right"):
    h = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=k.capitalize())
         for k, c in KIND_COLOR.items()]
    ax.legend(handles=h, loc=loc, frameon=False, fontsize=8)


def fig_maps(stays, tracks, ports):
    """Where ships move and where they stop: berths and anchorages emerge from the data."""
    fig, axes = plt.subplots(1, len(ports), figsize=(3.3 * len(ports), 3.6))
    for ax, port in zip(np.atleast_1d(axes), ports):
        t = tracks[tracks.port == port]
        s = stays[stays.port == port]
        for _, g in t.groupby("MMSI"):
            ax.plot(g.Longitude, g.Latitude, color=TRACK, lw=0.4, alpha=0.6)
        for kind, c in KIND_COLOR.items():
            k = s[s.kind == kind]
            ax.scatter(k.lon, k.lat, s=8 + 3 * k.hours, color=c, alpha=0.75,
                       edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_aspect(1 / np.cos(np.radians(t.Latitude.mean())))
        ax.set_title(f"{port.capitalize()}  ·  {len(s)} stays", loc="left")
        ax.grid(False)
        ax.tick_params(labelsize=7)
    kind_legend(np.atleast_1d(axes)[0], loc="lower left")
    top = title(fig, "Where large ships move and stop",
          "Grey: tracks of cargo and tanker ships ≥ 80 m. Dots: stays ≥ 1 h, sized by duration.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(FIG / "01_port_maps.png", dpi=200)
    plt.close(fig)


def fig_timeline(stays, ports, t0, t1):
    """Gantt of every stay: when ships arrived, how long they stayed, and overlaps."""
    n = [stays[stays.port == p].MMSI.nunique() for p in ports]
    fig, axes = plt.subplots(len(ports), 1, figsize=(11, 1.0 + 0.16 * sum(n) + 0.3 * len(ports)),
                             sharex=True, gridspec_kw={"height_ratios": [max(k, 3) for k in n]})
    for ax, port in zip(np.atleast_1d(axes), ports):
        s = stays[stays.port == port].sort_values("start")
        ships = list(dict.fromkeys(s.MMSI))
        for _, r in s.iterrows():
            y = ships.index(r.MMSI)
            ax.barh(y, r.end - r.start, left=r.start, height=0.7,
                    color=KIND_COLOR[r.kind], edgecolor=SURFACE, linewidth=1)
            if r.left_censored:
                ax.plot(r.start, y, marker="<", color=INK2, ms=4)
            if r.right_censored:
                ax.plot(r.end, y, marker=">", color=INK2, ms=4)
        ax.set_yticks(range(len(ships)), [""] * len(ships))
        ax.set_ylim(len(ships) - 0.4, -0.6)
        ax.set_ylabel(port.capitalize(), rotation=0, ha="right", va="center",
                      color=INK, fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ax = np.atleast_1d(axes)[-1]
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    h = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=k.capitalize())
         for k, c in KIND_COLOR.items()]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 1 - 0.1 / fig.get_figheight()),
               ncol=2, frameon=False, fontsize=8)
    top = title(fig, "Every port stay, one row per ship",
          "Bars run from arrival to departure. Arrows mark stays already under way at the start, "
          "or not finished at the end, of the data.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(FIG / "02_stay_timeline.png", dpi=200)
    plt.close(fig)


def fig_occupancy(stays, ports, t0, t1):
    """How many large ships are in port at once: the quantity a berth model must reproduce."""
    grid = pd.date_range(t0.ceil("15min"), t1.floor("15min"), freq="15min")
    fig, axes = plt.subplots(len(ports), 1, figsize=(11, 1.35 * len(ports) + 1), sharex=True, sharey=True)
    for ax, port in zip(np.atleast_1d(axes), ports):
        s = stays[stays.port == port]
        for kind, c in KIND_COLOR.items():
            k = s[s.kind == kind]
            occ = [((k.start <= t) & (k.end > t)).sum() for t in grid]
            if kind == "anchorage" and not any(occ):
                continue
            ax.step(grid, occ, where="post", color=c, lw=2)
        ax.text(0.005, 0.92, port.capitalize(), transform=ax.transAxes, fontweight="bold", va="top")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
    ax = np.atleast_1d(axes)[-1]
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    h = [Line2D([], [], color=c, lw=2, label=f"Ships at {k}") for k, c in KIND_COLOR.items()]
    np.atleast_1d(axes)[0].legend(handles=h, loc="upper right", frameon=False, fontsize=8, ncol=2)
    top = title(fig, "Ships in port at the same time",
          "Count of large ships stationary at berth or at anchor, every 15 minutes.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(FIG / "03_occupancy.png", dpi=200)
    plt.close(fig)


def fig_durations(stays, ports):
    """How long a berth stay lasts; hollow points are censored (true stay is longer)."""
    s = stays[stays.kind == "berth"]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    rng = np.random.default_rng(1)
    for i, port in enumerate(ports):
        k = s[s.port == port]
        y = i + rng.uniform(-0.18, 0.18, len(k))
        cens = k.left_censored | k.right_censored
        ax.scatter(k.hours[~cens], y[~cens], s=36, color=KIND_COLOR["berth"],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter(k.hours[cens], y[cens], s=36, facecolor=SURFACE,
                   edgecolor=KIND_COLOR["berth"], linewidth=1.4, zorder=3)
        med = k.hours[~cens].median()
        if pd.notna(med):
            ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
    ax.set_yticks(range(len(ports)), [p.capitalize() for p in ports])
    ax.set_ylim(len(ports) - 0.5, -0.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 24, 48, 96], ["1 h", "2 h", "5 h", "10 h", "1 d", "2 d", "4 d"])
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], marker="o", ls="", color=KIND_COLOR["berth"], label="Complete stay"),
         Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE,
                markeredgecolor=KIND_COLOR["berth"], label="Cut off by data window"),
         Line2D([], [], color=INK, lw=2, label="Median of complete stays")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8)
    top = title(fig, "How long ships stay at berth",
          "Hollow points are lower bounds: these need a censored likelihood in the Bayesian model.")
    fig.tight_layout(rect=(0, 0, 1, top - 0.06))
    fig.savefig(FIG / "04_berth_durations.png", dpi=200)
    plt.close(fig)


def main():
    stays, tracks = load()
    ports = [p for p in PORT_ORDER if p in set(stays.port)]
    t0, t1 = tracks.Timestamp.min(), tracks.Timestamp.max()
    fig_maps(stays, tracks, ports)
    fig_timeline(stays, ports, t0, t1)
    fig_occupancy(stays, ports, t0, t1)
    fig_durations(stays, ports)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
