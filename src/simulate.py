"""SimPy model of one port, driven by the Bayesian posterior. Run after src/calibrate.py:
    python src/simulate.py

Each replication draws one parameter set from data/posterior_draws.csv, so the
results carry parameter uncertainty as well as day-to-day randomness:

  ships arrive          Poisson process, rate lambda_p per day
  ship length           resampled from the port's observed calls
  time at berth         lognormal: exp(alpha_p + beta * z(length) + sigma * noise)
  berths                SimPy Resource; capacity = most ships seen at berth at once
  no free berth         ship waits at anchor (first come, first served)

Writes:
  data/sim_baseline.csv     one row per replication: occupancy, waits, utilisation
  figures/07_validation.png simulated vs observed occupancy and waiting
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from matplotlib.lines import Line2D

PORT = "aarhus"           # change to any calibrated port
N_REPS = 500
WARMUP_D = 10             # days discarded so the port starts in a typical state
HORIZON_D = 60            # days measured per replication

PROJECT = Path(__file__).resolve().parent.parent
DATA, FIG = PROJECT / "data", PROJECT / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#c9dcf3"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


# ---------------------------------------------------------------- the model
class Port:
    def __init__(self, env, berths, params, lengths, rng):
        self.env, self.rng, self.p, self.lengths = env, rng, params, lengths
        self.berths = simpy.Resource(env, capacity=berths)
        self.occ_time = np.zeros(berths + 1)   # hours spent with k ships at berth
        self.queue_time = 0.0                  # ship-hours waiting at anchor
        self.last_t, self.waits = 0.0, []

    def _tick(self):
        """Accumulate time in the current occupancy state (measured period only)."""
        t0 = max(self.last_t, WARMUP_D * 24)
        if self.env.now > t0:
            dt = self.env.now - t0
            self.occ_time[self.berths.count] += dt
            self.queue_time += len(self.berths.queue) * dt
        self.last_t = self.env.now

    def berth_hours(self, length_m):
        p = self.p
        z = (np.log(length_m) - p["log_length_mean"]) / p["log_length_sd"]
        return float(np.exp(p["alpha"] + p["beta_length"] * z + p["sigma"] * self.rng.normal()))

    def ship(self):
        arrive = self.env.now
        hours = self.berth_hours(self.rng.choice(self.lengths))
        self._tick()
        with self.berths.request() as req:
            yield req
            self._tick()
            if arrive >= WARMUP_D * 24:
                self.waits.append(self.env.now - arrive)
            yield self.env.timeout(hours)
            self._tick()
        self._tick()

    def arrivals(self):
        rate_h = self.p["lambda"] / 24
        while True:
            yield self.env.timeout(self.rng.exponential(1 / rate_h))
            self.env.process(self.ship())


def run_once(params, berths, lengths, seed):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    port = Port(env, berths, params, lengths, rng)
    env.process(port.arrivals())
    end = (WARMUP_D + HORIZON_D) * 24
    env.run(until=end)
    port.env = env
    port._tick()
    hours = HORIZON_D * 24
    w = np.array(port.waits)
    occ_share = port.occ_time / hours
    return {
        "lambda": params["lambda"], "median_berth_h": np.exp(params["alpha"]),
        "calls": len(w),
        "mean_at_berth": float((np.arange(berths + 1) * occ_share).sum()),
        "utilisation": float((np.arange(berths + 1) * occ_share).sum() / berths),
        "share_waiting": float((w > 0.01).mean()) if len(w) else 0.0,
        "mean_wait_h": float(w.mean()) if len(w) else 0.0,
        "p90_wait_h": float(np.quantile(w, 0.9)) if len(w) else 0.0,
        "mean_at_anchor": port.queue_time / hours,
        **{f"occ_{k}": occ_share[k] for k in range(berths + 1)},
    }


def load_inputs(port):
    draws = pd.read_csv(DATA / "posterior_draws.csv")
    summary = pd.read_csv(DATA / "calibration_summary.csv").set_index("port")
    calls = pd.read_csv(DATA / "calls.csv")
    lengths = calls.loc[calls.port == port, "length_m"].dropna().to_numpy()
    params = pd.DataFrame({
        "lambda": draws[f"lambda_{port}"], "alpha": draws[f"alpha_{port}"],
        "beta_length": draws.beta_length, "sigma": draws[f"sigma_{port}"],
        "log_length_mean": draws.log_length_mean, "log_length_sd": draws.log_length_sd,
    })
    return params, int(summary.loc[port, "max_ships_at_berth"]), lengths, calls


def simulate(params, berths, lengths, n_reps=N_REPS, seed=0, demand=1.0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(params), n_reps, replace=len(params) < n_reps)
    rows = []
    for r, i in enumerate(idx):
        p = params.iloc[i].to_dict()
        p["lambda"] *= demand
        rows.append(run_once(p, berths, lengths, seed * 100_000 + r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- observed
def observed_occupancy(port, berths):
    """Share of time with k ships at berth, from the detected stays (15-min grid)."""
    stays = pd.read_csv(DATA / "stays.csv", parse_dates=["start", "end"])
    t0, t1 = stays.start.min().ceil("15min"), stays.end.max().floor("15min")
    s = stays[(stays.port == port) & (stays.kind == "berth")]
    grid = pd.date_range(t0, t1, freq="15min")
    occ = np.array([((s.start <= t) & (s.end > t)).sum() for t in grid])
    return np.bincount(np.minimum(occ, berths), minlength=berths + 1) / len(occ), occ.mean()


# ---------------------------------------------------------------- validation figure
def fig_validation(sim, obs_share, obs_mean, calls, port, berths):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4.3),
                                     gridspec_kw={"width_ratios": [2.2, 1, 1]})
    k = np.arange(berths + 1)
    occ = sim[[f"occ_{i}" for i in k]].to_numpy()
    a1.fill_between(k, np.quantile(occ, 0.05, 0), np.quantile(occ, 0.95, 0), color=BAND, lw=0,
                    step="mid")
    a1.plot(k, np.median(occ, 0), color=BLUE, lw=2, drawstyle="steps-mid")
    a1.plot(k, obs_share, "D", color=INK, ms=6, zorder=3)
    a1.set_title("Ships at berth at the same time")
    a1.set_xlabel("Number of ships at berth")
    a1.set_ylabel("Share of time")
    a1.set_xticks(k)

    def dist(ax, values, observed, title, xlabel, fmt):
        ax.hist(values, bins=30, color=BAND, weights=np.full(len(values), 1 / len(values)))
        ax.axvline(np.median(values), color=BLUE, lw=2)
        ax.axvline(observed, color=INK, lw=2, ls=(0, (3, 2)))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.grid(False)
        ax.text(0.98, 0.95, f"Observed {fmt(observed)}\nSimulated {fmt(np.median(values))}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)

    dist(a2, sim.mean_at_berth, obs_mean, "Average ships at berth", "Ships", lambda v: f"{v:.1f}")
    c = calls[(calls.port == port) & calls.arrival_observed]
    obs_wait = (c.wait_h > 0).mean()
    dist(a3, 100 * sim.share_waiting, 100 * obs_wait, "Calls that wait at anchor", "% of calls",
         lambda v: f"{v:.0f}%")

    h = [Line2D([], [], color=BLUE, lw=2, label="Simulated median"),
         Line2D([], [], color=BAND, lw=8, label="Simulated 90% band / spread"),
         Line2D([], [], color=INK, marker="D", ls="", ms=5, label="Observed week"),
         Line2D([], [], color=INK, lw=2, ls=(0, (3, 2)), label="Observed week (single value)")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.80), ncol=4, frameon=False, fontsize=8)
    fig.text(0.01, 0.97, f"Does the simulation reproduce {port.capitalize()}?",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.905, f"{len(sim)} simulated 60-day runs, each with its own posterior draw; "
             f"{berths} berths.", fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(FIG / "07_validation.png", dpi=200)
    plt.close(fig)


def main():
    params, berths, lengths, calls = load_inputs(PORT)
    print(f"{PORT}: {berths} berths, {len(lengths)} observed ship lengths, {N_REPS} replications")
    sim = simulate(params, berths, lengths)
    sim.to_csv(DATA / "sim_baseline.csv", index=False)
    obs_share, obs_mean = observed_occupancy(PORT, berths)
    fig_validation(sim, obs_share, obs_mean, calls, PORT, berths)
    q = sim[["mean_at_berth", "utilisation", "share_waiting", "mean_wait_h", "p90_wait_h"]].quantile([.05, .5, .95])
    print(q.round(3).to_string())
    print(f"observed: mean at berth {obs_mean:.2f}")


if __name__ == "__main__":
    main()
