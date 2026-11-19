"""Bayesian calibration of the port model from detected stays. Run after src/stays.py:
    python src/calibrate.py

Builds port calls from stays, then fits two hierarchical models in PyMC:

  1. Arrivals:   calls per day at port p ~ Poisson(lambda_p),
                 log lambda_p ~ Normal(mu_lambda, sigma_lambda)
  2. Berth time: log(hours) ~ Normal(alpha_p + beta * z(length), sigma_p),
                 alpha_p ~ Normal(mu_alpha, tau_alpha), log sigma_p partially pooled
                 Stays cut off by the data window are right-censored: the
                 observed time is a lower bound, so they enter through the
                 survival function, not the density.

Partial pooling lets ports with few calls borrow strength from the others.

Writes:
  data/calls.csv                one row per port call (wait, berth time, censoring)
  data/posterior_draws.csv      parameter draws, used by the SimPy model
  data/calibration_summary.csv  posterior summary per port
  figures/05_posterior.png      arrival rates and berth times per port
  figures/06_ppc.png            posterior predictive checks
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parent.parent
DATA, FIG = PROJECT / "data", PROJECT / "figures"
CALL_GAP_H = 6  # stays of the same ship less than 6 h apart belong to one call

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#c9dcf3"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


# ---------------------------------------------------------------- port calls
def build_calls(stays: pd.DataFrame) -> pd.DataFrame:
    """Group a ship's consecutive stays at one port into a call: anchorage wait, then berth."""
    calls = []
    for (port, mmsi), g in stays.sort_values("start").groupby(["port", "MMSI"]):
        call_id = (g["start"] - g["end"].shift()).dt.total_seconds().div(3600).gt(CALL_GAP_H).cumsum()
        for _, c in g.groupby(call_id):
            berth = c[c.kind == "berth"]
            anch = c[c.kind == "anchorage"]
            if berth.empty:
                continue  # anchored only (e.g. still waiting at the end): no berth time
            calls.append({
                "port": port, "MMSI": mmsi, "length_m": c.length_m.iat[0],
                "arrival": c.start.min(),
                "wait_h": anch.hours.sum() if len(anch) and anch.start.min() < berth.start.min() else 0.0,
                "berth_h": (berth.end.max() - berth.start.min()).total_seconds() / 3600,
                "arrival_observed": not c.left_censored.iat[0],
                "censored": bool(berth.left_censored.iat[0] or berth.right_censored.iat[-1]),
                "whole_window": bool(c.left_censored.iat[0] and c.right_censored.iat[-1]),
            })
    return pd.DataFrame(calls)


def daily_counts(calls: pd.DataFrame, t0, t1, ports) -> pd.DataFrame:
    """Arrivals per port per full day (only arrivals seen inside the window count)."""
    days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
    a = calls[calls.arrival_observed].copy()
    a["day"] = a.arrival.dt.normalize()
    tab = a.groupby(["port", "day"]).size().unstack(fill_value=0)
    return tab.reindex(index=ports, columns=days, fill_value=0)


# ---------------------------------------------------------------- the model
def fit(calls: pd.DataFrame, counts: pd.DataFrame, ports):
    # laid-up ships present for the whole window carry no information on cargo operations
    svc = calls[~calls.whole_window].reset_index(drop=True)
    p_idx = svc.port.map({p: i for i, p in enumerate(ports)}).to_numpy()
    len_mu, len_sd = np.log(svc.length_m).mean(), np.log(svc.length_m).std()
    z_len = ((np.log(svc.length_m) - len_mu) / len_sd).to_numpy()
    log_h = np.log(svc.berth_h.to_numpy())
    # censored rows: upper bound = observed value (true time is at least this long)
    upper = np.where(svc.censored, log_h, np.inf)

    coords = {"port": ports, "day": [str(d.date()) for d in counts.columns], "call": np.arange(len(svc))}
    with pm.Model(coords=coords) as model:
        # arrivals
        mu_lam = pm.Normal("mu_log_lambda", np.log(4), 1.0)
        sd_lam = pm.HalfNormal("sigma_log_lambda", 0.5)
        z_lam = pm.Normal("z_lambda", 0, 1, dims="port")
        lam = pm.Deterministic("lambda", pm.math.exp(mu_lam + sd_lam * z_lam), dims="port")
        pm.Poisson("arrivals", lam[:, None], observed=counts.to_numpy(), dims=("port", "day"))

        # berth time, log hours
        mu_a = pm.Normal("mu_alpha", np.log(12), 1.0)
        tau_a = pm.HalfNormal("tau_alpha", 0.5)
        z_a = pm.Normal("z_alpha", 0, 1, dims="port")
        alpha = pm.Deterministic("alpha", mu_a + tau_a * z_a, dims="port")
        beta = pm.Normal("beta_length", 0, 0.5)
        # spread of berth times per port, partially pooled on the log scale
        mu_ls = pm.Normal("mu_log_sigma", np.log(0.8), 0.5)
        tau_ls = pm.HalfNormal("tau_log_sigma", 0.3)
        z_s = pm.Normal("z_sigma", 0, 1, dims="port")
        sigma = pm.Deterministic("sigma", pm.math.exp(mu_ls + tau_ls * z_s), dims="port")
        mean = alpha[p_idx] + beta * z_len
        pm.Censored("log_berth_h", pm.Normal.dist(mean, sigma[p_idx]), lower=None, upper=upper,
                    observed=log_h, dims="call")
        pm.Deterministic("median_berth_h", pm.math.exp(alpha), dims="port")

        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.99, random_seed=7)
        pm.sample_posterior_predictive(idata, var_names=["arrivals"], random_seed=7,
                                       extend_inferencedata=True)
    return model, idata, svc, (len_mu, len_sd)


# ---------------------------------------------------------------- outputs
def draws(idata, name, group="posterior"):
    """Posterior draws as a numpy array, chains flattened: (n_draws, *dims). Works on PyMC 5 and 6."""
    v = np.asarray(idata[group][name])
    return v.reshape(-1, *v.shape[2:])


def rhat(idata, name):
    v = draws(idata, name)
    chains = np.asarray(idata["posterior"][name])
    m, n = chains.shape[:2]
    w = chains.var(axis=1, ddof=1).mean(axis=0)
    b = n * chains.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def interval(v, q=0.9):
    return v.mean(0), np.quantile(v, (1 - q) / 2, 0), np.quantile(v, (1 + q) / 2, 0)


def save_draws(idata, ports, len_scale):
    lam, alpha, sig = draws(idata, "lambda"), draws(idata, "alpha"), draws(idata, "sigma")
    rows = {"draw": np.arange(len(lam))}
    for i, p in enumerate(ports):
        rows[f"lambda_{p}"] = lam[:, i]
        rows[f"alpha_{p}"] = alpha[:, i]
        rows[f"sigma_{p}"] = sig[:, i]
    rows["beta_length"] = draws(idata, "beta_length")
    rows["log_length_mean"], rows["log_length_sd"] = len_scale
    pd.DataFrame(rows).to_csv(DATA / "posterior_draws.csv", index=False)


def summary_table(idata, calls, counts, ports):
    lm, l5, l95 = interval(draws(idata, "lambda"))
    bm, b5, b95 = interval(draws(idata, "median_berth_h"))
    rh = np.maximum(rhat(idata, "lambda"), rhat(idata, "median_berth_h"))
    rows = []
    for i, p in enumerate(ports):
        c = calls[calls.port == p]
        rows.append({
            "port": p, "calls": len(c), "censored": int(c.censored.sum()),
            "arrivals_per_day_obs": round(counts.loc[p].mean(), 2),
            "lambda_mean": lm[i], "lambda_q5": l5[i], "lambda_q95": l95[i],
            "median_berth_h": bm[i], "berth_q5": b5[i], "berth_q95": b95[i],
            "max_ships_at_berth": None, "waits_obs": int((c.wait_h > 0).sum()),
            "r_hat_max": float(rh[i]),
        })
    return pd.DataFrame(rows)


def interval_plot(ax, ports, mid, lo, hi, obs=None, xlabel=""):
    y = np.arange(len(ports))
    ax.hlines(y, lo, hi, color=BLUE, lw=2.5)
    ax.plot(mid, y, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=3)
    if obs is not None:
        ax.plot(obs, y, "D", color=INK, ms=5, zorder=4)
    ax.set_yticks(y, [p.capitalize() for p in ports])
    ax.set_ylim(len(ports) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def fig_posterior(tab, svc, ports):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    interval_plot(a1, ports, tab.lambda_mean, tab.lambda_q5, tab.lambda_q95,
                  tab.arrivals_per_day_obs, "Arrivals per day")
    a1.set_title("Arrival rate")
    a1.set_xlim(left=0)
    obs_med = [svc[(svc.port == p) & ~svc.censored].berth_h.median() for p in ports]
    interval_plot(a2, ports, tab.median_berth_h, tab.berth_q5, tab.berth_q95, obs_med, "Hours")
    a2.set_title("Median time at berth")
    a2.set_xlim(left=0)
    h = [Line2D([], [], color=BLUE, lw=2.5, marker="o", ms=7, mec=SURFACE, label="Posterior mean, 90% interval"),
         Line2D([], [], color=INK, ls="", marker="D", ms=5, label="Naive estimate (complete data only)")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=2, frameon=False, fontsize=8)
    fig.text(0.01, 0.97, "What a week of AIS data says about each port", fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.905, "Naive medians drop stays cut off by the data window; the model counts them as lower bounds.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "05_posterior.png", dpi=200)
    plt.close(fig)


def fig_ppc(idata, counts, svc, ports, len_scale):
    rng = np.random.default_rng(3)
    pick = rng.choice(len(draws(idata, "sigma")), 400, replace=False)
    alpha, beta, sig = draws(idata, "alpha")[pick], draws(idata, "beta_length")[pick], draws(idata, "sigma")[pick]
    fig, axes = plt.subplots(2, len(ports), figsize=(2.3 * len(ports), 5.2), sharey="row")
    max_k = int(counts.to_numpy().max()) + 4
    ks = np.arange(max_k + 1)
    pred = draws(idata, "arrivals", "posterior_predictive")
    for i, p in enumerate(ports):
        # arrivals: predicted distribution of daily counts vs observed days
        ax = axes[0, i]
        freq = np.stack([np.bincount(d[i], minlength=max_k + 1)[:max_k + 1] / counts.shape[1]
                         for d in pred[rng.choice(len(pred), 400)]])
        ax.fill_between(ks, np.quantile(freq, 0.05, 0), np.quantile(freq, 0.95, 0),
                        step="mid", color=BAND, lw=0)
        obs = np.bincount(counts.loc[p].to_numpy(), minlength=max_k + 1)[:max_k + 1] / counts.shape[1]
        ax.plot(ks, obs, "o", color=INK, ms=4)
        ax.set_title(p.capitalize())
        ax.set_xlabel("Arrivals in a day")
        # berth time: predicted distribution vs complete (dots) and censored (hollow) stays
        ax = axes[1, i]
        s = svc[svc.port == p]
        z = (np.log(s.length_m.to_numpy()) - len_scale[0]) / len_scale[1]
        mu = alpha[:, i][:, None] + beta[:, None] * z[None, :]
        sim = np.exp(rng.normal(mu, sig[:, i][:, None])).ravel()
        bins = np.logspace(0, np.log10(200), 25)
        ax.hist(sim, bins=bins, weights=np.full(len(sim), 1 / len(sim)), color=BAND)  # share per log bin
        top = ax.get_ylim()[1]
        yj = rng.uniform(0.04, 0.16, len(s)) * top
        ax.scatter(s.berth_h[~s.censored], yj[~s.censored.to_numpy()], s=16, color=BLUE, zorder=3)
        ax.scatter(s.berth_h[s.censored], yj[s.censored.to_numpy()], s=16, facecolor=SURFACE,
                   edgecolor=BLUE, lw=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 24, 96], ["1 h", "5 h", "1 d", "4 d"])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].set_ylabel("Share of days")
    axes[1, 0].set_ylabel("Berth time")
    h = [Line2D([], [], color=BAND, lw=8, label="Posterior predictive (90% band / density)"),
         Line2D([], [], color=INK, ls="", marker="o", ms=4, label="Observed days"),
         Line2D([], [], color=BLUE, ls="", marker="o", ms=4, label="Complete stays"),
         Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE, mec=BLUE, label="Censored stays")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.975), ncol=4, frameon=False, fontsize=8)
    fig.text(0.01, 0.975, "Does the fitted model reproduce the data?", fontsize=13, fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "06_ppc.png", dpi=200)
    plt.close(fig)


def main():
    stays = pd.read_csv(DATA / "stays.csv", parse_dates=["start", "end"])
    ports = sorted(stays.port.unique())
    calls = build_calls(stays)
    calls.to_csv(DATA / "calls.csv", index=False)
    t0, t1 = stays.start.min(), stays.end.max()
    counts = daily_counts(calls, t0, t1, ports)
    # drop partial first/last days so every day counted is a full day
    counts = counts.loc[:, (counts.columns > t0.normalize()) | (t0 == t0.normalize())]
    counts = counts.loc[:, counts.columns + pd.Timedelta(days=1) <= t1 + pd.Timedelta(minutes=30)]
    print(f"{len(calls)} calls, {int(calls.censored.sum())} censored, {counts.shape[1]} full days")

    model, idata, svc, len_scale = fit(calls, counts, ports)
    sg = draws(idata, "sigma").mean(0)
    print("berth-time spread sigma per port:", dict(zip(ports, sg.round(2))))
    b = draws(idata, "beta_length")
    print(f"length effect beta: {b.mean():.2f} (90% interval {np.quantile(b, .05):.2f} to {np.quantile(b, .95):.2f})")

    tab = summary_table(idata, calls, counts, ports)
    tab["max_ships_at_berth"] = [max_concurrent(stays[(stays.port == p) & (stays.kind == "berth")]) for p in ports]
    tab.round(2).to_csv(DATA / "calibration_summary.csv", index=False)
    save_draws(idata, ports, len_scale)
    fig_posterior(tab, svc, ports)
    fig_ppc(idata, counts, svc, ports, len_scale)
    print(tab.round(2).to_string(index=False))


def max_concurrent(s: pd.DataFrame) -> int:
    ev = pd.concat([pd.Series(1, s.start), pd.Series(-1, s.end)]).sort_index(kind="stable")
    return int(ev.cumsum().max()) if len(ev) else 0


if __name__ == "__main__":
    main()
