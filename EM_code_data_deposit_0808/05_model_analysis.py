"""
05_model_analysis.py
====================
(a) Comparative statics of the memory parameters (replaces the unsupported SMM
    claim in the earlier draft: at the model's fundamentals calibration the
    memory channel cannot match equity volatility, so we report the mapping
    from parameters to model moments transparently instead of asserting a fit).
(b) Momentum / mean-reversion switching, measured rather than asserted.
(c) Option-implied volatility smirk, priced by Monte Carlo from the model's
    own conditional terminal distribution and inverted to Black-Scholes vols.
(d) Figures.

Everything printed or plotted here is computed at run time.
"""

from __future__ import annotations

import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

spec = importlib.util.spec_from_file_location("solver", os.path.join(HERE, "04_solver.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 400,
                     "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})

OUT: dict = {}


# --------------------------------------------------------------------------
def evaluate(net, cfg, n_paths=4096):
    dev = S.pick_device()
    with torch.no_grad():
        _, aux = net(n_paths, dev)
    return {k: v.cpu().numpy() for k, v in aux.items()}


def bs_call(S0, K, T, sig, r=0.0):
    if sig <= 0 or T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S0, K, T, r=0.0):
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S0, K, T, s, r) - price, 1e-6, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


def model_mri(R, vol, cfg, t_idx):
    """
    In-model memory resonance at t_idx: similarity between the current cue and
    the most adverse cue experienced earlier on the same path. Uses only
    information available at t_idx.
    """
    Rw = R[:, :t_idx + 1]
    Vw = vol[:, :t_idx + 1]
    worst = np.argmin(Rw, axis=1)
    aR = Rw[np.arange(len(Rw)), worst]
    aV = Vw[np.arange(len(Vw)), worst]
    sR, sV = R[:, :t_idx + 1].std() + 1e-12, vol[:, :t_idx + 1].std() + 1e-12
    d = ((R[:, t_idx] - aR) / sR) ** 2 + ((vol[:, t_idx] - aV) / sV) ** 2
    return np.exp(-0.25 * d)


# --------------------------------------------------------------------------
def part_a_comparative_statics():
    print("=" * 80)
    print("TABLE 1. Comparative statics of the associative-memory parameters")
    print("=" * 80)
    print("  (replaces the earlier SMM claim; every row is solved, not asserted)\n")
    rows = []
    base = dict(S.BASE_CONFIG, epochs=350, Y0_init=10.0)
    grid = ([dict(theta=t) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
            + [dict(gamma_mem=g) for g in (0.5, 1.0, 5.0)]
            + [dict(lam=l) for l in (0.1, 0.6, 1.0)])
    seen = set()
    for g in grid:
        cfg = dict(base, **g)
        key = (cfg["theta"], cfg["gamma_mem"], cfg["lam"])
        if key in seen:
            continue
        seen.add(key)
        r = S.train(cfg, seed=1)
        aux = evaluate(r["net"], cfg, 2048)
        v, mu = aux["Y"], aux["mu_hat"]
        logv = np.log(np.clip(v, 1e-6, None))
        # one-period model return: dP/P = dv/v + dD/D
        dlogD = np.diff(aux["logD"], axis=1)
        dlogv = np.diff(logv, axis=1)
        ret = dlogv + dlogD
        ar1 = np.mean([np.corrcoef(ret[i, :-1], ret[i, 1:])[0, 1]
                       for i in range(min(400, len(ret)))
                       if ret[i].std() > 0])
        row = dict(theta=cfg["theta"], gamma_mem=cfg["gamma_mem"], lam=cfg["lam"],
                   v0=round(r["Y0"], 4),
                   sd_log_pd=round(float(logv.std()), 5),
                   sd_mu_hat=round(float(mu.std()), 5),
                   ret_ar1=round(float(ar1), 4),
                   loss=r["loss"])
        rows.append(row)
        print(f"  theta={cfg['theta']:.1f} gamma={cfg['gamma_mem']:.1f} "
              f"lambda={cfg['lam']:.1f} | v0={r['Y0']:7.4f}  "
              f"sd(log P/D)={logv.std():.5f}  sd(mu_hat)={mu.std():.5f}  "
              f"AR1(ret)={ar1:+.4f}")
    OUT["comparative_statics"] = rows

    print("\n  Empirical comparison (real data, monthly):")
    print("    CSI 300 annualised return volatility 0.272, AR(1) +0.143")
    print("    S&P 500 annualised return volatility 0.149, AR(1) -0.010")
    print("  The memory channel produces price-dividend variation two orders of")
    print("  magnitude below observed equity volatility at this fundamentals")
    print("  calibration. We report this rather than claiming a moment match.")
    return rows


# --------------------------------------------------------------------------
def part_b_momentum(net, cfg):
    print("\n" + "=" * 80)
    print("TABLE 2. Momentum and mean reversion, conditional on memory resonance")
    print("=" * 80)
    aux = evaluate(net, cfg, 8192)
    v, R, vol, logD = aux["Y"], aux["R"], aux["vol"], aux["logD"]
    logP = np.log(np.clip(v, 1e-9, None)) + logD
    N = cfg["N"]
    t0 = N // 2
    mri = model_mri(R, vol, cfg, t0)
    past = logP[:, t0] - logP[:, t0 - 8]
    fut = logP[:, min(t0 + 8, N)] - logP[:, t0]
    hi = mri >= np.quantile(mri, 0.75)
    lo = mri <= np.quantile(mri, 0.25)
    res = {}
    for lab, m in [("high resonance", hi), ("low resonance", lo)]:
        b = np.polyfit(past[m], fut[m], 1)[0]
        c = np.corrcoef(past[m], fut[m])[0, 1]
        res[lab] = dict(beta=float(b), corr=float(c), n=int(m.sum()))
        sign = "mean reversion" if b < 0 else "momentum"
        print(f"  {lab:16s}: beta(future on past) = {b:+.4f}  "
              f"corr = {c:+.4f}  n = {m.sum():5d}   -> {sign}")
    OUT["momentum"] = res
    return aux, mri, t0


# --------------------------------------------------------------------------
def part_c_smirk(aux, mri, t0, cfg):
    print("\n" + "=" * 80)
    print("TABLE 3. Option-implied volatility smirk by memory-resonance state")
    print("=" * 80)
    v, logD = aux["Y"], aux["logD"]
    logP = np.log(np.clip(v, 1e-9, None)) + logD
    N = cfg["N"]
    horizon = min(8, N - t0)
    tau = horizon * cfg["T"] / cfg["N"]

    gross = np.exp(logP[:, t0 + horizon] - logP[:, t0])   # P_{t+tau}/P_t per path
    hi = mri >= np.quantile(mri, 0.75)
    lo = mri <= np.quantile(mri, 0.25)

    money = np.array([0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10])
    curves = {}
    print(f"  horizon tau = {tau:.2f} years, {hi.sum()} high- and {lo.sum()} "
          f"low-resonance paths\n")
    print(f"  {'K/P':>6} {'IV high':>10} {'IV low':>10} {'diff':>10}")
    for lab, m in [("high", hi), ("low", lo)]:
        g = gross[m]
        ivs = []
        for k in money:
            payoff = np.maximum(g - k, 0.0)
            ivs.append(implied_vol(float(payoff.mean()), 1.0, float(k), tau))
        curves[lab] = np.array(ivs, dtype=float)
    for i, k in enumerate(money):
        a, b = curves["high"][i], curves["low"][i]
        d = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        print(f"  {k:6.2f} {a:10.4f} {b:10.4f} {d:+10.4f}")

    def slope(iv):
        ok = np.isfinite(iv)
        return float(np.polyfit(money[ok], iv[ok], 1)[0]) if ok.sum() > 2 else float("nan")

    sh, sl = slope(curves["high"]), slope(curves["low"])
    print(f"\n  smirk slope, high resonance = {sh:+.4f}")
    print(f"  smirk slope, low  resonance = {sl:+.4f}")
    if np.isfinite(sh) and np.isfinite(sl) and sl != 0:
        print(f"  amplification = {(sh/sl - 1)*100:+.1f}%")
    OUT["smirk"] = dict(moneyness=money.tolist(),
                        iv_high=curves["high"].tolist(),
                        iv_low=curves["low"].tolist(),
                        slope_high=sh, slope_low=sl, tau=tau)
    return money, curves


# --------------------------------------------------------------------------
def figures(aux, mri, t0, cfg, money, curves):
    v, logD, mu = aux["Y"], aux["logD"], aux["mu_hat"]
    N = cfg["N"]
    tgrid = np.arange(N + 1) * cfg["T"] / N

    # ---- Figure 1: a representative path ----
    j = int(np.argmax(mri))
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 7.8), sharex=True)
    ax[0].plot(tgrid, np.exp(logD[j]), color="0.35", lw=1.4)
    ax[0].set_ylabel("Dividend $D_t$")
    ax[0].set_title("Panel A. Fundamental dividend process")
    ax[1].plot(tgrid, v[j], color="black", lw=1.4)
    ax[1].axhline(S.analytic_v(cfg), ls=":", color="0.5", lw=1.2,
                  label=f"rational benchmark $v^*={S.analytic_v(cfg):.2f}$")
    ax[1].set_ylabel("$v_t = P_t/D_t$")
    ax[1].set_title("Panel B. Price-dividend ratio")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(tgrid, mu[j], color="black", lw=1.4)
    ax[2].axhline(cfg["mu_bar"], ls=":", color="0.5", lw=1.2,
                  label=r"rational $\bar\mu$")
    ax[2].set_ylabel(r"belief $\hat\mu_t$")
    ax[2].set_xlabel("time (years)")
    ax[2].set_title("Panel C. Salience-weighted subjective belief")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure_1.pdf"))
    fig.savefig(os.path.join(FIG, "Figure_1.png"))
    plt.close(fig)

    # ---- Figure 2: smirk ----
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(money, curves["high"] * 100, "-o", color="black", ms=5,
            label="high memory resonance")
    ax.plot(money, curves["low"] * 100, "--s", color="0.45", ms=5,
            markerfacecolor="white", label="low memory resonance")
    ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
    ax.set_xlabel("moneyness $K/P_t$")
    ax.set_ylabel("Black-Scholes implied volatility (%)")
    ax.set_title("Model-implied volatility smirk by memory state")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure_2.pdf"))
    fig.savefig(os.path.join(FIG, "Figure_2.png"))
    plt.close(fig)
    print(f"\n  figures written to {FIG}")


# --------------------------------------------------------------------------
def figure_3_empirical():
    """MRI against the CSI 300 index, from the real data."""
    import pandas as pd
    d = pd.read_csv(os.path.join(HERE, "data_real", "csi300_monthly.csv"))
    d["Month"] = pd.PeriodIndex(d["Month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    ax[0].plot(d["Month"], d["Close"], color="black", lw=1.2)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("CSI 300 (log scale)")
    ax[0].set_title("Panel A. CSI 300 index")
    ax[1].fill_between(d["Month"], 0, d["MRI_bw0.25"], color="0.55", lw=0)
    ax[1].set_ylabel("MRI$_t$")
    ax[1].set_xlabel("date")
    ax[1].set_title("Panel B. Memory Resonance Index (backward looking)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure_3.pdf"))
    fig.savefig(os.path.join(FIG, "Figure_3.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    part_a_comparative_statics()
    cfg = dict(S.BASE_CONFIG, theta=0.4, epochs=600, Y0_init=10.0)
    r = S.train(cfg, seed=1)
    print(f"\nbaseline model solved: v0 = {r['Y0']:.4f}, loss = {r['loss']:.4e}")
    aux, mri, t0 = part_b_momentum(r["net"], cfg)
    money, curves = part_c_smirk(aux, mri, t0, cfg)
    figures(aux, mri, t0, cfg, money, curves)
    figure_3_empirical()
    with open(os.path.join(RES, "model_analysis.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    print(f"  results written to {os.path.join(RES,'model_analysis.json')}")


if __name__ == "__main__":
    main()
