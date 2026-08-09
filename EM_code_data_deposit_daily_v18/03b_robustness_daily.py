"""
03b_robustness_daily.py
=======================
Daily-frequency stress tests for the CSI 300 MRI result.

Same test battery as 03_robustness.py at DAILY frequency. Window constants
are rescaled with 1 month = 21 trading days:
    bootstrap block    : 12m  -> 252 trading days
    recursive step     : 24m  -> 504 trading days
    recursive origin   : 72m  -> 1512 trading days
    anchor-delay audit : 12m  -> 252 trading days ; 1m -> 21 days
    OOS training burn  : 60m  -> 1260 trading days
Inference uses HAC with 21 lags (the 21-day vol-window overlap).
R1 influence test uses a one-step (PRESS-type) influence decomposition so it
remains tractable at N~5000; this is documented below.
"""

from __future__ import annotations

import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_real")
RES = os.path.join(HERE, "results_daily")
os.makedirs(RES, exist_ok=True)

M = "MRI_bw0.25"
BASE = [M, "R_t", "sigma_t"]
CTRL = BASE + ["EP_t", "BM_t"]
Y = "Future_ExRet"
HAC_LAGS = 21
LINES: list[str] = []
RESULTS: dict[str, dict] = {}
rng = np.random.default_rng(20260807)

D = 21          # trading days per "month" for the rescaling

# daily-analogue window constants
BLOCK = 12 * D          # 252
REC_STEP = 24 * D       # 504
REC_ORIGIN = 72 * D     # 1512
LAG_AUDIT = 12 * D      # 252
LAG_ONE = D             # 21
OOS_BURN = 60 * D       # 1260


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def run(d: pd.DataFrame, y: str = Y, xs: list[str] = CTRL, lags: int = HAC_LAGS):
    dd = d[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < 30:
        return None
    X = sm.add_constant(dd[xs], has_constant="add")
    m = sm.OLS(dd[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    m.n_obs = len(dd)
    return m


def line(tag: str, m) -> None:
    if m is None:
        emit(f"  {tag:44s} insufficient observations")
        RESULTS[tag] = {"b": None, "t": None, "adjR2": None, "N": 0}
        return
    emit(f"  {tag:44s} b={m.params[M]:+.6f}  t={m.tvalues[M]:+.2f}  "
         f"adjR2={m.rsquared_adj*100:5.3f}%  N={m.n_obs}")
    RESULTS[tag] = {"b": float(m.params[M]), "t": float(m.tvalues[M]),
                     "adjR2": float(m.rsquared_adj), "N": int(m.n_obs)}


def one_step_timpact(d: pd.DataFrame) -> np.ndarray:
    """Leave-one-out MRI t-statistic via exact one-step PRESS-style influence.

    For OLS with regressor matrix X and residuals e, hat h_ii, the effect of
    deleting row i on the coefficient b (and thus its t-statistic, using the
    full-sample HAC covariance) is derived as
        b_{-i} ~ b - (X'X)^{-1} x_i e_i / (1 - h_ii).
    This is exact for the OLS coefficients (leave-one-out OLS identity);
    the influence on the t-statistic is evaluated using the HAC-21 VCOV from
    the full sample. Iterating removals updates influence with the same
    identity and is O(n) per update.
    """
    dd = d[[Y] + CTRL].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    X = sm.add_constant(dd[CTRL], has_constant="add")
    m = sm.OLS(dd[Y], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    cols = ["const"] + CTRL
    iM = cols.index(M)
    Xn = X.to_numpy(float)
    yv = dd[Y].to_numpy(float)
    b = m.params.to_numpy()
    XtX_inv = np.linalg.inv(Xn.T @ Xn)
    h = np.einsum("ij,jk,ik->i", Xn, XtX_inv, Xn)
    e = yv - Xn @ b
    # b_{-i}
    b_lo = np.stack([
        b - XtX_inv @ (Xn[i] * e[i] / max(1 - h[i], 1e-9)) for i in range(len(dd))
    ])
    # recover per-i t-statistic from the HAC covariance of full sample
    cov = m.cov_params()  # HAC
    se = np.sqrt(np.diag(cov))[iM]
    t_lo = b_lo[:, iM] / se
    return dd, m, t_lo


def main() -> None:
    cn = pd.read_csv(os.path.join(DATA, "csi300_daily_panel.csv"))
    cn["Date"] = pd.to_datetime(cn["Date"])
    cn["year"] = cn["Date"].dt.year
    us = pd.read_csv(os.path.join(DATA, "sp500_daily_panel.csv"))
    us["Date"] = pd.to_datetime(us["Date"])
    us["year"] = us["Date"].dt.year

    emit("=" * 90)
    emit("ROBUSTNESS (DAILY): CSI 300, MRI -> next-trading-day excess return, "
         "HAC-21, controls: R, sigma, E/P, B/M")
    emit("=" * 90)
    base = run(cn)
    line("Baseline", base)
    b0, t0 = base.params[M], base.tvalues[M]

    # ---- R1 influence ----------------------------------------------------
    emit("\n[R1] Influence of individual observations (one-step PRESS identity)")
    dd, m_full, t_lo = one_step_timpact(cn)
    emit(f"  leave-one-out t range: [{t_lo.min():+.2f}, {t_lo.max():+.2f}]  (baseline {t0:+.2f})")
    RESULTS["leave-one-out t range"] = {"t_min": float(t_lo.min()), "t_max": float(t_lo.max())}
    # drop k most favourable (largest t_lo)
    order = np.argsort(-t_lo)
    for k in (5, 10, 20, 100):
        m = run(dd.drop(index=order[:k]))
        line(f"drop {k} most favourable days", m)
    # greedy sequential with the same identity update (PR approximation)
    cur = dd.copy()
    X = sm.add_constant(cur[CTRL], has_constant="add")
    for k in range(1, 17):
        mm = sm.OLS(cur[Y], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        Xn = X.to_numpy()
        Xi = np.linalg.inv(Xn.T @ Xn)
        h = np.einsum("ij,jk,ik->i", Xn, Xi, Xn)
        e = (cur[Y].to_numpy() - Xn @ mm.params.to_numpy())
        inf = np.abs(e / np.clip(1 - h, 1e-9, 1.0))
        w = inf.argmax()
        cur = cur.drop(index=cur.index[w]).reset_index(drop=True)
        X = sm.add_constant(cur[CTRL], has_constant="add")
        if k in (5, 10, 16):
            line(f"greedy removal of {k} most influential days", run(cur))

    # ---- R2 crisis exclusion --------------------------------------------
    emit("\n[R2] Excluding crisis episodes (calendar years, daily)")
    line("exclude 2008", run(cn[cn.year != 2008]))
    line("exclude 2015-2016", run(cn[~cn.year.isin([2015, 2016])]))
    line("exclude 2008, 2015, 2016", run(cn[~cn.year.isin([2008, 2015, 2016])]))
    line("exclude 2008, 2015, 2016, 2020, 2022",
         run(cn[~cn.year.isin([2008, 2015, 2016, 2020, 2022])]))
    line("exclude top-decile sigma_t days", run(cn[cn.sigma_t < cn.sigma_t.quantile(0.90)]))

    # ---- R3 winsorization -------------------------------------------------
    emit("\n[R3] Winsorizing the dependent variable")
    for q in (0.01, 0.05):
        c = cn.copy()
        lo, hi = c[Y].quantile(q), c[Y].quantile(1 - q)
        c[Y] = c[Y].clip(lo, hi)
        line(f"winsorized at {int(q*100)}%", run(c))

    # ---- R4 subsample stability ------------------------------------------
    emit("\n[R4] Subsample stability (daily)")
    half = len(cn) // 2
    d0 = pd.to_datetime(cn['Date']).sort_values().iloc[0]
    line("first half", run(cn.iloc[:half]))
    line("second half", run(cn.iloc[half:]))
    for lo, hi in [(2005, 2011), (2011, 2017), (2017, 2025)]:
        line(f"{lo}-{hi}", run(cn[(cn.year >= lo) & (cn.year < hi)]))

    emit(f"\n  recursive expanding-origin beta (every {REC_STEP//D} months -> {REC_STEP} days):")
    for n in range(REC_ORIGIN, len(cn) + 1, REC_STEP):
        mth = run(cn.iloc[:n])
        if mth is not None:
            emit(f"    through {str(cn['Date'].iloc[n-1].date())}: b={mth.params[M]:+.6f} t={mth.tvalues[M]:+.2f}")

    # ---- R5 placebo ---------------------------------------------------------
    emit(f"\n[R5] Placebo: circular block bootstrap of MRI (block={BLOCK//21}m -> {BLOCK} days, 2000 reps)")
    dd = cn[[Y] + CTRL].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mri = dd[M].to_numpy()
    n, bl = len(dd), min(BLOCK, len(dd) - 1)
    ts = []
    for _ in range(2000):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend([(s + j) % n for j in range(bl)])
        perm = dd.copy()
        perm[M] = mri[np.array(idx[:n])]
        m = run(perm)
        if m is not None:
            ts.append(m.tvalues[M])
    ts = np.array(ts)
    p = float((np.abs(ts) >= abs(t0)).mean())
    emit(f"  |t| 90th={np.percentile(np.abs(ts),90):.2f}  95th={np.percentile(np.abs(ts),95):.2f}  "
         f"99th={np.percentile(np.abs(ts),99):.2f}  observed={abs(t0):.2f}  p={p:.4f}")
    RESULTS["R5 block bootstrap"] = {
        "pct90": float(np.percentile(np.abs(ts), 90)),
        "pct95": float(np.percentile(np.abs(ts), 95)),
        "pct99": float(np.percentile(np.abs(ts), 99)),
        "observed_abs_t": float(abs(t0)),
        "p": float(p)}

    # ---- R6 alternative targets --------------------------------------------
    emit("\n[R6] Alternative dependent variables (daily)")
    # daily-only raw next-day return target
    c = cn.copy()
    c["RET_raw"] = c["Close"].pct_change().shift(-1)
    line("raw (not excess) next-day return", run(c, y="RET_raw"))
    c = cn.copy()
    c["Fwd2"] = c["Close"].pct_change().shift(-2)
    line("skip-one-day-ahead return", run(c, y="Fwd2"))
    c = cn.copy()
    c["Cum5"] = c["Close"].pct_change(5).shift(-1)  # next 5-day horizon
    line("next-5-trading-day return", run(c, y="Cum5"))

    # ---- R7 deliberate anchor delay -------------------------------------
    emit(f"\n[R7] Look-ahead audit: MRI delayed further")
    c = cn.copy()
    c[M] = c[M].shift(LAG_AUDIT)
    line(f"MRI lagged {LAG_AUDIT//21} months ({LAG_AUDIT} days)", run(c))
    c = cn.copy()
    c[M] = c[M].shift(LAG_ONE)
    line(f"MRI lagged {LAG_ONE} day", run(c))

    # ---- R8 is MRI just a drawdown dummy? --------------------------------
    emit("\n[R8] Does MRI merely proxy a contemporaneous drawdown?")
    c = cn.copy()
    c["dd_dummy"] = (c["R_t"] < c["R_t"].quantile(0.20)).astype(float)
    c["sig_dummy"] = (c["sigma_t"] > c["sigma_t"].quantile(0.80)).astype(float)
    line("add low-return and high-vol dummies", run(c, xs=CTRL + ["dd_dummy", "sig_dummy"]))
    line("add squared R and sigma", run(
        c.assign(R2=c.R_t ** 2, S2=c.sigma_t ** 2), xs=CTRL + ["R2", "S2"]))
    line("add R x sigma interaction", run(
        c.assign(RS=c.R_t * c.sigma_t), xs=CTRL + ["RS"]))

    # ---- US mirror --------------------------------------------------------
    emit("\n" + "=" * 90)
    emit("US MIRROR (same tests, S&P 500, daily)")
    emit("=" * 90)
    us_ctrl = BASE + ["DP_t", "CAPE_t", "VIX_t", "VRP_t", "Mkt_RF_t", "SMB_t", "HML_t"]
    line("US: baseline", run(us, xs=us_ctrl))
    line("US: exclude 2008-2009", run(us[~us.year.isin([2008, 2009])], xs=us_ctrl))
    line("US: exclude 2020", run(us[us.year != 2020], xs=us_ctrl))
    line("US: first half", run(us.iloc[:len(us)//2], xs=us_ctrl))
    line("US: second half", run(us.iloc[len(us)//2:], xs=us_ctrl))

    with open(os.path.join(RES, "robustness_daily.txt"), "w") as f:
        f.write("\n".join(LINES))
    emit(f"\nWritten to {os.path.join(RES,'robustness_daily.txt')}")

    jp = os.path.join(RES, "robustness_daily.json")
    with open(jp, "w") as f:
        json.dump({"seed": 20260807, "n_reps": 2000, "results": RESULTS}, f, indent=1)
    emit(f"Written to {jp}")


if __name__ == "__main__":
    main()