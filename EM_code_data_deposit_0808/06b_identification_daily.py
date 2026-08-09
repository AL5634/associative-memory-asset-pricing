"""
06b_identification_daily.py
===========================
Daily-frequency version of the identification battery.

Test A — Flexible-control specification (deciles, splines, interaction).
Test B — Random-anchor placebo: MRI rebuilt from random (R, sigma) anchors.
Test C — Orthogonalised MRI.

Same logic as 06_identification.py; only the estimation sample is the DAILY
panel and HAC uses 21 lags (the 21-day vol-window overlap length). Test B is
vectorised over the daily grid. Seed identical (20260807) so the placebo
sequence is deterministic.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_real")
RES = os.path.join(HERE, "results_daily")
os.makedirs(RES, exist_ok=True)

M = "MRI_bw0.25"
HAC_LAGS = 21
rng = np.random.default_rng(20260807)

LINES: list[str] = []


def emit(s: str = ""):
    print(s)
    LINES.append(s)


def fit(df: pd.DataFrame, y: str, xs: list[str], coef: str | None = None) -> dict | None:
    d = df[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 40:
        return None
    X = sm.add_constant(d[xs], has_constant="add")
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    c = coef or M
    ok = c in m.params.index
    return {
        "b": float(m.params[c]) if ok else None,
        "t": float(m.tvalues[c]) if ok else None,
        "p": float(m.pvalues[c]) if ok else None,
        "adj_r2": float(m.rsquared_adj),
        "n": len(d),
        "model": m,
    }


def decile_dummies(s: pd.Series) -> pd.DataFrame:
    n = len(s.dropna())
    k = min(10, n)
    try:
        q = pd.qcut(s.rank(method="first"), k, labels=False, duplicates="drop")
        q = pd.Series(q, index=s.index, name="q").astype(int)
        r = pd.get_dummies(q, drop_first=True, dtype=float, prefix="D")
        r.columns = [f"D_{int(c.split('_')[1])}" if "_" in str(c) else f"D_{c}"
                     for c in r.columns]
        return r
    except Exception:
        return pd.DataFrame(index=s.index)


def natural_spline(x: pd.Series, df: int = 3) -> pd.DataFrame:
    q = np.linspace(0.05, 0.95, df - 1)
    knots = x.quantile(q).to_numpy()
    basis = x.to_numpy().copy()
    for k in knots:
        basis = np.column_stack([basis, np.maximum(x - k, 0) ** 3 -
                                 np.maximum(x - knots[-1], 0) ** 3 *
                                 (knots[-1] - k) / (knots[-1] - knots[0])])
    return pd.DataFrame(basis[:, 1:], index=x.index,
                        columns=[f"SP_{i}" for i in range(basis.shape[1] - 1)])


# ─────────────────────────────────────────────────────────────────────────────
def test_a(df_cn, df_us):
    emit("=" * 86)
    emit("TEST A (DAILY) — Flexible-control specification")
    emit("=" * 86)
    emit("Adding decile dummies of R_t and sigma_t, then splines, to the")
    emit("full-control daily return regression.")
    emit()

    base_xs = [M, "R_t", "sigma_t", "EP_t", "BM_t"]
    d = df_cn[["Future_ExRet"] + base_xs].dropna().copy()

    for v in ("R_t", "sigma_t"):
        dd = decile_dummies(df_cn[v].loc[d.index])
        dd_cols = list(dd.columns)
        d2 = df_cn.loc[d.index].join(dd)
        f = fit(d2, "Future_ExRet", base_xs + dd_cols)
        if f:
            emit(f"  A1: + {v} decile dummies  MRI b={f['b']:+.6f}  t={f['t']:+.2f}  "
                  f"p={f['p']:.4f}  adjR2={f['adj_r2']*100:.3f}%")

    dd_r = decile_dummies(df_cn["R_t"].loc[d.index]).add_prefix("R_")
    dd_s = decile_dummies(df_cn["sigma_t"].loc[d.index]).add_prefix("S_")
    dd_cols = list(dd_r.columns) + list(dd_s.columns)
    d2 = df_cn.loc[d.index].join(dd_r).join(dd_s)
    f = fit(d2, "Future_ExRet", base_xs + dd_cols)
    if f:
        emit(f"  A2: + both decile dummies   MRI b={f['b']:+.6f}  t={f['t']:+.2f}  "
              f"p={f['p']:.4f}  adjR2={f['adj_r2']*100:.3f}%")

    sp_r = natural_spline(df_cn["R_t"].loc[d.index], df=4).add_prefix("Rsp_")
    sp_s = natural_spline(df_cn["sigma_t"].loc[d.index], df=4).add_prefix("Ssp_")
    sp_cols = list(sp_r.columns) + list(sp_s.columns)
    d3 = df_cn.loc[d.index].join(sp_r).join(sp_s)
    f = fit(d3, "Future_ExRet", base_xs + sp_cols)
    if f:
        emit(f"  A3: + R + sigma splines      MRI b={f['b']:+.6f}  t={f['t']:+.2f}  "
              f"p={f['p']:.4f}  adjR2={f['adj_r2']*100:.3f}%")

    d4 = d3.copy()
    d4["MRIxR"] = d4[M] * d4["R_t"]
    d4["MRIxS"] = d4[M] * d4["sigma_t"]
    xs4 = base_xs + sp_cols + ["MRIxR", "MRIxS"]
    f = fit(d4, "Future_ExRet", xs4)
    if f:
        emit(f"  A4: + splines + MRIxR/S     MRI b={f['b']:+.6f}  t={f['t']:+.2f}  "
              f"p={f['p']:.4f}  adjR2={f['adj_r2']*100:.3f}%")
    emit()

    us_xs = [M, "R_t", "sigma_t", "DP_t", "CAPE_t", "VIX_t", "VRP_t",
             "Mkt_RF_t", "SMB_t", "HML_t"]
    dd_r_us = decile_dummies(df_us["R_t"]).add_prefix("R_")
    dd_s_us = decile_dummies(df_us["sigma_t"]).add_prefix("S_")
    dd_cols_us = list(dd_r_us.columns) + list(dd_s_us.columns)
    d_us = df_us.join(dd_r_us).join(dd_s_us)
    f = fit(d_us, "Future_ExRet", us_xs + dd_cols_us)
    if f:
        emit(f"  US A2: + both decile dummies  MRI b={f['b']:+.6f}  t={f['t']:+.2f}")
    return d


def _degrade(d: pd.Series, m: str) -> pd.Series:
    return d[m]


# ─────────────────────────────────────────────────────────────────────────────
def test_b(df_cn):
    """Random-anchor placebo on the DAILY panel (vectorised)."""
    emit("=" * 86)
    emit("TEST B (DAILY) — Random-anchor placebo")
    emit("=" * 86)
    emit("Replace each real crash anchor with a random (R, sigma) coordinate")
    emit("drawn from the empirical distribution of the cue vector, rebuild MRI,")
    emit("re-estimate the full daily return regression. 500 replications.")
    emit()

    cn = df_cn.copy()
    cn_d = cn["Date"].to_numpy()

    # Cue features are already in the daily panel (identical window as live MRI)
    ret = cn["R_t"]
    sig = cn["sigma_t"]

    valid = pd.DataFrame({"R": ret, "sigma": sig}).dropna()
    standardizer_n = len(valid)
    emp_dist = valid.to_numpy()

    sR = ret.expanding(min_periods=252).std()
    mR = ret.expanding(min_periods=252).mean()
    sS = sig.expanding(min_periods=252).std()
    mS = sig.expanding(min_periods=252).mean()

    n_anchors = 23  # matches CSV-committed 23 real CSI anchors
    emit(f"  {n_anchors} real anchors, {standardizer_n} candidate cue points in the daily panel")
    emit(f"  Daily sample N = {len(df_cn)}")

    base_cols = [M, "R_t", "sigma_t", "EP_t", "BM_t"]
    fake_base = ["MRI_FAKE", "R_t", "sigma_t", "EP_t", "BM_t"]

    true_f = fit(df_cn, "Future_ExRet", base_cols)
    true_t = true_f["t"] if true_f else None
    emit(f"  True MRI t-statistic: {true_t:+.2f}")
    emit()

    # vectorised fake MRI builder
    Rv = ret.to_numpy()
    Sv = sig.to_numpy()
    mRv, mSv = mR.to_numpy(), mS.to_numpy()
    s_Rv, s_Sv = sS.to_numpy(), sR.to_numpy()

    def build_fake(fake_anchors: np.ndarray) -> np.ndarray:
        ar = fake_anchors[:, 0][None, :]
        av = fake_anchors[:, 1][None, :]
        dz_r = (Rv[:, None] - ar) / s_Rv[:, None]
        dz_s = (Sv[:, None] - av) / s_Sv[:, None]
        kk = np.exp(-0.25 * (dz_r ** 2 + dz_s ** 2))
        out = kk.max(axis=1)
        bad = ~(np.isfinite(Rv) & np.isfinite(Sv) & (s_Rv > 1e-6) & (s_Sv > 1e-6))
        out[bad] = 0.0
        return out

    ts_placebo = []
    for rep in range(500):
        if rep > 0 and rep % 100 == 0:
            emit(f"  replication {rep}/500")
        idx = rng.integers(0, len(emp_dist), n_anchors)
        fake_anchors = emp_dist[idx]
        fake_mri = build_fake(fake_anchors)
        cn = df_cn.copy()
        cn["MRI_FAKE"] = np.where(np.isfinite(fake_mri), fake_mri, 0.0)
        f = fit(cn, "Future_ExRet", fake_base, coef="MRI_FAKE")
        if f is not None:
            ts_placebo.append(f["t"])

    ts_placebo = np.array(ts_placebo)
    ts_placebo = ts_placebo[np.isfinite(ts_placebo)]
    p = float((np.abs(ts_placebo) >= abs(true_t)).mean())
    emit(f"\n  500 replications complete.")
    if len(ts_placebo):
        emit(f"  Placebo t mean = {ts_placebo.mean():+.3f}   SD = {ts_placebo.std():.3f}")
        emit("  Placebo |t| distribution:")
        emit(f"    90th pctile = {np.percentile(np.abs(ts_placebo), 90):.2f}")
        emit(f"    95th pctile = {np.percentile(np.abs(ts_placebo), 95):.2f}")
        emit(f"    99th pctile = {np.percentile(np.abs(ts_placebo), 99):.2f}")
    emit(f"  True |t| = {abs(true_t):.2f}")
    if p == 0:
        emit(f"  Empirical p-value = <1/{len(ts_placebo)}")
    else:
        tail = "  *** SIGNIFICANT ***" if p < 0.01 else "  *** NOT SIGNIFICANT ***"
        emit(f"  Empirical p-value = {p:.4f}{tail}")
    return ts_placebo


# ─────────────────────────────────────────────────────────────────────────────
def test_c(df_cn, df_us):
    emit("=" * 86)
    emit("TEST C (DAILY) — Orthogonalised MRI")
    emit("=" * 86)
    emit("Step 1: regress MRI on R_t, sigma_t, splines, interaction.")
    emit("Step 2: test whether the residual predicts Future_ExRet.")
    emit()

    cols = ["Future_ExRet", M, "R_t", "sigma_t", "EP_t", "BM_t", "IV_t", "VRP_t"]
    d = df_cn[cols].dropna().copy()
    sp_r = natural_spline(d["R_t"], df=4)
    sp_s = natural_spline(d["sigma_t"], df=4)
    d = d.join(sp_r.add_prefix("Rsp_")).join(sp_s.add_prefix("Ssp_"))
    d["R2"] = d["R_t"] ** 2
    d["S2"] = d["sigma_t"] ** 2
    d["RS"] = d["R_t"] * d["sigma_t"]

    xs = (["R_t", "sigma_t", "R2", "S2", "RS"]
          + [f"Rsp_{c}" for c in sp_r.columns]
          + [f"Ssp_{c}" for c in sp_s.columns])
    X = sm.add_constant(d[xs], has_constant="add")
    res_mri = sm.OLS(d[M], X).fit()
    d["MRI_resid"] = res_mri.resid

    emit(f"  R²  MRI on flexible R, sigma: {res_mri.rsquared:.4f}")
    emit(f"  sd(MRI) = {d[M].std():.4f}  sd(residual) = {d['MRI_resid'].std():.4f}")
    emit(f"  fraction of MRI variance orthogonal to (R,σ): "
         f"{d['MRI_resid'].var() / d[M].var():.2%}")
    emit()

    for xs_v in [["MRI_resid", "R_t", "sigma_t", "EP_t", "BM_t"],
                 ["MRI_resid", "R_t", "sigma_t", "EP_t", "BM_t", "IV_t", "VRP_t"]]:
        f = fit(d, "Future_ExRet", xs_v, coef="MRI_resid")
        if f:
            emit(f"  Residual MRI alone:  b={f['b']:+.6f}  t={f['t']:+.2f}  p={f['p']:.4f}  "
                  f"adjR2={f['adj_r2']*100:.3f}%")
            if f["t"] is not None and abs(f["t"]) >= 1.96:
                emit("  -> SURVIVES. The MRI residual has independent predictive content.")
            else:
                emit("  -> FAILS. MRI adds nothing beyond flexible (R, σ) control.")

    emit()
    # NOTE: the v2 monthly pipeline's test_c "US mirror" erroneously used the
    # CSI 300 panel (pre-existing bug). The daily variant uses true S&P 500.
    d2 = df_us[["Future_ExRet", M, "R_t", "sigma_t"]].dropna().copy()
    sp_r2 = natural_spline(d2["R_t"], df=4)
    sp_s2 = natural_spline(d2["sigma_t"], df=4)
    d2 = d2.join(sp_r2.add_prefix("Rsp_")).join(sp_s2.add_prefix("Ssp_"))
    d2["R2"] = d2["R_t"] ** 2
    d2["S2"] = d2["sigma_t"] ** 2
    d2["RS"] = d2["R_t"] * d2["sigma_t"]
    xs2 = (["R_t", "sigma_t", "R2", "S2", "RS"]
           + [f"Rsp_{c}" for c in sp_r2.columns]
           + [f"Ssp_{c}" for c in sp_s2.columns])
    X2 = sm.add_constant(d2[xs2], has_constant="add")
    res2 = sm.OLS(d2[M], X2).fit()
    d2["MRI_resid"] = res2.resid
    f = fit(d2, "Future_ExRet", ["MRI_resid", "R_t", "sigma_t"], coef="MRI_resid")
    if f:
        emit(f"  US residual MRI:  b={f['b']:+.6f}  t={f['t']:+.2f}  p={f['p']:.4f}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cn = pd.read_csv(os.path.join(DATA, "csi300_daily_panel.csv"))
    us = pd.read_csv(os.path.join(DATA, "sp500_daily_panel.csv"))
    for _ in (cn, us):
        pass

    _ = test_a(cn, us)
    emit()
    ts = test_b(cn)
    emit()
    _ = test_c(cn, us)

    with open(os.path.join(RES, "identification_summary_daily.txt"), "w") as f:
        f.write("\n".join(LINES))
    print(f"\nWritten to {os.path.join(RES,'identification_summary_daily.txt')}")