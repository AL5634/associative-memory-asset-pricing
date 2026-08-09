"""
02b_analysis_daily.py
=====================
Predictive regressions for the Memory Resonance Index at DAILY frequency.

This is the daily appendage of 02_analysis.py. Everything is identical except:
  * Observation age / frequency: daily trading observations.
  * Dependent variables:
        Future_ExRet(t)      = next-trading-day excess return = ret(t+1)-rf(t)
        Future_Volatility(t) = annualised realised vol of the next 21 trading days
  * Inference: Newey-West HAC with 21 lags (= the 21-day realised-volatility
    window size, hence the overlap length; the monthly design used 6 months,
    and a 21-day rolling window is the direct daily analogue).
  * All control variables, MRI bandwidth variants, and sample windows are the
    same as in the monthly analysis.

Outputs: results_daily/*.csv, results_daily/summary_daily.txt
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_real")
RES = os.path.join(HERE, "results_daily")
os.makedirs(RES, exist_ok=True)

BASELINE_MRI = "MRI_bw0.25"
HAC_LAGS = 21
LINES: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fit(df: pd.DataFrame, y: str, xs: list[str], hac: int = HAC_LAGS):
    cols = [y] + xs
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30:
        return None
    X = sm.add_constant(d[xs], has_constant="add")
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    m_hc1 = sm.OLS(d[y], X).fit(cov_type="HC1")
    m.n_obs = len(d)
    m.hc1 = m_hc1
    return m


def spec_table(df: pd.DataFrame, label: str, specs: list[tuple[str, str, list[str]]],
               fname: str) -> pd.DataFrame:
    """specs: list of (column label, dependent variable, regressor list)."""
    all_x: list[str] = []
    for _, _, xs in specs:
        for x in xs:
            if x not in all_x:
                all_x.append(x)

    rows: dict[str, list[str]] = {}
    order = ["const"] + all_x
    for nm in order:
        rows[nm] = []
        rows[nm + "__t"] = []
    meta = {"Adj. R2": [], "N": []}

    fitted = []
    for cl, y, xs in specs:
        m = fit(df, y, xs)
        fitted.append((cl, m))
        for nm in order:
            if m is not None and nm in m.params.index:
                rows[nm].append(f"{m.params[nm]:.6f}{stars(m.pvalues[nm])}")
                rows[nm + "__t"].append(f"({m.tvalues[nm]:.2f})")
            else:
                rows[nm].append("---")
                rows[nm + "__t"].append("")
        meta["Adj. R2"].append(f"{m.rsquared_adj*100:.3f}%" if m is not None else "---")
        meta["N"].append(str(m.n_obs) if m is not None else "---")
    if "const" not in order:
        order.insert(0, "const")

    idx, data = [], []
    pretty = {"const": "Constant", BASELINE_MRI: "MRI$_t$", "R_t": "$R_t$",
              "sigma_t": "$\\sigma_t$", "EP_t": "E/P$_t$", "BM_t": "B/M$_t$",
              "DP_t": "D/P$_t$", "CAPE_t": "CAPE$_t$", "VIX_t": "VIX$_t$",
              "IV_t": "QVIX$_t$", "VRP_t": "VRP$_t$", "Mkt_RF_t": "MKT$_t$",
              "SMB_t": "SMB$_t$", "HML_t": "HML$_t$"}
    for nm in order:
        idx.append(pretty.get(nm, nm)); data.append(rows[nm])
        idx.append(""); data.append(rows[nm + "__t"])
    for k, v in meta.items():
        idx.append(k); data.append(v)

    out = pd.DataFrame(data, index=idx, columns=[c for c, _, _ in specs])
    out.to_csv(os.path.join(RES, fname + ".csv"))

    emit(f"\n{'='*86}\n{label}\n{'='*86}")
    emit(out.to_string())

    emit("\n  MRI robustness across covariance estimators:")
    for cl, m in fitted:
        if m is None or BASELINE_MRI not in m.params.index:
            continue
        b = m.params[BASELINE_MRI]
        emit(f"    {cl:34s} b={b:+.6f}  t(HAC22)={m.tvalues[BASELINE_MRI]:+.3f}  "
             f"t(HC1)={m.hc1.tvalues[BASELINE_MRI]:+.3f}  N={m.n_obs}")
    return out


def oos_r2(df: pd.DataFrame, y: str, xs: list[str], min_train: int = 1260) -> tuple[float, int]:
    """
    Campbell-Thompson out-of-sample R^2 against the prevailing-mean benchmark.
    Expanding window, strictly recursive. min_train in DAYS: 60 months x 21
    trading days = 1260, the daily analogue of the monthly min_train=60.
    """
    d = df[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(d) < min_train + 21 * 24 * 4:
        return float("nan"), 0
    e_m, e_b = [], []
    for i in range(min_train, len(d)):
        tr, te = d.iloc[:i], d.iloc[i]
        X = sm.add_constant(tr[xs], has_constant="add")
        try:
            beta = sm.OLS(tr[y], X).fit().params
        except Exception:
            continue
        xrow = np.concatenate([[1.0], te[xs].to_numpy(float)])
        pred = float(np.dot(beta.to_numpy(), xrow))
        e_m.append((te[y] - pred) ** 2)
        e_b.append((te[y] - tr[y].mean()) ** 2)
    if not e_m:
        return float("nan"), 0
    return (1.0 - np.sum(e_m) / np.sum(e_b)) * 100.0, len(e_m)


def main() -> None:
    cn = pd.read_csv(os.path.join(DATA, "csi300_daily_panel.csv"))
    us = pd.read_csv(os.path.join(DATA, "sp500_daily_panel.csv"))
    for dfr in (cn, us):
        dfr["Date"] = pd.to_datetime(dfr["Date"])
        dfr["year"] = dfr["Date"].dt.year

    emit("=" * 86)
    emit("PREDICTIVE REGRESSIONS -- real data, DAILY, next-trading-day target")
    emit("=" * 86)
    emit(f"  CSI 300 : N = {len(cn)} trading days  {cn['Date'].iloc[0].date()} -> {cn['Date'].iloc[-1].date()}")
    emit(f"  S&P 500 : N = {len(us)} trading days  {us['Date'].iloc[0].date()} -> {us['Date'].iloc[-1].date()}")
    emit(f"  Target ExRet: next-day excess return; Target Vol: 21-trading-day RV (annualised)")
    emit(f"  Baseline MRI column: {BASELINE_MRI} (gamma = 0.25, standardized cue space)")
    emit(f"  Inference: Newey-West HAC, {HAC_LAGS} lags; HC1 reported as a check.")

    M = BASELINE_MRI
    base = [M, "R_t", "sigma_t"]

    # ---------------- China, full sample 2005-2025 ----------------
    cn_ctrl = base + ["EP_t", "BM_t"]
    spec_table(
        cn, "TABLE 1D. CSI 300, 2005-2025 daily (real E/P and B/M controls)",
        [("(1) Excess return", "Future_ExRet", base),
         ("(2) Volatility", "Future_Volatility", base),
         ("(3) Excess return (ctrl)", "Future_ExRet", cn_ctrl),
         ("(4) Volatility (ctrl)", "Future_Volatility", cn_ctrl)],
        "table1_csi300_daily")

    # ---------------- China, option-era subsample 2015-2025 ----------------
    cn_iv = cn[cn["IV_t"].notna()].copy()
    cn_ctrl_iv = cn_ctrl + ["IV_t", "VRP_t"]
    spec_table(
        cn_iv,
        f"CSI 300, 2015-2025 option era (adds QVIX/VRP; N={len(cn_iv)})",
        [("(1) Excess return", "Future_ExRet", base),
         ("(2) Volatility", "Future_Volatility", base),
         ("(3) Excess return (ctrl)", "Future_ExRet", cn_ctrl_iv),
         ("(4) Volatility (ctrl)", "Future_Volatility", cn_ctrl_iv)],
        "table2_csi300_optionera_daily")

    # ---------------- US, 1990-2023 ----------------
    us_ctrl = base + ["DP_t", "CAPE_t", "VIX_t", "VRP_t", "Mkt_RF_t", "SMB_t", "HML_t"]
    spec_table(
        us, "S&P 500, 1990-2023 daily (Shiller CAPE/DP, VIX, Fama-French)",
        [("(1) Excess return", "Future_ExRet", base),
         ("(2) Volatility", "Future_Volatility", base),
         ("(3) Excess return (ctrl)", "Future_ExRet", us_ctrl),
         ("(4) Volatility (ctrl)", "Future_Volatility", us_ctrl)],
        "table3_sp500_daily")

    # ---------------- Bandwidth sensitivity ----------------
    emit(f"\n{'='*86}\nDaily bandwidth sensitivity to MRI kernel gamma\n{'='*86}")
    rows = []
    for col, g in [("MRI_bw0.125", 0.125), ("MRI_bw0.25", 0.25), ("MRI_bw0.5", 0.5),
                   ("MRI_bw1.0", 1.0), ("MRI_t", 2.0), ("MRI_bw5.0", 5.0)]:
        r = {"gamma": g}
        for lab, df in [("CN", cn), ("US", us)]:
            for yv, yl in [("Future_ExRet", "ret"), ("Future_Volatility", "vol")]:
                xs = [col, "R_t", "sigma_t"] + (["EP_t", "BM_t"] if lab == "CN" else
                                                ["DP_t", "CAPE_t", "VIX_t", "VRP_t",
                                                 "Mkt_RF_t", "SMB_t", "HML_t"])
                m = fit(df, yv, xs)
                if m is not None and col in m.params.index:
                    r[f"{lab}_{yl}_b"] = round(float(m.params[col]), 6)
                    r[f"{lab}_{yl}_t"] = round(float(m.tvalues[col]), 3)
        rows.append(r)
    bw = pd.DataFrame(rows)
    bw.to_csv(os.path.join(RES, "tableA1_bandwidth_daily.csv"), index=False)
    emit(bw.to_string(index=False))

    # ---------------- Out-of-sample ----------------
    emit(f"\n{'='*86}\nOut-of-sample R2 (Campbell-Thompson, expanding, min_train=60m->1260d)\n{'='*86}")
    oos_rows = []
    for lab, df, ctrl in [("CSI 300", cn, cn_ctrl), ("S&P 500", us, us_ctrl)]:
        for yv, yl in [("Future_ExRet", "Excess return"), ("Future_Volatility", "Volatility")]:
            full, n1 = oos_r2(df, yv, ctrl)
            nomri, _ = oos_r2(df, yv, [c for c in ctrl if c != M])
            oos_rows.append({"Market": lab, "Target": yl,
                             "OOS R2 with MRI (%)": round(full, 3),
                             "OOS R2 without MRI (%)": round(nomri, 3),
                             "Gain (pp)": round(full - nomri, 3),
                             "N forecasts": n1})
    oos = pd.DataFrame(oos_rows)
    oos.to_csv(os.path.join(RES, "tableA2_oos_daily.csv"), index=False)
    emit(oos.to_string(index=False))

    # ---------------- Descriptives ----------------
    emit(f"\n{'='*86}\nDaily descriptive statistics\n{'='*86}")
    for lab, df, vs in [("CSI 300", cn, [M, "R_t", "sigma_t", "EP_t", "BM_t", "IV_t",
                                          "Future_ExRet", "Future_Volatility"]),
                        ("S&P 500", us, [M, "R_t", "sigma_t", "DP_t", "CAPE_t", "VIX_t",
                                          "Future_ExRet", "Future_Volatility"])]:
        de = df[vs].describe().T[["count", "mean", "std", "min", "50%", "max"]]
        de.to_csv(os.path.join(RES, f"tableA3_desc_{lab.replace(' ','').replace('&','')}_daily.csv"))
        emit(f"\n{lab}")
        emit(de.round(6).to_string())

    with open(os.path.join(RES, "summary_daily.txt"), "w") as f:
        f.write("\n".join(LINES))
    emit(f"\nWritten to {RES}")


if __name__ == "__main__":
    main()