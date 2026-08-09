"""
01b_build_data_daily.py
=======================
Builds the DAILY-frequency regression panel from the committed data_real/
snapshots (no network calls).

Design (mirrors the monthly design decision-for-decision, but at daily
granularity so the two can be compared):

  * Predictor day t: everything that is known AT the close of trading day t
    (MRI, R_t, sigma_t, valuation ratios, VIX/QVIX, VRP).
  * Future_ExRet(t)     = ret(t+1) - rf_t        (next trading day, excess).
  * Future_Volatility(t)= annualised realised volatility of the NEXT 21
    TRADING DAYS (t+1..t+21), sqrt(252) scaling, at least 15 valid days in
    the window (the daily analogue of the monthly >=15-day rule).
  * R_t  = 21-trading-day trailing simple return (drawn straight from the
           daily CSV, already built by 01_build_data.py).
  * sigma_t = annualised 21-day realised volatility (same).
  * MRI_* columns are the strictly backward-looking index from 01_build_data.py,
    untouched: the memory index machinery is identical to the monthly design.
  * China control VRP_t = IV_t/100 - sigma_t. US VRP_t = VIX_t/100 - sigma_t.
  * CAPE_t / DP_t (Shiller, monthly) are carried to each trading day using the
    PREVIOUS month's value (backward-looking: a month's CAPE enters the daily
    series only after that month has ended), then forward-filled.
  * Sample windows identical to the monthly design:
        CSI 300 : 2005-01-01 .. 2025-12-31
        S&P 500 : 1990-01-01 .. 2023-12-31

Outputs (written to ./data_real/)
  csi300_daily_panel.csv
  sp500_daily_panel.csv
  data_manifest_daily.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_real")
OUT = os.path.join(HERE, "data_real")
os.makedirs(OUT, exist_ok=True)

TRADING_DAYS = 252
VOL_WINDOW = 21          # next 21 trading days realised volatility
MIN_OBS = 15             # intra-window opens threshold (monthly used 15/21)
CN_SAMPLE = ["2005-01-01", "2025-12-31"]
US_SAMPLE = ["1990-01-01", "2023-12-31"]


def fmt_date(pd_dt):
    return pd.Timestamp(pd_dt).strftime("%Y-%m-%d")


def build_cn():
    d = pd.read_csv(os.path.join(DATA, "csi300_daily.csv"), parse_dates=["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    ret = d["ret_d"].to_numpy(float)
    n = len(d)
    fwd_vol = np.full(n, np.nan)
    for i in range(n):
        w = ret[i + 1: i + 1 + VOL_WINDOW]
        w = w[np.isfinite(w)]
        if len(w) >= MIN_OBS:
            fwd_vol[i] = np.std(w) * np.sqrt(TRADING_DAYS)

    d["Future_Volatility"] = fwd_vol
    d["rf_d"] = d["rf_annual_pct"] / 100.0 / TRADING_DAYS
    d["Future_ExRet"] = d["ret_d"].shift(-1) - d["rf_d"]
    d["VRP_t"] = (d["IV_t"] / 100.0) - d["sigma_t"]

    panel = d[d["Date"] >= CN_SAMPLE[0]].copy()
    panel = panel[panel["Date"] <= CN_SAMPLE[1]].reset_index(drop=True)
    return panel


def ffill_prev_month(daily: pd.DataFrame, monthly: pd.DataFrame,
                     col: str) -> np.ndarray:
    """Backward-looking monthly->daily mapping (previous month's value)."""
    mm = monthly[["Month", col]].dropna().copy()
    mm["p"] = pd.PeriodIndex(mm["Month"], freq="M").shift(-1).astype(str)
    mm = mm.dropna(subset=["p"])
    day = daily.copy()
    day["p"] = day["Date"].dt.to_period("M").astype(str)
    m = day[["p"]].merge(mm[["p", col]], on="p", how="left")
    return m[col].ffill().to_numpy()


def build_us():
    d = pd.read_csv(os.path.join(DATA, "sp500_daily.csv"), parse_dates=["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    mon = pd.read_csv(os.path.join(DATA, "sp500_monthly.csv"),
                      parse_dates=["Date"])
    mon["date"] = mon["Month"]

    d["CAPE_t"] = ffill_prev_month(d, mon, "CAPE_t")
    d["DP_t"] = ffill_prev_month(d, mon, "DP_t")

    ret = d["ret_d"].to_numpy(float)
    n = len(d)
    fwd_vol = np.full(n, np.nan)
    for i in range(n):
        w = ret[i + 1: i + 1 + VOL_WINDOW]
        w = w[np.isfinite(w)]
        if len(w) >= MIN_OBS:
            fwd_vol[i] = np.std(w) * np.sqrt(TRADING_DAYS)

    d["Future_Volatility"] = fwd_vol
    d["rf_d"] = d["RF_t"].to_numpy(float)
    d["Future_ExRet"] = d["ret_d"].shift(-1) - d["rf_d"]
    d["VRP_t"] = (d["VIX_t"] / 100.0) - d["sigma_t"]

    panel = d[d["Date"] >= US_SAMPLE[0]].copy()
    panel = panel[panel["Date"] <= US_SAMPLE[1]].reset_index(drop=True)
    return panel


def main() -> None:
    print("=" * 78)
    print("Building DAILY panels from the existing monthly/daily snapshots.")
    print("=" * 78)

    cn = build_cn()
    us = build_us()

    keep_cn = ["Date", "Close", "ret_d", "R_t", "sigma_t",
               "MRI_t", "MRI_bw0.125", "MRI_bw0.25", "MRI_bw0.5",
               "MRI_bw1.0", "MRI_bw5.0", "EP_t", "BM_t", "IV_t",
               "rf_d", "VRP_t", "Future_ExRet", "Future_Volatility"]
    keep_us = ["Date", "Close", "VIX_t", "ret_d", "R_t", "sigma_t",
               "MRI_t", "MRI_bw0.125", "MRI_bw0.25", "MRI_bw0.5",
               "MRI_bw1.0", "MRI_bw5.0", "Mkt_RF_t", "SMB_t", "HML_t",
               "RF_t", "DP_t", "CAPE_t", "VRP_t", "Future_ExRet",
               "Future_Volatility"]
    cn_out = cn[keep_cn].copy()
    us_out = us[keep_us].copy()

    cn_out = cn_out[cn_out["Date"] >= CN_SAMPLE[0]]
    cn_out = cn_out[cn_out["Date"] <= CN_SAMPLE[1]].reset_index(drop=True)
    us_out = us_out[us_out["Date"] >= US_SAMPLE[0]]
    us_out = us_out[us_out["Date"] <= US_SAMPLE[1]].reset_index(drop=True)

    cn_out.to_csv(os.path.join(OUT, "csi300_daily_panel.csv"), index=False)
    us_out.to_csv(os.path.join(OUT, "sp500_daily_panel.csv"), index=False)

    report = {
        "csi300_daily_panel": {
            "rows": int(len(cn_out)),
            "range": [fmt_date(cn_out["Date"].iloc[0]),
                      fmt_date(cn_out["Date"].iloc[-1])],
            "Future_ExRet_full_obs": int(cn_out["Future_ExRet"].notna().sum()),
            "Future_Volatility_full_obs": int(cn_out["Future_Volatility"].notna().sum()),
        },
        "sp500_daily_panel": {
            "rows": int(len(us_out)),
            "range": [fmt_date(us_out["Date"].iloc[0]),
                      fmt_date(us_out["Date"].iloc[-1])],
            "Future_ExRet_full_obs": int(us_out["Future_ExRet"].notna().sum()),
            "Future_Volatility_full_obs": int(us_out["Future_Volatility"].notna().sum()),
        },
        "design": {
            "Future_ExRet": "next-TRADING-DAY excess return = ret(t+1) - rf(t)",
            "Future_Volatility": "annualised realised vol of next 21 trading days",
            "Future_Volatility_window_days": VOL_WINDOW,
            "overlap": "next-day return: non-overlapping; vol target: overlapping 21-day",
            "HAC_recommendation_days": 21,
            "note": "MRI/R_t/sigma_t identical machinery and windows as the "
                    "monthly panel; only the observation frequency differs.",
        },
    }
    with open(os.path.join(OUT, "daily_manifest.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nSUMMARY (daily panels)")
    print("=" * 78)
    for lab, p in [("CSI 300", cn_out), ("S&P 500", us_out)]:
        print(f"  {lab}: N={len(p)}  {fmt_date(p['Date'].iloc[0])} -> "
              f"{fmt_date(p['Date'].iloc[-1])}")
        s = p["Future_ExRet"].describe()
        print(f"        MRI sd={p['MRI_bw0.25'].std():.4f} "
              f"Future_ExRet mean={s['mean']*100:.2f}%/day sd={s['std']*100:.2f}%/day")
    print(f"  Written to {OUT}")


if __name__ == "__main__":
    main()