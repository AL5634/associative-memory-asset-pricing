"""
08b_frequency_comparison.py
==========================
Monthly vs. Daily comparison report, generated from the committed
results/ and results_daily/ outputs of both pipelines.

Writes results_daily/FREQUENCY_COMPARISON.md
"""

from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RESD = os.path.join(HERE, "results_daily")


def read(path):
    with open(path) as f:
        return f.read()


def m_row(raw, col):
    for i, v in enumerate(raw.index):
        if "MRI" in str(v):
            return str(raw.iloc[i][col]), str(raw.iloc[i + 1][col])
    return "--", "--"


def adjN(raw, col):
    ik = [i for i, v in enumerate(raw.index) if "Adj" in str(v)][0]
    iN = [i for i, v in enumerate(raw.index) if str(v).strip() == "N"][0]
    return str(raw.iloc[ik][col]), str(raw.iloc[iN][col])


def grab(txt, pat):
    for line in txt.splitlines():
        if pat in line:
            return line.strip()
    return "--"


def main() -> None:
    mon_cn = pd.read_csv(os.path.join(RES, "table1_csi300_full.csv"), index_col=0)
    daily_cn = pd.read_csv(os.path.join(RESD, "table1_csi300_daily.csv"), index_col=0)
    mon_us = pd.read_csv(os.path.join(RES, "table3_sp500.csv"), index_col=0)
    daily_us = pd.read_csv(os.path.join(RESD, "table3_sp500_daily.csv"), index_col=0)
    mon_id = read(os.path.join(RES, "identification_summary.txt"))
    daily_id = read(os.path.join(RESD, "identification_summary_daily.txt"))
    mon_rob = read(os.path.join(RES, "robustness.txt"))
    daily_rob = read(os.path.join(RESD, "robustness_daily.txt"))

    C1, C3 = "(1) Excess return", "(3) Excess return (ctrl)"

    L = []
    L.append("# Monthly vs. Daily Frequency Comparison")
    L.append("")
    L.append("Comparison report generated from the committed results of both pipelines "
             "in this deposit. Monthly numbers come from the original "
             "monthly (v2) pipeline; daily numbers from this daily-frequency "
             "extension.")
    L.append("")
    L.append("## 1. CSI 300 — MRI predictive coefficient")
    L.append("")
    L.append("| Spec | Monthly (HAC-6) | Daily (HAC-21) |")
    L.append("|---|---|---|")
    b1, t1 = m_row(mon_cn, C1); db1, dt1 = m_row(daily_cn, C1)
    b3, t3 = m_row(mon_cn, C3); db3, dt3 = m_row(daily_cn, C3)
    a2m, Nm = adjN(mon_cn, C3); a2d, Nd = adjN(daily_cn, C3)
    L.append(f"| b, col (1) | {b1} | {db1} |")
    L.append(f"| t, col (1) | {t1} | {dt1} |")
    L.append(f"| b, col (3) full ctrl | {b3} | {db3} |")
    L.append(f"| t, col (3) full ctrl | {t3} | {dt3} |")
    L.append(f"| Adj. R2, col (3) | {a2m} | {a2d} |")
    L.append(f"| N, col (3) | {Nm} | {Nd} |")
    L.append("")
    L.append("Same scale note: monthly b is per-month excess return; daily b is "
             "per-trading-day return. Coefficient interpreted: 1-sd shift in "
             "MRI corresponds to monthly b*sd(MRI) pp / month vs daily "
             "b*sd(MRI) pp per day.")
    L.append("")
    L.append("## 2. S&P 500 — MRI coefficient, full control (3)")
    L.append("")
    L.append("| | Monthly | Daily |")
    L.append("|---|---|---|")
    bmu, tmu = m_row(mon_us, C3); bdu, tdu = m_row(daily_us, C3)
    L.append(f"| b | {bmu} | {bdu} |")
    L.append(f"| t | {tmu} | {tdu} |")
    L.append("")
    L.append("## 3. Identification (CSI 300)")
    L.append("")
    for key in ["A2:", "A3:", "A4:"]:
        L.append(f"| {key} monthly | {grab(mon_id, key)} |")
        L.append(f"| {key} daily   | {grab(daily_id, key)} |")
    L.append(f"| Test B placebo (m) | {grab(mon_id, 'Placebo t mean')} |")
    L.append(f"| Test B placebo (d) | {grab(daily_id, 'Placebo t mean')} |")
    L.append(f"| Test B p-value (m) | {[x for x in mon_id.splitlines() if 'p-value' in x][0] if any('p-value' in x for x in mon_id.splitlines()) else grab(mon_id, 'p-value')} |")
    L.append(f"| Test B p-value (d) | {[x for x in daily_id.splitlines() if 'p-value' in x][0] if any('p-value' in x for x in daily_id.splitlines()) else grab(daily_id, 'p-value')} |")
    L.append(f"| Test C resid (m) | {grab(mon_id, 'Residual MRI alone')} |")
    L.append(f"| Test C resid (d) | {grab(daily_id, 'Residual MRI alone')} |")
    L.append("")
    L.append("## 4. Robustness baseline")
    L.append("")
    L.append(f"| Monthly | `{grab(mon_rob,'Baseline')}` |")
    L.append(f"| Daily   | `{grab(daily_rob,'Baseline')}` |")
    L.append("")
    L.append("## 5. Out-of-sample R2 with MRI (CSI 300 excess return)")
    L.append("")
    L.append("See tableA2_oos.csv (monthly) vs tableA2_oos_daily.csv (daily).")

    with open(os.path.join(RESD, "FREQUENCY_COMPARISON.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWritten {os.path.join(RESD, 'FREQUENCY_COMPARISON.md')}")


if __name__ == "__main__":
    main()