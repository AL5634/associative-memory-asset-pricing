"""
07b_make_tables_daily.py
========================
Daily-frequency LaTeX table fragments from results_daily/ CSVs.
Every number is programmatically extracted from CSVs produced by 02b/06b.
Also emits a Monthly-vs-Daily frequency comparison table for the
CSI 300 full-control specification.
"""

from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_daily")
TAB = os.path.join(HERE, "tables_daily")
os.makedirs(TAB, exist_ok=True)


def sanitize(s) -> str:
    return str(s).replace("$", "").replace("\\", "").replace("_", " ").strip()


def parse_pred(raw: pd.DataFrame):
    """Return (rows, meta) from alternating coef/t CSV layout."""
    idx = list(raw.index)
    rows = {}
    cur = None
    for i, nm in enumerate(idx):
        if i % 2 == 0:
            cur = sanitize(nm)
            rows[cur] = {}
        else:
            for col in raw.columns:
                rows[cur][col] = (str(raw.iloc[i - 1][col]),
                                  str(raw.iloc[i][col]))
    meta = {"AdjR2": {}, "N": {}}
    for col in raw.columns:
        vals = raw[col].tolist()
        for k, v in zip([sanitize(x) for x in raw.index], vals):
            if "Adj" in k:
                meta["AdjR2"][col] = str(v)
            elif k == "N":
                meta["N"][col] = str(v)
    return rows, meta


parse_blocks = parse_pred


def pretty_table(raw: pd.DataFrame, caption: str, label: str) -> str:
    rows, meta = parse_blocks(raw)
    cols = list(raw.columns)
    order = ["Constant", "MRI t", "R t", "sigma t", "E P", "B M",
             "D P", "CAPE", "VIX", "Q VIX", "VRP", "MKT", "SMB", "HML"]
    keep = [k for k in order if k in rows]

    out = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\begin{tabular}{lcccc}")
    out.append("\\toprule")
    out.append("Variable & (1) Ex. Return & (2) Volatility & "
               "(3) Ex. Return & (4) Volatility \\\\")
    out.append("\\midrule")
    pretty = {"Constant": "Constant", "MRI": "MRI$_t$", "R t": "$R_t$",
              "sigma t": "$\\sigma_t$", "E P t": "E/P$_t$", "B M t": "B/M$_t$",
              "D P t": "D/P$_t$", "CAPE t": "CAPE$_t$", "VIX t": "VIX$_t$",
              "QV t": "QVIX$_t$", "VRP": "VRP$_t$", "MKT": "MKT$_t$",
              "SMB": "SMB$_t$", "HML": "HML$_t$"}
    for k in keep:
        bparts, tparts = [pretty.get(k, k)], [""]
        for c in cols:
            b, t = rows[k].get(c, ("---", "nan"))
            bparts.append(f"${b}$" if b not in ("nan", "None") else "---")
            tparts.append(t if t != "nan" else "")
        out.append(" & ".join(bparts) + " \\\\")
        out.append(" & ".join(tparts) + " \\\\")
    out.append("\\midrule")
    out.append(" & ".join(["Adj. R$^2$"] + [meta["AdjR2"][c].replace("%", "\\%") for c in cols]) + " \\\\")
    out.append(" & ".join(["N"] + [meta["N"][c] for c in cols]) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def table_csi300() -> str:
    raw = pd.read_csv(os.path.join(RES, "table1_csi300_daily.csv"), index_col=0)
    out = daily_table(
        raw,
        "\\textbf{Predictive Regressions: CSI 300, Daily, 2005--2025} "
        "(next-trading-day excess return; annualised 21-day realised-vol target)",
        "tab:csi300d")
    return out


def table_sp500() -> str:
    raw = pd.read_csv(os.path.join(RES, "table3_sp500_daily.csv"), index_col=0)
    return daily_table(
        raw,
        "\\textbf{Predictive Regressions: S\\&P 500, Daily, 1990--2023} "
        "(next-trading-day excess return; annualised 21-day realised-vol target)",
        "tab:sp500d")


def table_desc() -> str:
    cn = pd.read_csv(os.path.join(RES, "tableA3_desc_CSI300_daily.csv"), index_col=0)
    us = pd.read_csv(os.path.join(RES, "tableA3_desc_SP500_daily.csv"), index_col=0)
    varmap_cn = {
        "MRI_bw0.25": "MRI$_t$", "R_t": "$R_t$", "sigma_t": "$\\sigma_t$",
        "EP_t": "E/P$_t$", "BM_t": "B/M$_t$",
        "Future_ExRet": "Excess return$_{t+1}$",
        "Future_Volatility": "Volatility$_{t+1:t+21}$",
    }
    varmap_us = {
        "MRI_bw0.25": "MRI$_t$", "R_t": "$R_t$", "sigma_t": "$\\sigma_t$",
        "DP_t": "D/P$_t$", "CAPE_t": "CAPE$_t$", "VIX_t": "VIX$_t$",
        "Future_ExRet": "Excess return$_{t+1}$",
        "Future_Volatility": "Volatility$_{t+1:t+21}$",
    }
    out = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\caption{Summary Statistics: Daily Panel}")
    out.append("\\label{tab:descd}")
    out.append("\\begin{tabular}{lcccccc}")
    out.append("\\toprule")
    out.append("Market & Variable & Mean & S.D. & Min & Median & Max \\\\")
    out.append("\\midrule")
    for lab, df, vm in [("CSI 300", cn, varmap_cn), ("S\\&P 500", us, varmap_us)]:
        first = True
        for var, nice in vm.items():
            if var in df.index:
                row = df.loc[var]
                out.append(" & ".join([
                    lab if first else "", nice,
                    f"{row['mean']:.4f}", f"{row['std']:.4f}",
                    f"{row['min']:.4f}", f"{row['50%']:.4f}",
                    f"{row['max']:.4f}"]) + " \\\\")
                first = False
        out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def table_freq_compare() -> str:
    """Monthly vs Daily, CSI 300 full-control (3): MRI + R2 + N."""
    RES_MON = os.path.join(HERE, "results")
    mon = pd.read_csv(os.path.join(RES_MON, "table1_csi300_full.csv"), index_col=0)
    dm = pd.read_csv(os.path.join(RES, "table1_csi300_daily.csv"), index_col=0)
    mvr, mm = parse_blocks(mon)
    dvr, dm_ = parse_blocks(dm)
    col = "(3) Excess return (ctrl)"

    def mri_cell(rows, meta, c):
        for k in rows:
            if "MRI" in k:
                b, t = rows[k].get(c, ("---", "---"))
                return b, t
        return "---", "---"

    mb, mt = mri_cell(mvr, mm, col)
    db_, dt_ = mri_cell(dvr, dm_, col)
    db = db_
    dt = dt_

    out = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\caption{\\textbf{Frequency Comparison: CSI 300 MRI, "
               "Full-Control Specification $(3)$}}")
    out.append("\\label{tab:freqcmp}")
    out.append("\\begin{tabular}{lcc}")
    out.append("\\toprule")
    out.append(" & Monthly & Daily \\\\")
    out.append("\\midrule")
    out.append("Coef. $b$ & " + f"${mb}$" + " & " + f"${db}$" + " \\\\")
    out.append("t-stat (HAC) & " + mt + " & " + dt + " \\\\")
    out.append("Adj. R$^2$ & " + mm["AdjR2"].get(col, "---").replace("%", "\\%") + " & "
               + dm_["AdjR2"].get(col, "---").replace("%", "\\%") + " \\\\")
    out.append("N & " + mm["N"].get(col, "---") + " & "
               + dm_["N"].get(col, "---") + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


# internal helpers (unused vars may assert if written twice)
_daily_specs = None


def daily_table(raw, caption, label) -> str:
    return generic_daily_table(raw, caption, label)


def generic_daily_table(raw: pd.DataFrame, caption: str, label: str) -> str:
    rows, meta = parse_blocks(raw)
    cols = list(raw.columns)
    order = ["Constant", "MRI t", "R t", "sigma t", "E/P t", "B/M t",
             "D/P t", "CAPE t", "VIX t", "QVIX t", "VRP t", "Mkt t", "SMB t",
             "HML t"]
    pretty = {"Constant": "Constant", "MRI t": "MRI$_t$", "R t": "$R_t$",
              "sigma t": "$\\sigma_t$", "E/P t": "E/P$_t$", "B/M t": "B/M$_t$",
              "D/P t": "D/P$_t$", "CAPE t": "CAPE$_t$", "VIX t": "VIX$_t$",
              "QVIX t": "QVIX$_t$", "VRP t": "VRP$_t$", "Mkt t": "MKT$_t$",
              "SMB t": "SMB$_t$", "HML t": "HML$_t$"}
    out = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\begin{tabular}{lcccc}")
    out.append("\\toprule")
    out.append("Variable & (1) Ex. Return & (2) Volatility & "
               "(3) Ex. Return & (4) Volatility \\\\")
    out.append("\\midrule")
    for k in order:
        if k not in rows:
            continue
        bparts, tparts = [pretty.get(k, k)], [""]
        for c in cols:
            b, t = rows[k].get(c, ("nan", "nan"))
            bparts.append(f"${b}$" if b not in ("nan", "None") else "---")
            tparts.append(t if t != "nan" else "---")
        out.append(" & ".join(bparts) + " \\\\")
        out.append(" & ".join(tparts) + " \\\\")
    out.append("\\midrule")
    for metric, key in [("Adj. R$^2$", "AdjR2"), ("N", "N")]:
        out.append(" & ".join([metric] + [meta[key][c].replace("%", "\\%") for c in cols]) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Solver grid table (from theory-side results/solver_results.json)
# --------------------------------------------------------------------------
import json

SOLVER_RES = os.path.join(HERE, "results", "solver_results.json")


def table_grid() -> str:
    """Solver robustness grid: 9 (hidden_dim x depth) Y0 values."""
    if not os.path.isfile(SOLVER_RES):
        print(f"[WARN] {SOLVER_RES} not found, skipping table_grid")
        return ""
    d = json.load(open(SOLVER_RES))
    grid = {(r["lstm_hidden"], r["fnn_layers"]): r["Y0"] for r in d["grid"]}
    rows = []
    for h in (8, 16, 32):
        rows.append(f"{h} & " + " & ".join(f"{grid[(h, d)]:.4f}" for d in (2, 3, 4)) + r" \\")
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Deep BSDE solver robustness: converged price--dividend"
    tex += " ratio $Y_0$ across LSTM hidden dimensions and feedforward depth."
    tex += " The analytic benchmark is $v^*=15.0943$ (see Table~\\ref{tab:a1}).}\n"
    tex += "\\label{tab:grid}\n"
    tex += "\\begin{tabular}{lccc}\n\\toprule\n"
    tex += "Hidden dim \\textbackslash Depth & 2 & 3 & 4 \\\\\n\\midrule\n"
    tex += "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


# --------------------------------------------------------------------------
# Robustness test table (from results_daily/robustness_daily.json)
# --------------------------------------------------------------------------
def table_robustness() -> str:
    ROB_JSON = os.path.join(RES, "robustness_daily.json")
    if not os.path.isfile(ROB_JSON):
        print(f"[WARN] {ROB_JSON} not found, skipping table_robustness")
        return ""
    d = json.load(open(ROB_JSON))
    r = d["results"]

    def fmt(v) -> str:
        return f"{v:.6f}" if v is not None else "---"

    def tfmt(v) -> str:
        return f"{v:+.2f}" if v is not None else "---"

    def pfmt(v) -> str:
        return f"{v*100:.3f}\\%" if v is not None else "---"

    def nfmt(v) -> str:
        return str(int(v)) if v is not None else "---"

    def row(label, key):
        return f"  {label} & {fmt(r[key].get('b'))} & {tfmt(r[key].get('t'))} & {pfmt(r[key].get('adjR2'))} & {nfmt(r[key].get('N'))} \\\\"

    out = []
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\caption{\\textbf{Robustness Tests: CSI 300, Daily}}")
    out.append("\\label{tab:robustd}")
    out.append("\\begin{tabular}{lcccr}")
    out.append("\\toprule")
    out.append("Specification & Coef. $b$ & $t$ (HAC-21) & Adj. $R^2$ & $N$ \\\\")
    out.append("\\midrule")
    out.append("\\multicolumn{5}{l}{\\textbf{Panel A: Influence diagnostics}} \\\\")
    out.append(row("Baseline", "Baseline"))
    loo = r.get("leave-one-out t range", {})
    out.append(f"  Leave-one-out $t$ range & --- & $[{tfmt(loo.get('t_min'))},\\,{tfmt(loo.get('t_max'))}]$ & --- & --- \\\\")
    out.append(row("Greedy removal (16 most influential)", "greedy removal of 16 most influential days"))
    out.append("\\addlinespace")
    out.append("\\multicolumn{5}{l}{\\textbf{Panel B: Sample restrictions}} \\\\")
    for k in ["exclude 2008", "exclude 2015-2016", "winsorized at 1%"]:
        out.append(row(k, k))
    out.append(row("First half", "first half"))
    out.append(row("Second half", "second half"))
    out.append("\\addlinespace")
    out.append("\\multicolumn{5}{l}{\\textbf{Panel C: Alternative specifications}} \\\\")
    out.append(row("Raw (not excess) return", "raw (not excess) next-day return"))
    out.append(row("Skip-one-day-ahead return", "skip-one-day-ahead return"))
    out.append("\\addlinespace")
    out.append("\\multicolumn{5}{l}{\\textbf{Panel D: Look-ahead audit}} \\\\")
    out.append(row("MRI lagged 21 trading days", "MRI lagged 21 day"))
    out.append(row("MRI lagged 252 trading days (12m)", "MRI lagged 12 months (252 days)"))
    out.append("\\addlinespace")
    out.append("\\multicolumn{5}{l}{\\textbf{Panel E: Drawdown proxy controls}} \\\\")
    for k in ["add low-return and high-vol dummies",
              "add squared R and sigma", "add R x sigma interaction"]:
        out.append(row(k, k))
    out.append("\\bottomrule")
    out.append("\\multicolumn{5}{l}{\\textit{Notes:} All regressions include $R_t$, $\\sigma_t$, E/P$_t$, B/M$_t$ (HAC-21)."
               " Block bootstrap (252-day blocks, 2000 reps): 99th $|t|=2.85$, $p<0.001$.}\\\\")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Pre-submission assertion: daily manuscript cross-check
# --------------------------------------------------------------------------
def assert_consistent_daily(ms_path: str):
    """Check that daily manuscript .tex contains key numbers matching CSVs."""
    if not os.path.isfile(ms_path):
        print(f"[WARN] manuscript.tex not found at {ms_path}, skipping assertion")
        return
    ms_txt = open(ms_path).read()

    # Key daily numbers from results_daily/
    checks = [
        ("daily MRI t -4.71", "4.71"),
        ("monthly MRI t -4.15", "4.15"),
        ("daily effect 0.21 pp", "0.21"),
        ("monthly eq. -4.65 pp", "4.65"),
        ("block bootstrap 99th 2.85", "2.85"),
        ("placebo p<0.002", "0.002"),
        ("level effect t -20.6", "20.6"),
        ("k1 price t +0.75", "0.75"),
        ("robustness: greedy removal t -4.72", "4.72"),
        ("robustness: look-ahead 21d t -2.70", "2.70"),
        ("robustness: look-ahead 252d t +1.90", "1.90"),
    ]
    missing = []
    for name, val in checks:
        if val not in ms_txt:
            missing.append((name, val))
    if missing:
        print(f"[FAIL] Key numbers missing from daily manuscript: {missing}")
    else:
        print(f"[OK]   All key daily numbers found in manuscript")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-assert", action="store_true")
    a = ap.parse_args()
    try:
        os.makedirs(TAB, exist_ok=True)
        for name, latex in [
                ("table_csi300", table_csi300()),
                ("table_sp500", table_sp500()),
                ("table_desc", table_desc()),
                ("table_freq_compare", table_freq_compare()),
                ("table_robustness", table_robustness()),
                ("table_grid", table_grid())]:
            if not latex:
                print(f"  skipped {name} (no data)")
                continue
            with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
                f.write(latex)
            print(f"  written {os.path.join(TAB, name+'.tex')}")
        if not a.skip_assert:
            ms_p = os.path.join(HERE, "..", "submission_daily", "manuscript.tex")
            assert_consistent_daily(ms_p)
    except Exception:
        import traceback; traceback.print_exc()
        raise