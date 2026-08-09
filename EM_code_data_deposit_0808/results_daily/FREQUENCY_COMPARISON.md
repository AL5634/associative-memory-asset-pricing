# Monthly vs. Daily Frequency Comparison

Comparison report generated from the committed results of both pipelines in this deposit. Monthly numbers come from the original monthly (v2) pipeline; daily numbers from this daily-frequency extension.

## 1. CSI 300 — MRI predictive coefficient

| Spec | Monthly (HAC-6) | Daily (HAC-21) |
|---|---|---|
| b, col (1) | -0.1594*** | -0.007191*** |
| t, col (1) | (-3.83) | (-4.36) |
| b, col (3) full ctrl | -0.1887*** | -0.008452*** |
| t, col (3) full ctrl | (-4.15) | (-4.71) |
| Adj. R2, col (3) | 13.16% | 0.720% |
| N, col (3) | 248 | 5039 |

Same scale note: monthly b is per-month excess return; daily b is per-trading-day return. Coefficient interpreted: 1-sd shift in MRI corresponds to monthly b*sd(MRI) pp / month vs daily b*sd(MRI) pp per day.

## 2. S&P 500 — MRI coefficient, full control (3)

| | Monthly | Daily |
|---|---|---|
| b | 0.0175 | -0.000723 |
| t | (1.51) | (-0.86) |

## 3. Identification (CSI 300)

| A2: monthly | A2: + both decile dummies   MRI b=-0.1697  t=-2.77  p=0.0056  adjR2=11.55% |
| A2: daily   | A2: + both decile dummies   MRI b=-0.008469  t=-3.26  p=0.0011  adjR2=1.128% |
| A3: monthly | A3: + R + sigma splines      MRI b=-0.1625  t=-2.58  p=0.0098  adjR2=14.04% |
| A3: daily   | A3: + R + sigma splines      MRI b=-0.007817  t=-2.95  p=0.0031  adjR2=1.071% |
| A4: monthly | A4: + splines + MRIxR/S     MRI b=-0.0693  t=-0.77  p=0.4429  adjR2=13.98% |
| A4: daily   | A4: + splines + MRIxR/S     MRI b=-0.007626  t=-1.66  p=0.0966  adjR2=1.036% |
| Test B placebo (m) | Placebo t mean = -0.469   SD = 0.829 |
| Test B placebo (d) | Placebo t mean = -0.282   SD = 0.501 |
| Test B p-value (m) |   Empirical p-value = <1/500 |
| Test B p-value (d) |   Empirical p-value = <1/500 |
| Test C resid (m) | Residual MRI alone:  b=-0.2599  t=-1.37  p=0.1708  adjR2=6.39% |
| Test C resid (d) | Residual MRI alone:  b=-0.001632  t=-0.13  p=0.9001  adjR2=0.121% |

## 4. Robustness baseline

| Monthly | `Baseline                                     b=-0.1887  t=-4.15  adjR2=13.16%  N=248` |
| Daily   | `Baseline                                     b=-0.008452  t=-4.71  adjR2=0.720%  N=5039` |

## 5. Out-of-sample R2 with MRI (CSI 300 excess return)

See tableA2_oos.csv (monthly) vs tableA2_oos_daily.csv (daily).
