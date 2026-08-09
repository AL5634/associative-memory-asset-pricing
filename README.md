©️Liu Xiaoquan, Assistant Professor and Program director BBA in Finance,Hong Kong Chu Hai College ;Offshore Fund Director,Shanghai AJ Group

Paper: "Associative Memory, Path-Dependent Beliefs, and Asset Pricing
Dynamics: A Deep Learning Approach" 

Reproducibility
---------------
1. `python run_all_daily.py` reproduces every results_daily/ file.
2. Model-side: `python 04_solver.py --stage all` (optional if results/ committed),
   `python 05_model_analysis.py`, `python 09b_model_predictability.py`.
3. Daily identification test B uses rng seed 20260807; placebo distribution
   reproducible (mean -0.282, sd 0.501, |t| 99th pctile 1.55, p<1/500).
   All results committed and hash-verified.
