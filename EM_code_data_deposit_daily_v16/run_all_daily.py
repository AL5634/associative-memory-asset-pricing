# run_all_daily.py — Daily-frequency reproduction pipeline
# Run: python run_all_daily.py
# Requires the committed data_real/ snapshots (01b derives panels offline;
# it will NOT touch the network).

import subprocess, sys, time, json, hashlib, os

SCRIPTS = [
    "01b_build_data_daily.py",
    "02b_analysis_daily.py",
    "03b_robustness_daily.py",
    "06b_identification_daily.py",
    "07b_make_tables_daily.py",
]
t0 = time.time()
for s in SCRIPTS:
    print(f"[RUN]  {s} ...")
    r = subprocess.run(["python3", s], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"[FAIL] {s}\n{r.stderr[:500]}")
        sys.exit(1)
    print(f"[OK]   {s}")

RES = "results_daily"
manifest = {}
for fn in sorted(os.listdir(RES)):
    if fn.endswith((".csv", ".json", ".txt")) and fn != "manifest_daily.json":
        h = hashlib.sha256(open(os.path.join(RES, fn), "rb").read()).hexdigest()
        manifest[fn] = h
with open(os.path.join(RES, "manifest_daily.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[OK]   manifest written  ({time.time()-t0:.0f}s total)")