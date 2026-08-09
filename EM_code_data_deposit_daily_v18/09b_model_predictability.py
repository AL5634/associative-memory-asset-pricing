"""
09b_model_predictability.py
===========================
In-model predictability: regress future log price changes on in-model MRI.
Horizon k steps on the model grid (dt = T/N = 0.25 years).

Writes results_daily/model_predictability.json and prints a short table.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RES_D = os.path.join(HERE, "results_daily")
os.makedirs(RES_D, exist_ok=True)

spec = importlib.util.spec_from_file_location("solver", os.path.join(HERE, "04_solver.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

spec2 = importlib.util.spec_from_file_location("ma", os.path.join(HERE, "05_model_analysis.py"))
MA = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(MA)


def ols(y, x):
    """Simple OLS of y on [1, x]; returns beta_x, se, t, n, corr."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid @ resid) / max(n - 2, 1)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t = beta / se
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return dict(alpha=float(beta[0]), beta=float(beta[1]),
                se_beta=float(se[1]), t_beta=float(t[1]),
                n=int(n), corr=corr, r2=float(1 - (resid @ resid) / max(((y - y.mean()) ** 2).sum(), 1e-18)))


def main():
    torch.manual_seed(20260807)
    np.random.seed(20260807)
    cfg = dict(S.BASE_CONFIG, theta=0.4, Y0_init=10.0)
    dev = S.pick_device()
    net = S.Solver(cfg).to(dev)
    pth = os.path.join(RES, "solver.pth")
    if not os.path.isfile(pth):
        print("solver.pth missing; training theta=0.4 ...")
        r = S.train(dict(cfg, epochs=350), seed=1)
        net = r["net"]
        torch.save(net.state_dict(), pth)
    else:
        net.load_state_dict(torch.load(pth, map_location=dev, weights_only=True))
        net.eval()

    n_paths = 8192
    aux = MA.evaluate(net, cfg, n_paths)
    Y, R, vol, logD = aux["Y"], aux["R"], aux["vol"], aux["logD"]
    logP = np.log(np.clip(Y, 1e-9, None)) + logD
    N = cfg["N"]
    dt = cfg["T"] / N
    t0 = N // 2
    mri = MA.model_mri(R, vol, cfg, t0)
    mri_z = (mri - mri.mean()) / (mri.std() + 1e-12)

    # also: contemporaneous level effect P/D ~ MRI
    v0 = Y[:, t0]
    level = ols(np.log(np.clip(v0, 1e-9, None)), mri_z)

    rows = {}
    print("=" * 72)
    print("In-model predictability: fut_k ~ MRI_t0  (theta=0.4)")
    print(f"  grid: T={cfg['T']}, N={N}, dt={dt:.4f} yr (~{dt*12:.1f} months/step)")
    print(f"  n_paths={n_paths}, t0={t0}, MRI mean={mri.mean():.4f} sd={mri.std():.4f}")
    print("-" * 72)
    print(f"  LEVEL: log(P/D)_t0 ~ MRI_z : beta={level['beta']:+.4f}  t={level['t_beta']:+.2f}  R2={level['r2']:.4f}")
    rows["level_log_pd"] = level

    for k in (1, 2, 4, 8):
        t1 = min(t0 + k, N)
        fut = logP[:, t1] - logP[:, t0]
        # also excess-style: change in log P/D only
        dlogv = np.log(np.clip(Y[:, t1], 1e-9, None)) - np.log(np.clip(Y[:, t0], 1e-9, None))
        r_price = ols(fut, mri_z)
        r_pd = ols(dlogv, mri_z)
        # mu_hat contemporaneous
        r_mu = ols(aux["mu_hat"][:, t0], mri_z)
        print(f"  k={k:2d} ({k*dt:.2f}y): price beta={r_price['beta']:+.5f} t={r_price['t_beta']:+.2f} | "
              f"dlog(P/D) beta={r_pd['beta']:+.5f} t={r_pd['t_beta']:+.2f} | "
              f"corr(price,MRI)={r_price['corr']:+.4f}")
        rows[f"k{k}_price"] = r_price
        rows[f"k{k}_dlog_pd"] = r_pd
        if k == 1:
            rows["mu_hat_on_mri"] = r_mu
            print(f"         mu_hat_t0 ~ MRI_z : beta={r_mu['beta']:+.5f} t={r_mu['t_beta']:+.2f}")

    # sign summary
    b1 = rows["k1_price"]["beta"]
    t1 = rows["k1_price"]["t_beta"]
    bl = rows["level_log_pd"]["beta"]
    print("-" * 72)
    print(f"  SIGN level log(P/D)~MRI: {'NEGATIVE' if bl < 0 else 'POSITIVE'} (t={level['t_beta']:+.2f})")
    print(f"  SIGN k=1 price~MRI:      {'NEGATIVE' if b1 < 0 else 'POSITIVE'} (t={t1:+.2f})")
    print("=" * 72)

    out = dict(
        cfg=dict(theta=cfg["theta"], gamma_mem=cfg["gamma_mem"], lam=cfg["lam"],
                 T=cfg["T"], N=N, dt=dt, n_paths=n_paths, t0=t0),
        mri_mean=float(mri.mean()), mri_sd=float(mri.std()),
        regressions=rows,
        interpretation=dict(
            level_sign="negative" if bl < 0 else "positive",
            k1_price_sign="negative" if b1 < 0 else "positive",
            k1_price_beta=b1, k1_price_t=t1,
            level_beta=bl, level_t=level["t_beta"],
            horizon_note="model step = 0.25 years; not daily",
        ),
    )
    path = os.path.join(RES_D, "model_predictability.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {path}")


def figure_loss():
    """Plot training loss curve from simulated_paths.npz."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping Figure_loss")
        return
    npz_path = os.path.join(RES, "simulated_paths.npz")
    if not os.path.isfile(npz_path):
        print(f"[WARN] {npz_path} not found, skipping Figure_loss")
        return
    z = np.load(npz_path)
    if "loss_hist" not in z:
        print("[WARN] loss_hist not in simulated_paths.npz, skipping Figure_loss")
        return
    lh = z["loss_hist"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.semilogy(np.arange(1, len(lh) + 1), lh, color="0.2", lw=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title(r"Deep BSDE training loss ($\theta=0.4$)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(os.path.join(HERE, "figures"), "Figure_loss.pdf")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    fig.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"  written {fig_path}")


if __name__ == "__main__":
    main()
    figure_loss()
