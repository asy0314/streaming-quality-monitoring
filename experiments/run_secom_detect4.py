#!/usr/bin/env python3
"""SECOM change detection, redesign v4 --- score-anchored monitor.

The three earlier attempts all monitored the LEVEL of the running average against
a fixed target x0.  They died of the same root cause: the running average is
autocorrelated (variance != V/m) and, on misspecified data, slowly biased (drifts
even with no change) -> either it never fires or a Delta=0 null fires.

v4 removes the root cause by NOT averaging the adapting iterate.  We monitor the
SCORE at the fixed in-control reference theta0:
    s_t = (sigmoid(a_t . theta0) - y_t) a_t ,
which is i.i.d. across t and has mean EXACTLY 0 in control (theta0 is the
generating parameter).  A non-overlapping window of m gives
    gbar = mean_s s_t ,   T = m gbar' Sigma^{-1} gbar  ~ chi^2_d   (in control),
standardized by the score covariance Sigma = E[p0(1-p0) a a'] (the info matrix).
Because the s_t are independent and mean-zero, T is stationary by construction:
no relaxation-time issue, no bias drift, no warm-up.  A process shift theta0 ->
theta1 makes E[s_t] != 0, so T (and an EWMA/CUSUM of gbar) climbs -> detection.

Benchmark: REAL SECOM sensor covariates (realistic ill-conditioning), controlled
semi-synthetic logistic labels so the in-control model and the injected shift are
exactly defined.  theta0 is the ridge-logistic fit to the real SECOM labels, so the
coefficients are realistic.

Modes:
  D_MODE=stat  -> in-control STATIONARITY CHECK ONLY (mean T ~ d? flat over time?
                  Delta=0 firing rate ~ alpha? ARL0 ~ 1/alpha?).  Reported first.
  D_MODE=arl   -> full ARL0/ARL1 table (Shewhart + MEWMA), after stat passes.

Env: D_NTOP, D_LAMBDA, D_M, D_REPS, D_ALPHA, D_SHIFTS, D_DIR (coord index or 'min'),
D_ICLEN, D_TAU, D_HORIZON, D_EWMA_R, D_MODE.
Output: secom_detect4_summary.csv (arl mode)
"""
from __future__ import annotations
import math, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE / ".mpl-cache"))
EIG_FLOOR = 1e-10
NTOP = int(os.environ.get("D_NTOP", 20))
LAMBDA = float(os.environ.get("D_LAMBDA", 0.1))
M = int(os.environ.get("D_M", 2000))
REPS = int(os.environ.get("D_REPS", 20))
ALPHA0 = float(os.environ.get("D_ALPHA", 0.05))
SHIFTS = tuple(float(x) for x in os.environ.get("D_SHIFTS", "1,2,3,4").split(","))
DIR = os.environ.get("D_DIR", "min")           # 'min' eigenvector, or a coord index
ICLEN = int(os.environ.get("D_ICLEN", 4000000))
TAU = int(os.environ.get("D_TAU", 200000))
HORIZON = int(os.environ.get("D_HORIZON", 400000))
EWMA_R = float(os.environ.get("D_EWMA_R", 0.2))
MODE = os.environ.get("D_MODE", "stat")
BASE = 20260711


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500))))


def load():
    X = np.genfromtxt(_HERE / "secom_data" / "secom.data", dtype=float)
    lab = np.genfromtxt(_HERE / "secom_data" / "secom_labels.data", usecols=0)
    y = (lab > 0).astype(float)
    cm = np.nanmean(X, 0); nf = np.isnan(X).mean(0)
    Xi = np.where(np.isnan(X), cm, X)
    keep = (Xi.std(0) > 1e-8) & (nf < 0.5); Xi = Xi[:, keep]
    order = np.argsort(-Xi.std(0))[:NTOP]
    Z = StandardScaler().fit_transform(Xi[:, order])
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
    return np.ascontiguousarray(Z), y, Z.shape[1]


def fit_theta0(X, y, d):
    """Ridge-logistic fit to real SECOM labels -> realistic generating parameter."""
    x = np.zeros(d)
    for _ in range(300):
        p = sigmoid(X @ x)
        g = X.T @ (p - y) / X.shape[0] + LAMBDA * x
        W = p * (1 - p); H = (X.T * W) @ X / X.shape[0] + LAMBDA * np.eye(d)
        dx = np.linalg.solve(H, g); x = x - dx
        if np.linalg.norm(dx) < 1e-12: break
    return x


def score_cov(X, theta0):
    """Sigma = E[p0(1-p0) a a'] = info matrix under theta0-generated labels."""
    p = sigmoid(X @ theta0); w = p * (1 - p)
    return (X.T * w) @ X / X.shape[0]


def stream_T(rng, X, theta0, Sigma_inv, m, length, theta1=None, tau=-1):
    n = X.shape[0]; d = X.shape[1]
    idx = rng.integers(0, n, size=length)
    a = X[idx]
    p0 = sigmoid(a @ theta0)
    p_lab = p0.copy()
    if theta1 is not None and tau >= 0:
        p1 = sigmoid(a[tau:] @ theta1)
        p_lab[tau:] = p1
    y = (rng.random(length) < p_lab).astype(float)
    s = (p0 - y)[:, None] * a                       # i.i.d. mean-0 in control
    nwin = length // m
    gbar = s[:nwin * m].reshape(nwin, m, d).mean(axis=1)
    T = m * np.einsum('wi,ij,wj->w', gbar, Sigma_inv, gbar)
    return T, gbar


def main():
    X, y, d = load()
    theta0 = fit_theta0(X, y, d)
    Sigma = score_cov(X, theta0)
    ev, U = eigh(Sigma); ev = np.clip(ev, EIG_FLOOR, None)
    Sigma_inv = U @ np.diag(1.0 / ev) @ U.T
    kappa = ev[-1] / ev[0]
    h = float(chi2.ppf(1 - ALPHA0, d))
    print(f"SECOM score monitor: d={d}  kappa(Sigma)={kappa:.1f}  m={M}  "
          f"chi2_{d},.95={h:.1f}  target ARL0={1/ALPHA0:.0f} win")

    # ---------- STATIONARITY CHECK (in control) ----------
    allT = []
    decile_means = np.zeros((REPS, 10))
    for r in range(REPS):
        rng = np.random.default_rng(BASE + r)
        T, _ = stream_T(rng, X, theta0, Sigma_inv, M, ICLEN)
        allT.append(T)
        # split this rep's windows into 10 time-deciles -> mean T (drift check)
        parts = np.array_split(T, 10)
        decile_means[r] = [p.mean() for p in parts]
    allT = np.concatenate(allT)
    fa = float((allT > h).mean())
    arl0 = (1.0 / fa) if fa > 0 else float("inf")
    dm = decile_means.mean(0)
    print(f"\n  === IN-CONTROL STATIONARITY CHECK  (Delta=0 null) ===")
    print(f"  windows={allT.size}  mean T={allT.mean():.2f} (target d={d})  "
          f"sd={allT.std():.2f}  p95={np.quantile(allT,0.95):.2f} (chi2 p95={h:.1f})")
    print(f"  Delta=0 per-window firing rate={fa:.4f} (target {ALPHA0})  "
          f"ARL0={arl0:.1f} win (target {1/ALPHA0:.0f})")
    print("  mean T over time-deciles (flat => no drift; the attempt-3 killer):")
    print("   " + "  ".join(f"{v:.1f}" for v in dm))
    drift = (dm.max() - dm.min()) / dm.mean()
    print(f"  relative drift across deciles = {drift:.3f}  "
          f"({'FLAT/PASS' if drift < 0.1 else 'DRIFT/FAIL'})")

    if MODE != "arl":
        print("\n(stationarity check only; set D_MODE=arl for the full ARL table)")
        return

    # ---------- ARL0 / ARL1 (Shewhart on T + MEWMA on gbar) ----------
    # shift direction
    if DIR == "min":
        u = U[:, 0]                                  # min-eigenvector of Sigma
        dlabel = "min-eig"
    else:
        u = np.zeros(d); u[int(DIR)] = 1.0; dlabel = f"coord{DIR}"
    uHu = float(u @ Sigma @ u)
    # MEWMA steady-state covariance and calibrated limit (simulate in control)
    Rr = EWMA_R
    Sig_z = (Rr / (2 - Rr)) * Sigma / M
    Sig_z_inv = np.linalg.inv(Sig_z)

    def ewma_stat(gbar):
        z = np.zeros(gbar.shape[1]); Q = np.empty(gbar.shape[0])
        for i in range(gbar.shape[0]):
            z = (1 - Rr) * z + Rr * gbar[i]
            Q[i] = z @ Sig_z_inv @ z
        return Q

    # calibrate MEWMA limit h_e to ARL0 = 1/alpha (in-control quantile of Q)
    Qic = []
    for r in range(REPS):
        rng = np.random.default_rng(BASE + 500 + r)
        _, gb = stream_T(rng, X, theta0, Sigma_inv, M, ICLEN)
        Qic.append(ewma_stat(gb))
    Qic = np.concatenate(Qic)
    h_e = float(np.quantile(Qic, 1 - ALPHA0))
    arl0_shew = 1.0 / max((np.concatenate([stream_T(np.random.default_rng(BASE+900+r),X,theta0,Sigma_inv,M,ICLEN//4)[0] for r in range(4)]) > h).mean(), 1e-9)
    print(f"\n  ARL table: dir={dlabel}, Shewhart h={h:.1f}, MEWMA r={Rr} h_e={h_e:.1f}")

    rows = []
    tau_win = TAU // M
    for D in SHIFTS:
        # scale shift so per-window noncentrality ~ D^2
        c = D / math.sqrt(M * uHu)
        theta1 = theta0 + c * u
        d_shew, d_ewma, det_s, det_e = [], [], 0, 0
        for r in range(REPS):
            rng = np.random.default_rng(BASE + 3000 + int(D * 100) + r)
            length = TAU + HORIZON
            T, gb = stream_T(rng, X, theta0, Sigma_inv, M, length, theta1, TAU)
            Q = ewma_stat(gb)
            fs = next((wi for wi in range(tau_win, T.size) if T[wi] > h), None)
            fe = next((wi for wi in range(tau_win, Q.size) if Q[wi] > h_e), None)
            if fs is not None:
                d_shew.append((fs + 1) * M - TAU); det_s += 1
            if fe is not None:
                d_ewma.append((fe + 1) * M - TAU); det_e += 1
        rows.append({
            "shift": D,
            "ARL1_shewhart": float(np.median(d_shew)) if d_shew else float("nan"),
            "detrate_shewhart": det_s / REPS,
            "ARL1_mewma": float(np.median(d_ewma)) if d_ewma else float("nan"),
            "detrate_mewma": det_e / REPS,
        })
    df = pd.DataFrame(rows)
    print(f"\n  in-control ARL0: Shewhart~{arl0_shew:.0f}, MEWMA={1/ALPHA0:.0f} (calibrated) win  (x{M} obs)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    df.to_csv(_HERE / "secom_detect4_summary.csv", index=False)
    print("\nsaved secom_detect4_summary.csv")


if __name__ == "__main__":
    main()
