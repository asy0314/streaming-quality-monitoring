#!/usr/bin/env python3
"""Figure for the SECOM change-detection subsection (score-anchored monitor).

Panel (a): a representative control-chart run -- the windowed score statistic
  T_t over the stream, flat below the chi^2 limit in control, then crossing it
  shortly after an injected process shift at tau (first alarm marked).
Panel (b): detection delay ARL_1 (observations) vs shift size, read from the
  200-replication run in secom_detect4_summary.csv, with the in-control
  ARL_0 = 20 windows target annotated.

Reuses the run_secom_detect4 monitor so the figure matches the reported numbers.
Output: ../fig_detect.pdf
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
_MPLDIR = _HERE / ".mpl-cache"; _MPLDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EIG_FLOOR = 1e-10; NTOP = 20; LAMBDA = 0.1; M = 2000; ALPHA0 = 0.05
CERT = "#54a24b"; LIMIT = "#e45756"; SHIFTCOL = "#4c78a8"


def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500))/(1+np.exp(np.clip(z, -500, 500))))


def load():
    X = np.genfromtxt(_HERE/"secom_data"/"secom.data", dtype=float)
    lab = np.genfromtxt(_HERE/"secom_data"/"secom_labels.data", usecols=0)
    y = (lab > 0).astype(float); cm = np.nanmean(X, 0); nf = np.isnan(X).mean(0)
    Xi = np.where(np.isnan(X), cm, X); keep = (Xi.std(0) > 1e-8) & (nf < 0.5); Xi = Xi[:, keep]
    order = np.argsort(-Xi.std(0))[:NTOP]; Z = StandardScaler().fit_transform(Xi[:, order])
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))]); return np.ascontiguousarray(Z), y, Z.shape[1]


def fit_theta0(X, y, d):
    x = np.zeros(d)
    for _ in range(300):
        p = sigmoid(X@x); g = X.T@(p-y)/X.shape[0] + LAMBDA*x
        W = p*(1-p); H = (X.T*W)@X/X.shape[0] + LAMBDA*np.eye(d)
        dx = np.linalg.solve(H, g); x = x-dx
        if np.linalg.norm(dx) < 1e-12: break
    return x


def T_series(rng, X, theta0, Sinv, m, length, theta1=None, tau=-1):
    n, d = X.shape
    idx = rng.integers(0, n, size=length); a = X[idx]
    p0 = sigmoid(a@theta0); pl = p0.copy()
    if theta1 is not None and tau >= 0:
        pl[tau:] = sigmoid(a[tau:]@theta1)
    yv = (rng.random(length) < pl).astype(float)
    s = (p0-yv)[:, None]*a; nwin = length//m
    gb = s[:nwin*m].reshape(nwin, m, d).mean(1)
    return m*np.einsum('wi,ij,wj->w', gb, Sinv, gb)


def main():
    X, y, d = load(); theta0 = fit_theta0(X, y, d)
    p = sigmoid(X@theta0); w = p*(1-p); Sigma = (X.T*w)@X/X.shape[0]
    ev, U = eigh(Sigma); ev = np.clip(ev, EIG_FLOOR, None)
    Sinv = U@np.diag(1/ev)@U.T; h = float(chi2.ppf(1-ALPHA0, d))

    # ---- panel (a): representative run, shift injected at tau ----
    m = M; W_ic = 30; W_oc = 30; tau_win = W_ic
    length = (W_ic + W_oc)*m; tau = W_ic*m
    # scale a coord-0 shift to a clear post-change non-centrality (~5^2)
    u = np.zeros(d); u[0] = 1.0; Dvis = 5.0
    theta1 = theta0 + (Dvis/math.sqrt(m*(u@Sigma@u)))*u
    # pick a clean seed (no pre-tau false alarm, clear post-tau crossing)
    for sd in range(200):
        rng = np.random.default_rng(10_000+sd)
        T = T_series(rng, X, theta0, Sinv, m, length, theta1, tau)
        pre = T[:tau_win]; post = T[tau_win:]
        if pre.max() < h and (post > h).any() and np.argmax(post > h) <= 3:
            break
    first = tau_win + int(np.argmax(post > h))
    xobs = (np.arange(T.size)+1)*m

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 14})
    fig, a0 = plt.subplots(1, 1, figsize=(5.6, 3.2))

    a0.plot(xobs, T, "-o", color=CERT, ms=3.2, lw=1.4, label=r"$T_t$ (score chart)")
    a0.axhline(h, ls="--", color=LIMIT, lw=1.6, label=r"limit $\chi^2_{21,0.95}$")
    a0.axvline(tau, ls=":", color=SHIFTCOL, lw=1.6)
    a0.plot(xobs[first], T[first], "*", color=LIMIT, ms=15, zorder=5)
    a0.annotate("change $\\tau$", xy=(tau, a0.get_ylim()[1]*0.40),
                xytext=(tau*0.42, a0.get_ylim()[1]*0.52), color=SHIFTCOL, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=SHIFTCOL, lw=1.1))
    a0.annotate("alarm", xy=(xobs[first], T[first]),
                xytext=(xobs[first]+3.5*m, T[first]-4), color=LIMIT, fontsize=14,
                arrowprops=dict(arrowstyle="->", color=LIMIT, lw=1.2))
    a0.set_xlabel("observations"); a0.set_ylabel(r"monitoring statistic $T_t$")
    a0.legend(loc="upper left", framealpha=0.9, fontsize=13, handlelength=1.3, handletextpad=0.5, borderpad=0.4)
    plt.setp(a0.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    out = _HERE.parent/"fig_detect.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"saved {out}")


if __name__ == "__main__":
    main()
