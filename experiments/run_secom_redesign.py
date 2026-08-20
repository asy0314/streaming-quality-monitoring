#!/usr/bin/env python
"""SECOM redesign runner -- Experiment A (self-calibration) + B (kappa sweep).

Supersedes the earlier EMA / plug-in runners (same data pipeline, optimizer
updates, gain schedules, base seeds), with these corrections:

  * corrected oracle meat  Sigma = S - lambda^2 x* x*^T; both
    the corrected and the raw (shipped) sandwiches are reported so old and new
    numbers stay comparable.
  * SECOM_MODE=anchor : d=21, six methods, plug-in Vhat accumulated for the
    certified arms -> one pass yields Experiment A and B's kappa~7 anchor.
  * SECOM_MODE=sweep  : SECOM_K in {40,60,90}, four diagonal arms
    (identity, SA-RMSProp, ema_05, ema_099), no plug-in -> B's sweep points.
  * cross-seed empirical covariance V_emp(n) = n * Cov_seeds(xbar_n), computed
    in the parent; coverage scored against plug-in / corrected oracle /
    raw oracle / V_emp. Vhat consistency scored against BOTH the analytic
    corrected sandwich (deterministic target, no MC floor) and V_emp
    (MC floor ~ sqrt(2/(seeds-1)) -- dominant at small seed counts).

Env: SECOM_MODE(anchor) SECOM_K(20) SECOM_METHODS SECOM_PLUGIN
     SECOM_SEEDS(200) SECOM_NMAX(1e7) SECOM_WORKERS SECOM_LAMBDA(0.1)
     SECOM_ETA0(1.0) SECOM_TAG("")
Output: secom_redesign_<mode>_d<d>[_<tag>].csv
"""
from __future__ import annotations
import math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

_HERE = Path(__file__).resolve().parent

MODE = os.environ.get("SECOM_MODE", "anchor")
assert MODE in ("anchor", "sweep"), f"bad SECOM_MODE={MODE}"
N_TOP = int(os.environ.get("SECOM_K", 20))
_DEFAULT_METHODS = ("identity,adagrad,rmsprop,ons,ema_05,ema_099" if MODE == "anchor"
                    else "identity,rmsprop,ema_05,ema_099")
_DEFAULT_PLUGIN = ("identity,adagrad,rmsprop,ons" if MODE == "anchor" else "")
METHODS = [m for m in os.environ.get("SECOM_METHODS", _DEFAULT_METHODS).split(",") if m]
PLUGIN = {m for m in os.environ.get("SECOM_PLUGIN", _DEFAULT_PLUGIN).split(",") if m}
PLUGIN &= set(METHODS)

EMA_RHO = {"ema_05": 0.5, "ema_099": 0.999}
EIG_FLOOR = 1e-10
ALPHA = 0.7
ETA0 = float(os.environ.get("SECOM_ETA0", 1.0))
SA_C = 1.0
LAMBDA_REG = float(os.environ.get("SECOM_LAMBDA", 0.1))
GRADIENT_CLIP = 50.0
# Preconditioner offset epsilon (Q_0 = eps*I and the denominator floor). The
# shipped scripts hardwire eps=1, which dominates SECOM's small weighted second
# moments (E[w a^2+lambda] ~ 0.27) and leaves the RMSProp/AdaGrad preconditioner
# nearly inert (~identity) -- so the constant-gain EMA has nothing to destabilize.
# A deployment-realistic (smaller) eps activates the preconditioner. Accumulators
# keep their unit warm-start init so the eps=1-tuned eta_0=1 stays stable early.
EPS = float(os.environ.get("SECOM_EPS", 1.0))
Z_VALUE = 1.96
SEED_COUNT = int(os.environ.get("SECOM_SEEDS", 200))
BASE_SEED = 20260711          # smoke seeds are a prefix of production seeds
NMAX = int(float(os.environ.get("SECOM_NMAX", 10_000_000)))
TAG = os.environ.get("SECOM_TAG", "")
SAMPLE_SIZES = tuple(sorted(set(int(x) for x in np.unique(
    np.logspace(np.log10(2000), np.log10(NMAX), 16).astype(int)))))
_G = {}


def spd_inv_sqrt(matrix, eig_floor=EIG_FLOOR):
    ev, U = eigh(matrix, check_finite=False)
    ev = np.clip(ev, eig_floor, None)
    return U @ np.diag(ev ** -0.5) @ U.T


def load_data():
    X = np.genfromtxt(_HERE / "secom_data" / "secom.data", dtype=float)
    lab = np.genfromtxt(_HERE / "secom_data" / "secom_labels.data", usecols=0)
    y = (lab > 0).astype(float)
    col_mean = np.nanmean(X, axis=0); nanfrac = np.isnan(X).mean(axis=0)
    Xi = np.where(np.isnan(X), col_mean, X)
    keep = (Xi.std(axis=0) > 1e-8) & (nanfrac < 0.5); Xi = Xi[:, keep]
    order = np.argsort(-Xi.std(axis=0))[:N_TOP]
    Z = StandardScaler().fit_transform(Xi[:, order])
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
    return Z, y, Z.shape[1]


def compute_truth(X, y, d):
    n = X.shape[0]; x = np.zeros(d)
    for _ in range(300):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ x, -500, 500)))
        grad = X.T @ (p - y) / n + LAMBDA_REG * x
        W = p * (1.0 - p); H = (X.T * W) @ X / n + LAMBDA_REG * np.eye(d)
        dx = np.linalg.solve(H, grad); x = x - dx
        if np.linalg.norm(dx) < 1e-12: break
    x_star = x
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ x_star, -500, 500))); W = p * (1.0 - p)
    H = (X.T * W) @ X / n + LAMBDA_REG * np.eye(d)
    res = p - y; G = X * res[:, None]
    S_raw = G.T @ G / n                                   # shipped oracle meat
    Sigma = S_raw - LAMBDA_REG**2 * np.outer(x_star, x_star)   # REDESIGN 0.1
    ev, U = eigh(H); ev = np.clip(ev, EIG_FLOOR, None)
    H_inv = U @ np.diag(1.0 / ev) @ U.T
    V_raw = H_inv @ S_raw @ H_inv
    V_corr = H_inv @ Sigma @ H_inv
    return {"x_star": x_star, "V_corr": V_corr,
            "sdq_corr": np.sqrt(np.clip(np.diag(V_corr), 0, None)),
            "sdq_raw": np.sqrt(np.clip(np.diag(V_raw), 0, None)),
            "tr_corr": float(np.trace(V_corr)), "tr_raw": float(np.trace(V_raw)),
            "mism_raw": float(np.linalg.norm(S_raw - H) / np.linalg.norm(H)),
            "mism_corr": float(np.linalg.norm(Sigma - H) / np.linalg.norm(H)),
            "condition_number": float(ev[-1] / ev[0])}


def _init(X, y, truth, d):
    _G.update(X=X, y=y, truth=truth, d=d)


def run_one_seed(seed):
    X = _G["X"]; y = _G["y"]; truth = _G["truth"]; d = _G["d"]
    rng = np.random.default_rng(seed)
    n_data = X.shape[0]; n_max = max(SAMPLE_SIZES); checkpoints = set(SAMPLE_SIZES)
    eye_d = np.eye(d); lamI = LAMBDA_REG * eye_d
    x_star = truth["x_star"]; sdq_c = truth["sdq_corr"]; sdq_r = truth["sdq_raw"]
    tr_c = truth["tr_corr"]; tr_r = truth["tr_raw"]; V_corr = truth["V_corr"]
    st = {m: {"x": np.zeros(d), "x_sum": np.zeros(d),
              "C": eye_d.copy() if m == "adagrad" else None,
              "B": eye_d.copy() if m == "ons" else None,
              "D": np.ones(d) if (m == "rmsprop" or m in EMA_RHO) else None,
              "Hacc": np.zeros((d, d)) if m in PLUGIN else None,
              "Sacc": np.zeros((d, d)) if m in PLUGIN else None} for m in METHODS}
    idxs = rng.integers(0, n_data, size=n_max)
    rows = []
    for t in range(n_max):
        step = t + 1; a_t = X[idxs[t]]; y_t = y[idxs[t]]
        eta_t = ETA0 * step ** (-ALPHA); rho_t = SA_C / (step + 1.0)
        for m in METHODS:
            s = st[m]; s["x_sum"] += s["x"]
            logit = float(a_t @ s["x"])
            p_t = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
            w_t = p_t * (1.0 - p_t)
            g_full = (p_t - y_t) * a_t + LAMBDA_REG * s["x"]
            if s["Hacc"] is not None:            # unclipped score, as in the shipped plug-in
                s["Hacc"] += w_t * np.outer(a_t, a_t)
                s["Sacc"] += np.outer(g_full, g_full)
            g_t = g_full
            gn = float(np.linalg.norm(g_t))
            if gn > GRADIENT_CLIP: g_t = g_t * (GRADIENT_CLIP / gn)
            if m == "identity":
                pre = g_t
            elif m == "adagrad":
                pre = spd_inv_sqrt(s["C"] + EPS * eye_d) @ g_t
                s["C"] = (1 - rho_t) * s["C"] + rho_t * (w_t * np.outer(a_t, a_t) + lamI)
            elif m == "ons":
                pre = np.linalg.solve(s["B"] + EPS * eye_d, g_t)
                s["B"] = (1 - rho_t) * s["B"] + rho_t * (w_t * np.outer(a_t, a_t) + lamI)
            elif m == "rmsprop":
                pre = g_t / np.sqrt(s["D"] + EPS)
                s["D"] = (1 - rho_t) * s["D"] + rho_t * (w_t * (a_t * a_t) + LAMBDA_REG)
            else:                                 # ema_05 / ema_099: constant gain
                pre = g_t / np.sqrt(s["D"] + EPS)
                rr = EMA_RHO[m]
                s["D"] = (1 - rr) * s["D"] + rr * (w_t * (a_t * a_t) + LAMBDA_REG)
            s["x"] = s["x"] - eta_t * pre
        if step in checkpoints:
            for m in METHODS:
                s = st[m]; x_bar = s["x_sum"] / step; err = x_bar - x_star
                row = {"seed": seed, "method": m, "n": step,
                       "nmse": float(step * np.dot(err, err) / tr_c),
                       "nmse_raw": float(step * np.dot(err, err) / tr_r),
                       "xbar": x_bar.copy(), "Vhat": None,
                       "cov_plugin": np.nan, "vhat_vs_sigma": np.nan}
                se_c = sdq_c / math.sqrt(step)
                row["cov_oracle"] = float(np.mean(np.abs(err) <= Z_VALUE * se_c))
                se_r = sdq_r / math.sqrt(step)
                row["cov_oracle_raw"] = float(np.mean(np.abs(err) <= Z_VALUE * se_r))
                if s["Hacc"] is not None:
                    Hhat = s["Hacc"] / step + lamI
                    Shat = s["Sacc"] / step
                    Hi = np.linalg.inv(Hhat)
                    Vhat = Hi @ Shat @ Hi
                    se_pi = np.sqrt(np.clip(np.diag(Vhat), 0, None) / step)
                    row["cov_plugin"] = float(np.mean(np.abs(err) <= Z_VALUE * se_pi))
                    row["vhat_vs_sigma"] = float(np.linalg.norm(Vhat - V_corr) /
                                                 np.linalg.norm(V_corr))
                    row["Vhat"] = Vhat
                rows.append(row)
    return rows


def main():
    X, y, d = load_data()
    print(f"SECOM redesign [{MODE}]: n_data={X.shape[0]}, d={d}, "
          f"fails={int(y.sum())} ({y.mean():.1%})", flush=True)
    truth = compute_truth(X, y, d)
    print(f"  kappa(H)={truth['condition_number']:.1f}  "
          f"||S-H||/||H|| raw={truth['mism_raw']:.4f} corr={truth['mism_corr']:.4f}  "
          f"Tr raw={truth['tr_raw']:.3f} corr={truth['tr_corr']:.3f}", flush=True)
    workers = int(os.environ.get("SECOM_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
    print(f"  methods={METHODS} plugin={sorted(PLUGIN)} seeds={SEED_COUNT} "
          f"n_max={max(SAMPLE_SIZES)} lambda={LAMBDA_REG} eps={EPS:g} workers={workers}", flush=True)
    seeds = [int(s) for s in BASE_SEED + np.arange(SEED_COUNT)]
    all_rows = []; t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(X, y, truth, d)) as ex:
        for rows in ex.map(run_one_seed, seeds, chunksize=1):
            all_rows.extend(rows); done += 1
            if done % 5 == 0 or done == SEED_COUNT:
                el = time.time() - t0
                print(f"  seed {done}/{SEED_COUNT} elapsed={el:.0f}s "
                      f"eta={el/done*(SEED_COUNT-done):.0f}s", flush=True)
    # ---- cross-seed post-processing: V_emp per (method, n) -----------------
    x_star = truth["x_star"]
    by_key = {}
    for r in all_rows:
        by_key.setdefault((r["method"], r["n"]), []).append(r)
    for (m, n), rows in by_key.items():
        XB = np.stack([r["xbar"] for r in rows])              # R x d
        V_emp = n * np.cov(XB.T, ddof=1)                      # MC noise ~ sqrt(2/(R-1))
        se_e = np.sqrt(np.clip(np.diag(V_emp), 0, None) / n)
        nV = np.linalg.norm(V_emp)
        for r in rows:
            err = r["xbar"] - x_star
            r["cov_emp"] = float(np.mean(np.abs(err) <= Z_VALUE * se_e))
            r["vhat_vs_emp"] = (float(np.linalg.norm(r["Vhat"] - V_emp) / nV)
                                if r["Vhat"] is not None and nV > 0 else np.nan)
            r.pop("xbar"); r.pop("Vhat")
    df = pd.DataFrame(all_rows)
    sem = lambda v: v.std() / math.sqrt(len(v))
    summ = (df.groupby(["method", "n"]).agg(
        cov_plugin_mean=("cov_plugin", "mean"), cov_plugin_se=("cov_plugin", sem),
        cov_oracle_mean=("cov_oracle", "mean"), cov_oracle_se=("cov_oracle", sem),
        cov_oracle_raw_mean=("cov_oracle_raw", "mean"),
        cov_emp_mean=("cov_emp", "mean"), cov_emp_se=("cov_emp", sem),
        vhat_vs_sigma_mean=("vhat_vs_sigma", "mean"),
        vhat_vs_emp_mean=("vhat_vs_emp", "mean"),
        nmse_mean=("nmse", "mean"), nmse_se=("nmse", sem),
        nmse_raw_mean=("nmse_raw", "mean")).reset_index())
    summ.insert(0, "mode", MODE); summ.insert(1, "d", d)
    summ.insert(2, "kappa", round(truth["condition_number"], 3))
    summ.insert(3, "eps", EPS); summ.insert(4, "seeds", SEED_COUNT)
    _epstag = f"_eps{EPS:g}" if EPS != 1.0 else ""
    out = _HERE / f"secom_redesign_{MODE}_d{d}{_epstag}{('_' + TAG) if TAG else ''}.csv"
    summ.to_csv(out, index=False)
    print(f"saved {out.name}  wall={time.time()-t0:.0f}s", flush=True)
    fin = summ[summ.n == max(SAMPLE_SIZES)].sort_values("method")
    for _, r in fin.iterrows():
        pi = ("plug={:.3f}+-{:.3f} ".format(r.cov_plugin_mean, r.cov_plugin_se)
              if not np.isnan(r.cov_plugin_mean) else "")
        vh = ("Vh~Sig={:.3f} Vh~emp={:.3f} ".format(r.vhat_vs_sigma_mean, r.vhat_vs_emp_mean)
              if not np.isnan(r.vhat_vs_sigma_mean) else "")
        print(f"  {r.method:9s} n={int(r.n)}: {pi}orc={r.cov_oracle_mean:.3f} "
              f"raw={r.cov_oracle_raw_mean:.3f} emp={r.cov_emp_mean:.3f} "
              f"{vh}nmse={r.nmse_mean:.3f} (raw {r.nmse_raw_mean:.3f})", flush=True)


if __name__ == "__main__":
    main()
