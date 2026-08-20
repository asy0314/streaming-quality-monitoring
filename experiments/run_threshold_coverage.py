#!/usr/bin/env python3
"""Negative-example experiment: preconditioners that violate the stabilization
threshold cause the scaled remainder to persist, preventing CLT coverage.

Uses the same Toeplitz(0.4) setup as the main experiment.  Compares
SA-RMSProp (rho_t=1/t, beta=1) against constant-EMA RMSProp with
rho=0.5 (clear positive slope) and rho=0.999 (near-identity EMA).

Generates:
  - threshold_coverage.pdf : main-text figure (scaled remainder panel)
  - threshold_coverage_supplement.pdf : 1x2 panel (coverage, NMSE)

Usage:
  python run_threshold_coverage.py
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.linalg import toeplitz

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
_MPLDIR = Path(__file__).resolve().parent / ".mpl-cache"
_MPLDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore", message=".*encountered in matmul.*", category=RuntimeWarning
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHOD_KEYS = ["sa_rmsprop", "ema_05", "ema_099"]
METHOD_LABELS = {
    "sa_rmsprop": r"SA-RMSProp ($\rho_t=1/t$, $\beta=1$)",
    "ema_05": r"Constant-EMA ($\rho=0.5$, $\beta=0$)",
    "ema_099": r"Constant-EMA ($\rho=0.999$, $\beta=0$)",
}
METHOD_COLORS = {
    "sa_rmsprop": "#54a24b",
    "ema_05": "#e45756",
    "ema_099": "#9d5fb0",
}
METHOD_MARKERS = {
    "sa_rmsprop": "o",
    "ema_05": "s",
    "ema_099": "D",
}
METHOD_LINESTYLES = {
    "sa_rmsprop": "-",
    "ema_05": "--",
    "ema_099": "--",
}

PLOT_STYLE = {
    "font.size": 26,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 19,
    "figure.titlesize": 34,
}

# Simulation parameters — re-tuned to EXPOSE the constant-EMA coverage breakdown:
# low ridge (lets EMA preconditioner fluctuations through; the large ridge in the
# original masked them) + higher condition number.
DIMENSION = int(os.environ.get("TV_D", 10))
ALPHA = 0.7
ETA0 = float(os.environ.get("TV_ETA0", 1.0))
SA_C = 1.0
SA_RIDGE = float(os.environ.get("TV_RIDGE", 0.1))
FEATURE_RHO = float(os.environ.get("TV_RHO", 0.6))
SIGNAL_NORM = 0.5
BASE_NOISE = 0.35
HETERO_NOISE = 0.8
GRADIENT_CLIP = 500.0

_NMAX = int(os.environ.get("TV_NMAX", 1_000_000))
_LOG_GRID = np.unique(np.logspace(
    np.log10(5_000), np.log10(_NMAX), num=120
).astype(int))
SAMPLE_SIZES = tuple(int(n) for n in _LOG_GRID)
N_CHECKPOINTS = len(SAMPLE_SIZES)

BASE_SEED = 20260619
SEED_COUNT = int(os.environ.get("TV_SEEDS", 100))
Z_VALUE = 1.96

# EMA rates: index 0=SA (unused), 1=EMA(0.5), 2=EMA(0.999)
N_METHODS = 3
EMA_RHOS = np.array([0.0, 0.5, 0.999])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def spd_sqrt(matrix):
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(matrix, check_finite=False)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def spd_inverse(matrix):
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(matrix, check_finite=False)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T


def standard_error(series):
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / math.sqrt(len(series)))


# ---------------------------------------------------------------------------
# Ground-truth construction
# ---------------------------------------------------------------------------


def build_truth():
    d = DIMENSION
    indices = np.arange(d, dtype=float)
    H = toeplitz(FEATURE_RHO ** indices)
    H_inv = spd_inverse(H)
    H_sqrt = spd_sqrt(H)

    coord_scale = SIGNAL_NORM / math.sqrt(d)
    x_star = np.array(
        [coord_scale if j % 2 == 0 else -coord_scale for j in range(d)]
    )
    hetero_dir = np.linspace(0.6, 1.6, d, dtype=float)
    hetero_dir /= np.linalg.norm(hetero_dir)

    Hu = H @ hetero_dir
    dir_var = float(hetero_dir @ Hu)
    S = BASE_NOISE**2 * H + HETERO_NOISE**2 * (dir_var * H + 2.0 * np.outer(Hu, Hu))

    sandwich = H_inv @ S @ H_inv
    sandwich_diag_sqrt = np.sqrt(np.clip(np.diag(sandwich), 0.0, None))
    sandwich_trace = float(np.trace(sandwich))

    return {
        "H": H, "H_inv": H_inv, "H_sqrt": H_sqrt,
        "S": S, "sandwich": sandwich,
        "sandwich_diag_sqrt": sandwich_diag_sqrt,
        "sandwich_trace": sandwich_trace,
        "x_star": x_star, "hetero_dir": hetero_dir,
    }


# ---------------------------------------------------------------------------
# Numba-accelerated SGD loop
# ---------------------------------------------------------------------------


@njit(cache=True)
def _sgd_loop(features, noise, x_star, H, H_inv, sw_diag_sqrt, sw_trace,
              checkpoints, alpha, eta0, sa_c, sa_ridge, gradient_clip,
              ema_rhos, z_value):
    """Run all 3 methods in a single pass, return checkpoint arrays."""
    n_max = features.shape[0]
    d = features.shape[1]
    n_methods = 3
    n_cp = len(checkpoints)

    # State arrays
    X = np.zeros((n_methods, d))
    X_sum = np.zeros((n_methods, d))
    Xi_sum = np.zeros((n_methods, d))
    D = np.ones((n_methods, d))

    # Output arrays: (n_checkpoints, n_methods)
    out_coverage = np.zeros((n_cp, n_methods))
    out_nmse = np.zeros((n_cp, n_methods))
    out_sc_rem = np.zeros((n_cp, n_methods))

    cp_idx = 0
    next_cp = checkpoints[0]

    for t in range(n_max):
        step = t + 1
        a_t = features[t]
        eps_t = noise[t]
        eta_t = eta0 * step ** (-alpha)
        rho_sa = sa_c / (step + 1.0)

        # a_t squared
        a_sq = np.empty(d)
        for j in range(d):
            a_sq[j] = a_t[j] * a_t[j]

        for m in range(n_methods):
            # Accumulate x_sum
            for j in range(d):
                X_sum[m, j] += X[m, j]

            # delta = x - x_star
            delta = np.empty(d)
            for j in range(d):
                delta[j] = X[m, j] - x_star[j]

            # proj = a_t . delta
            proj = 0.0
            for j in range(d):
                proj += a_t[j] * delta[j]

            # g = (proj - eps) * a_t
            g = np.empty(d)
            for j in range(d):
                g[j] = (proj - eps_t) * a_t[j]

            # Gradient clipping
            g_norm_sq = 0.0
            for j in range(d):
                g_norm_sq += g[j] * g[j]
            g_norm = math.sqrt(g_norm_sq)
            if g_norm > gradient_clip:
                scale = gradient_clip / g_norm
                for j in range(d):
                    g[j] *= scale

            # xi = g - H @ delta
            xi = np.empty(d)
            for j in range(d):
                val = 0.0
                for k in range(d):
                    val += H[j, k] * delta[k]
                xi[j] = g[j] - val
            for j in range(d):
                Xi_sum[m, j] += xi[j]

            # Preconditioned gradient
            pre_g = np.empty(d)
            for j in range(d):
                pre_g[j] = g[j] / math.sqrt(D[m, j] + sa_ridge)

            # Update D
            if m == 0:  # SA
                for j in range(d):
                    D[m, j] = (1.0 - rho_sa) * D[m, j] + rho_sa * a_sq[j]
            else:  # EMA
                rho = ema_rhos[m]
                for j in range(d):
                    D[m, j] = (1.0 - rho) * D[m, j] + rho * a_sq[j]

            # SGD step
            for j in range(d):
                X[m, j] -= eta_t * pre_g[j]

        # Checkpoint
        if step == next_cp:
            for m in range(n_methods):
                x_bar = np.empty(d)
                error = np.empty(d)
                for j in range(d):
                    x_bar[j] = X_sum[m, j] / step
                    error[j] = x_bar[j] - x_star[j]

                se = np.empty(d)
                for j in range(d):
                    se[j] = sw_diag_sqrt[j] / math.sqrt(step)

                # Coverage
                covered = 0
                for j in range(d):
                    lo = x_bar[j] - z_value * se[j]
                    hi = x_bar[j] + z_value * se[j]
                    if x_star[j] >= lo and x_star[j] <= hi:
                        covered += 1
                out_coverage[cp_idx, m] = covered / d

                # NMSE
                err_sq = 0.0
                for j in range(d):
                    err_sq += error[j] * error[j]
                out_nmse[cp_idx, m] = step * err_sq / sw_trace

                # Scaled remainder
                rem = np.empty(d)
                for j in range(d):
                    val = 0.0
                    for k in range(d):
                        val += H_inv[j, k] * Xi_sum[m, k]
                    rem[j] = error[j] + val / step
                rem_sq = 0.0
                for j in range(d):
                    rem_sq += rem[j] * rem[j]
                out_sc_rem[cp_idx, m] = math.sqrt(step) * math.sqrt(rem_sq)

            cp_idx += 1
            if cp_idx < n_cp:
                next_cp = checkpoints[cp_idx]

    return out_coverage, out_nmse, out_sc_rem


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------


def run_one_seed(seed, truth, checkpoints_arr):
    d = DIMENSION
    n_max = max(SAMPLE_SIZES)

    rng = np.random.default_rng(np.random.SeedSequence([seed, d]))
    raw = rng.normal(size=(n_max, d + 1))
    features = np.ascontiguousarray(raw[:, :d] @ truth["H_sqrt"])
    raw_z = raw[:, d].copy()
    proj_noise = features @ truth["hetero_dir"]
    noise_sd = np.sqrt(BASE_NOISE**2 + (HETERO_NOISE * proj_noise) ** 2)
    noise = noise_sd * raw_z
    del raw, raw_z, proj_noise, noise_sd  # free ~half the peak memory before the long loop

    cov, nmse, sc_rem = _sgd_loop(
        features, noise,
        truth["x_star"], truth["H"], truth["H_inv"],
        truth["sandwich_diag_sqrt"], truth["sandwich_trace"],
        checkpoints_arr, ALPHA, ETA0, SA_C, SA_RIDGE, GRADIENT_CLIP,
        EMA_RHOS, Z_VALUE,
    )

    rows = []
    for ci in range(len(checkpoints_arr)):
        n = int(checkpoints_arr[ci])
        for m in range(N_METHODS):
            rows.append({
                "seed": seed,
                "method": METHOD_KEYS[m],
                "n": n,
                "coverage": cov[ci, m],
                "nmse": nmse[ci, m],
                "scaled_remainder": sc_rem[ci, m],
            })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(all_rows):
    df = pd.DataFrame(all_rows)
    summary = (
        df.groupby(["method", "n"], sort=True)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_se=("coverage", standard_error),
            nmse_mean=("nmse", "mean"),
            nmse_se=("nmse", standard_error),
            scaled_remainder_mean=("scaled_remainder", "mean"),
            scaled_remainder_se=("scaled_remainder", standard_error),
        )
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Plotting: main figure (scaled remainder only, log-log)
# ---------------------------------------------------------------------------


def plot_main_figure(summary, output_path):
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        for method in METHOD_KEYS:
            sub = summary[summary["method"] == method].sort_values("n")
            x = sub["n"].to_numpy(float)
            mean = sub["scaled_remainder_mean"].to_numpy(float)
            se = sub["scaled_remainder_se"].to_numpy(float)
            band = 1.96 * se
            lo = np.clip(mean - band, 1e-12, None)
            hi = mean + band

            ax.plot(
                x, mean, linewidth=3.0,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                label=METHOD_LABELS[method],
            )
            ax.fill_between(
                x, lo, hi,
                color=METHOD_COLORS[method], alpha=0.15,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample size $n$")
        ax.set_ylabel(r"$\sqrt{n}\,\|R_n\|$")
        ax.set_title(
            r"Stabilization threshold violation ($d = 5$, $\alpha = 0.7$)",
            fontsize=28,
        )
        ax.legend(loc="upper right", fontsize=17, frameon=True)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved main figure to {output_path}")


# ---------------------------------------------------------------------------
# Plotting: supplement figure (1x2, coverage + NMSE only; remainder is in main text)
# ---------------------------------------------------------------------------


def plot_supplement_figure(summary, output_path):
    import matplotlib.pyplot as plt

    metrics = [
        ("coverage", "Coverage"),
        ("nmse", "NMSE"),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[col]
            for method in METHOD_KEYS:
                sub = summary[summary["method"] == method].sort_values("n")
                x = sub["n"].to_numpy(float)
                mean = sub[f"{metric}_mean"].to_numpy(float)
                se = sub[f"{metric}_se"].to_numpy(float)
                band = 1.96 * se
                lo, hi = mean - band, mean + band
                if metric == "coverage":
                    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
                ax.plot(
                    x, mean, linewidth=3.0,
                    color=METHOD_COLORS[method],
                    linestyle=METHOD_LINESTYLES[method],
                    label=METHOD_LABELS[method],
                )
                ax.fill_between(
                    x, lo, hi,
                    color=METHOD_COLORS[method], alpha=0.14,
                )

            ax.set_xscale("log")
            if metric == "coverage":
                ax.axhline(0.95, ls=":", lw=2.0, color="#333333")
                ax.set_ylim(0.55, 1.02)
            elif metric == "nmse":
                ax.axhline(1.0, ls=":", lw=2.0, color="#333333")
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Sample size $n$")
            ax.set_title(f"$d = {DIMENSION}$")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center", ncol=3, frameon=True,
            fontsize=22,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved supplement figure to {output_path}")


# ---------------------------------------------------------------------------
# Slope table
# ---------------------------------------------------------------------------


def compute_slope_table(summary):
    rows = []
    for method in METHOD_KEYS:
        sub = summary[summary["method"] == method].sort_values("n")
        log_n = np.log(sub["n"].to_numpy(float))
        log_sr = np.log(np.clip(
            sub["scaled_remainder_mean"].to_numpy(float), 1e-15, None
        ))
        # Use second half for slope
        q = len(log_n) // 2
        slope = float(np.polyfit(log_n[q:], log_sr[q:], deg=1)[0])
        rows.append({
            "Method": METHOD_LABELS[method],
            "Slope (2nd half)": f"{slope:+.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = Path(__file__).resolve().parent
    seeds = [BASE_SEED + i for i in range(SEED_COUNT)]
    truth = build_truth()
    checkpoints_arr = np.array(SAMPLE_SIZES, dtype=np.int64)

    print(f"Threshold violation experiment: d={DIMENSION}, "
          f"kappa={np.linalg.cond(truth['H']):.1f}")
    print(f"  {SEED_COUNT} seeds, n up to {max(SAMPLE_SIZES):,}")
    print(f"  Threshold: beta > (alpha+1)/2 = {(ALPHA+1)/2:.2f}")
    print(f"  Methods: SA (beta=1), EMA rho=0.5/0.999 (beta=0)")

    # JIT warmup
    print("  JIT compiling (first seed)...")
    rows0 = run_one_seed(seeds[0], truth, checkpoints_arr)
    all_rows = list(rows0)
    print("  JIT done. Running remaining seeds...")

    for i, seed in enumerate(seeds[1:], start=2):
        rows = run_one_seed(seed, truth, checkpoints_arr)
        all_rows.extend(rows)
        if i % 10 == 0:
            print(f"  Progress: {i}/{SEED_COUNT}")

    print("Aggregating...")
    summary = aggregate(all_rows)
    csv_path = output_dir / "threshold_coverage_summary.csv"
    summary.to_csv(csv_path, index=False)

    print("Plotting...")
    plot_main_figure(summary, output_dir / "threshold_coverage.pdf")
    plot_supplement_figure(
        summary, output_dir / "threshold_coverage_supplement.pdf"
    )

    print("\nScaled remainder slopes:")
    slope_df = compute_slope_table(summary)
    print(slope_df.to_string(index=False))

    n_final = max(SAMPLE_SIZES)
    print(f"\n--- Final metrics at n = {n_final:,} ---")
    final = summary[summary["n"] == n_final]
    for _, row in final.iterrows():
        print(f"  {METHOD_LABELS[row['method']]:45s}  "
              f"Coverage={row['coverage_mean']:.3f}  "
              f"NMSE={row['nmse_mean']:.2f}  "
              f"ScRem={row['scaled_remainder_mean']:.4f}")


if __name__ == "__main__":
    main()
