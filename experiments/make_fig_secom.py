#!/usr/bin/env python
"""Render the SECOM self-calibration figure (three panels) from the production
CSV secom_redesign_anchor_d21_prod.csv, writing fig_secom.pdf into the repo root
(make_figures.sh then moves it into figs/). Panels:
  (a) 95% Wald coverage vs n: plug-in Vhat (solid, markers) overlaid with the
      oracle V (dashed) for the four certified optimizers -> self-calibration
      (plug-in tracks oracle) and preconditioner-independence (arms coincide);
  (b) normalized MSE vs n -> 1;
  (c) online covariance error ||Vhat_n - Sigma||_F/||Sigma||_F vs n -> 0, the
      direct evidence that the streaming covariance estimate is consistent.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, os, tempfile
# Matplotlib config dir. The original hard-coded a cluster scratch path here;
# this is a config location only -- no effect on any number.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7,
                     "mathtext.fontset": "dejavuserif"})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
df = pd.read_csv(HERE / "secom_redesign_anchor_d21_prod.csv")

styles = [("identity", "Identity (PR-SGD)", "#000000", "o"),
          ("adagrad", "SA-AdaGrad", "#1f77b4", "s"),
          ("rmsprop", "SA-RMSProp", "#d62728", "^"),
          ("ons", "SA-ONS", "#2ca02c", "D")]

fig, (axc, axm, axv) = plt.subplots(1, 3, figsize=(7.2, 2.5))
for m, lab, c, mk in styles:
    d = df[df.method == m].sort_values("n"); n = d.n.values
    # (a) plug-in coverage (solid + markers + band) and oracle coverage (dashed)
    axc.plot(n, d.cov_plugin_mean, color=c, marker=mk, ls="-", ms=3.2, lw=1.1, label=lab)
    axc.fill_between(n, d.cov_plugin_mean - 1.96 * d.cov_plugin_se,
                     d.cov_plugin_mean + 1.96 * d.cov_plugin_se, color=c, alpha=0.12, lw=0)
    axc.plot(n, d.cov_oracle_mean, color=c, ls="--", lw=0.9, alpha=0.75)
    # (b) NMSE
    axm.plot(n, d.nmse_mean, color=c, marker=mk, ls="-", ms=3.2, lw=1.1)
    axm.fill_between(n, d.nmse_mean - 1.96 * d.nmse_se,
                     d.nmse_mean + 1.96 * d.nmse_se, color=c, alpha=0.12, lw=0)
    # (c) online covariance error
    axv.plot(n, d.vhat_vs_sigma_mean, color=c, marker=mk, ls="-", ms=3.2, lw=1.1)

axc.axhline(0.95, color="gray", ls=":", lw=0.8)
axc.set_xscale("log"); axc.set_xlabel(r"sample size $n$"); axc.set_ylabel("95\\% CI coverage")
axc.set_title("(a) Self-calibrated coverage", fontsize=9); axc.set_ylim(0.70, 1.0)
axc.text(0.04, 0.90, "nominal 0.95", transform=axc.transAxes, fontsize=7, color="gray")
axc.text(0.96, 0.10, r"solid: plug-in $\hat V_n$" + "\n" + r"dashed: oracle $V$",
         transform=axc.transAxes, fontsize=6.4, ha="right", va="bottom", color="0.25")

axm.axhline(1.0, color="gray", ls=":", lw=0.8)
axm.set_xscale("log"); axm.set_yscale("log"); axm.set_xlabel(r"sample size $n$")
axm.set_ylabel(r"NMSE $=n\|\bar x_n-x^*\|^2/\mathrm{Tr}\,V$", fontsize=8)
axm.set_title("(b) Normalized MSE", fontsize=9); axm.set_ylim(0.8, 12)

axv.set_xscale("log"); axv.set_yscale("log"); axv.set_xlabel(r"sample size $n$")
axv.set_ylabel(r"$\|\hat V_n-V\|_F/\|V\|_F$", fontsize=8.5)
axv.set_title("(c) Online covariance error", fontsize=9); axv.set_ylim(1.5e-3, 3)

axc.legend(fontsize=6.4, frameon=False, loc="upper left", bbox_to_anchor=(0.30, 0.62))
for ax in (axc, axm, axv):
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
fig.tight_layout(pad=0.4)
out = ROOT / "fig_secom.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)

fin = df[df.n == df.n.max()].set_index("method")
print("final n =", int(df.n.max()))
for m in ["identity", "adagrad", "rmsprop", "ons"]:
    r = fin.loc[m]
    print(f"  {m:9s} plug-in cov={r.cov_plugin_mean:.3f} oracle cov={r.cov_oracle_mean:.3f} "
          f"nmse={r.nmse_mean:.3f} Vhat_err={r.vhat_vs_sigma_mean:.4f}")
