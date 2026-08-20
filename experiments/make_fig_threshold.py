#!/usr/bin/env python3
"""Render the threshold-violation / constant-EMA coverage-breakdown figure in the
clean two-panel style matching the other paper figures. Reads
threshold_coverage_summary.csv, writes fig_threshold.pdf into the repo root."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7,
                     "mathtext.fontset": "dejavuserif"})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
df = pd.read_csv(HERE / "threshold_coverage_summary.csv")

styles = [("sa_rmsprop", r"SA-RMSProp ($\rho_t=1/t$)", "#2ca02c", "o", "-"),
          ("ema_05",     r"Const-EMA ($\rho=0.5$)",     "#ff7f0e", "s", "--"),
          ("ema_099",    r"Const-EMA ($\rho=0.999$)",   "#9467bd", "D", "-.")]

fig, (axc, axr) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for m, lab, c, mk, ls in styles:
    d = df[df.method == m].sort_values("n"); n = d.n.values
    axc.plot(n, d.coverage_mean, color=c, marker=mk, ls=ls, ms=3.0, lw=1.1,
             markevery=8, label=lab)
    axc.fill_between(n, d.coverage_mean - 1.96*d.coverage_se,
                     d.coverage_mean + 1.96*d.coverage_se, color=c, alpha=0.12, lw=0)
    axr.plot(n, d.scaled_remainder_mean, color=c, marker=mk, ls=ls, ms=3.0, lw=1.1,
             markevery=8, label=lab)
    axr.fill_between(n, d.scaled_remainder_mean - 1.96*d.scaled_remainder_se,
                     d.scaled_remainder_mean + 1.96*d.scaled_remainder_se,
                     color=c, alpha=0.12, lw=0)

axc.axhline(0.95, color="gray", ls="--", lw=0.8)
axc.set_xscale("log"); axc.set_xlabel(r"sample size $n$"); axc.set_ylabel("95% CI coverage")
axc.set_title("(a) Marginal coverage", fontsize=9); axc.set_ylim(0.45, 1.0)
axc.text(0.03, 0.90, "nominal 0.95", transform=axc.transAxes, fontsize=7, color="gray")

axr.set_xscale("log"); axr.set_yscale("log")
axr.set_xlabel(r"sample size $n$"); axr.set_ylabel(r"scaled remainder $\sqrt{n}\,\|R_n\|$", fontsize=8)
axr.set_title("(b) Dynamic remainder", fontsize=9)

axc.legend(fontsize=6.8, frameon=False, loc="lower right")
for ax in (axc, axr):
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
fig.tight_layout(pad=0.4)
out = ROOT / "fig_threshold.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
fin = df[df.n == df.n.max()].set_index("method")
for m in ["sa_rmsprop", "ema_05", "ema_099"]:
    print(f"  {m:11s} cov={fin.loc[m,'coverage_mean']:.3f} screm={fin.loc[m,'scaled_remainder_mean']:.2f}")
