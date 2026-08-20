#!/usr/bin/env bash
# Re-run the paper's experiments. NO CLUSTER REQUIRED -- the scripts parallelize
# with plain Python multiprocessing (concurrent.futures), so this runs anywhere;
# the cluster was used for speed, not for any dependency.
#
#   ./run_experiments.sh smoke    # laptop scale, minutes -- verifies the pipeline
#   ./run_experiments.sh full     # the paper's exact settings, ~100 core-hours
#
# The env vars below ARE the experiment configuration; prodA.sbatch is only a
# site-specific SLURM wrapper around the same variables.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-smoke}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}"
t0=$SECONDS
say() { printf '\n=== %s  (t+%ds) ===\n' "$1" "$((SECONDS-t0))"; }

if [ "$MODE" = "full" ]; then
  # ---- Fig. 1: SECOM self-calibration (identical to prodA.sbatch) ----------
  say "Fig. 1  SECOM anchor: 200 seeds x n=1e7  [~100 core-hours]"
  SECOM_MODE=anchor \
  SECOM_METHODS="identity,adagrad,rmsprop,ons" \
  SECOM_PLUGIN="identity,adagrad,rmsprop,ons" \
  SECOM_EPS=1.0 SECOM_K=20 SECOM_LAMBDA=0.1 \
  SECOM_SEEDS=200 SECOM_NMAX=10000000 SECOM_TAG=prod \
    python run_secom_redesign.py          # -> secom_redesign_anchor_d21_prod.csv

  # ---- Fig. 2 + Table 2: threshold violation ------------------------------
  # NOTE: TV_NMAX must be set. The script's default is 1e6; the paper's curve
  # runs to 5e7. Serial + numba-JIT, so this is the long single-core job.
  say "Fig. 2  threshold violation: 100 seeds x n=5e7  [serial, hours]"
  TV_SEEDS=100 TV_NMAX=50000000 TV_D=10 \
    python run_threshold_coverage.py      # -> threshold_coverage_summary.csv

  # ---- §6.5 text numbers: certified change detection ----------------------
  say "6.5  detection: in-control stationarity check"
  D_MODE=stat python run_secom_detect4.py
  say "6.5  detection: ARL0 / ARL1 table"
  D_MODE=arl  python run_secom_detect4.py # -> secom_detect4_summary.csv
else
  # Smoke scale: same code path, same seeds (smoke seeds are a PREFIX of the
  # production seeds), fewer of them and a shorter stream. Seed counts are tuned
  # to the smallest values that still reproduce the paper's QUALITATIVE result --
  # see README section 5b for the measured smoke-vs-production comparison.
  # Total: ~50 s on a 10-core laptop.
  say "SMOKE  Fig. 1  SECOM anchor: 8 seeds x n=2e5"
  SECOM_MODE=anchor \
  SECOM_METHODS="identity,adagrad,rmsprop,ons" \
  SECOM_PLUGIN="identity,adagrad,rmsprop,ons" \
  SECOM_EPS=1.0 SECOM_K=20 SECOM_LAMBDA=0.1 \
  SECOM_SEEDS=8 SECOM_NMAX=200000 SECOM_TAG=smoke \
    python run_secom_redesign.py          # -> secom_redesign_anchor_d21_smoke.csv

  # 25 seeds, not 5: coverage is a per-seed count over d=10 coordinates, so few
  # seeds give a badly noisy estimate that hides the certified-vs-EMA ordering.
  say "SMOKE  Fig. 2  threshold violation: 25 seeds x n=2e5"
  TV_SEEDS=25 TV_NMAX=200000 TV_D=10 \
    python run_threshold_coverage.py      # -> threshold_coverage_summary.csv (OVERWRITES)

  # REPS=10 x ICLEN=1e6 -> 5000 windows. The script's FLAT/PASS drift test needs
  # >~1500 windows; fewer makes it report a spurious DRIFT/FAIL from pure MC noise.
  say "SMOKE  6.5  detection stationarity check"
  D_MODE=stat D_REPS=10 D_ICLEN=1000000 python run_secom_detect4.py
fi

say "done"
echo "Note: smoke mode writes threshold_coverage_summary.csv (plus two scratch PDFs)"
echo "into this directory; all are gitignored. The paper's canonical CSVs live in"
echo "results/ and are never written to -- make_figures.sh always reads from there."
