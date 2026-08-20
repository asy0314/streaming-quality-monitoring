#!/usr/bin/env bash
# Regenerate all three paper figures from the committed CSVs in results/.
# Takes seconds; no cluster run required.
#
# The plot scripts are kept byte-identical to the versions that produced the
# paper: each reads its CSV from its OWN directory and writes its PDF to the
# PARENT directory. This wrapper stages the CSVs in and moves the PDFs out.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(mktemp -d)}"

cp results/secom_redesign_anchor_d21_prod.csv results/threshold_coverage_summary.csv .
trap 'rm -f secom_redesign_anchor_d21_prod.csv threshold_coverage_summary.csv' EXIT

python make_fig_secom.py        # -> ../fig_secom.pdf
python make_fig_threshold.py    # -> ../fig_threshold.pdf
python make_fig_detect.py       # -> ../fig_detect.pdf   (re-simulates from secom_data/)

mkdir -p ../figs
mv ../fig_secom.pdf ../fig_threshold.pdf ../fig_detect.pdf ../figs/
echo "figures written to figs/"
