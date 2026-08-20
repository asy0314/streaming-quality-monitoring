#!/bin/bash
#SBATCH -J secom_prodA
#SBATCH -A <your-allocation>
#SBATCH -q inferno
#SBATCH -p cpu-gnr
#SBATCH -C graniterapids
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH -o secom_prodA_%j.out
#SBATCH -e secom_prodA_%j.err

# PRODUCTION — Experiment A (online self-calibration), the real-data headline.
# 4 certified arms + online plug-in sandwich, d=21, eps=1 (paper's case-study
# setting), 200 seeds, n=1e7. -C graniterapids forces PACE onto a cpu-gnr node.
# 100 workers => 200 seeds run as 2 balanced waves on fast GNR cores (~1h).
set -u
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/z00_lmod.sh
module load anaconda3/2023.03
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$(readlink -f "$0")")"   # the experiments/ directory
echo "host=$(hostname) part=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"

export SECOM_MODE=anchor
export SECOM_METHODS="identity,adagrad,rmsprop,ons"
export SECOM_PLUGIN="identity,adagrad,rmsprop,ons"
export SECOM_EPS=1.0 SECOM_K=20 SECOM_LAMBDA=0.1
export SECOM_SEEDS=200 SECOM_NMAX=10000000 SECOM_WORKERS=100 SECOM_TAG=prod
python run_secom_redesign.py

echo "end=$(date -Is)"
ls -la secom_redesign_anchor_d21_prod.csv
