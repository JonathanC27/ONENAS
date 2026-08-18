#!/bin/bash -l
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH -J prod72
#SBATCH -o /anvil/scratch/x-jchang5/wwtp/v2/prod72_%a_%j.out
#SBATCH -e /anvil/scratch/x-jchang5/wwtp/v2/prod72_%a_%j.err
#
# PRODUCTION ARM, H=72 (6 h) -- the pre-registered primary configuration of
# PROTOCOL.md.  Submit as an array over seeds:  sbatch --array=41-45 prod72.sbatch
#
# DO NOT EDIT the parameters below without editing PROTOCOL.md first and saying
# so.  They are pre-registered; changing them after seeing a result is exactly
# the failure mode this campaign is recovering from.
set -euo pipefail
S=${SLURM_ARRAY_TASK_ID:-41}
export DATA=/anvil/scratch/x-jchang5/wwtp/v2/h72_L144
export OUT=/anvil/scratch/x-jchang5/wwtp/v2/prod72_s$S/run
export L=144 T=80 V=5 H=72 ISL=16 POP=5 ELITE=8 BP=10 NP=16 SEED=$S
# GEN unset => run to max_generation (610 for this prep)
bash "$HOME/wwtp_scripts/v2/run_v2.sh"

/apps/anvil/external/apps/conda/2024.02/bin/python3 \
  "$HOME/wwtp_scripts/v2/score_v2.py" "$OUT" "$DATA/baselines"
