#!/bin/bash -l
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 04:00:00
#SBATCH -J ctlprod
#SBATCH -o /anvil/scratch/x-jchang5/wwtp/v3/control/prod_%a.out
#SBATCH -e /anvil/scratch/x-jchang5/wwtp/v3/control/prod_%a.err
#
# THE FIXED-ARCHITECTURE CONTROL ARM, production scale.  DO NOT LAUNCH WITHOUT
# APPROVAL.  Submit as an array over seeds, one seed per array task:
#
#   sbatch --array=41-45 ~/wwtp_scripts/v3/control/prod_control.sbatch
#
# Env overrides:  DATA (prep dir), ARCH (seed genome basename), H, L, T, V.
# Defaults point at v2's H=72 prep; repoint DATA at the v3 prep when it exists.
#
# Wall-clock: the control's per-generation cost is FLAT (the architecture never
# grows), measured at ~3.1 s/generation for lstm4 on 16 ranks.  610 generations
# is ~32 min wall / ~8.5 core-h per seed.  The 4 h limit is ~7x headroom.
set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID:?submit this as an array; the task id is the seed}
PY=/apps/anvil/external/apps/conda/2024.02/bin/python3
CTL=$HOME/wwtp_scripts/v3/control
ROOT=/anvil/scratch/x-jchang5/wwtp/v3/control

export DATA=${DATA:-/anvil/scratch/x-jchang5/wwtp/v2/h72_L144}
export ARCH=${ARCH:-lstm4}
export SEED_BIN=$ROOT/seeds/$ARCH.bin
export OUT=$ROOT/prod_${ARCH}_s${SEED}/run
export L=${L:-144} T=${T:-80} V=${V:-5} H=${H:-72}
export ISL=16 POP=5 ELITE=1 BP=10 NP=16
export SEED
# GEN unset => run to max_generation from the prep manifest.

mkdir -p "$(dirname "$OUT")"
bash "$CTL/run_control.sh"

# --- the 16-champion ensemble, then score both it and the single global best ---
ENS=$ROOT/prod_${ARCH}_s${SEED}/ensemble
"$PY" "$CTL/ensemble_control.py" "$OUT" "$ENS" --combiner "${COMBINER:-mean}" --members champions

SCORE=${SCORE:-$HOME/wwtp_scripts/v3/score_v3.py}
[ -f "$SCORE" ] || SCORE=$HOME/wwtp_scripts/v2/score_v2.py

echo "################ CONTROL: single global best ################"
"$PY" "$SCORE" "$OUT"  "$DATA/baselines" || true
echo "################ CONTROL: 16-champion ensemble #############"
"$PY" "$SCORE" "$ENS"  "$DATA/baselines" || true
echo "################ matching verification #####################"
"$PY" "$CTL/verify_control.py" "$OUT" || true
