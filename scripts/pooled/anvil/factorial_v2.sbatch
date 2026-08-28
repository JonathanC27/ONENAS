#!/bin/bash -l
# Two-factor decomposition on the v2 panels: does the gain come from the
# cross-sectional TARGET, from IC-based SELECTION, or both?
#   sbatch --array=1-4 --export=ALL,TARGET=RET,SELECT=mse       factorial_v2.sbatch   # A: data fixes only
#   sbatch --array=1-4 --export=ALL,TARGET=RET_CS,SELECT=mse    factorial_v2.sbatch   # B: + CS target
#   sbatch --array=1-4 --export=ALL,TARGET=RET_CS,SELECT=ic_gated factorial_v2.sbatch # C: + IC selection
#SBATCH -J onenas_fact
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 01:15:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
PANELS="${PANELS:-v2}"   # v2 | core7 — selects panels_<PANELS>/set<N>_<PANELS>
DATA="/anvil/scratch/x-jchang5/panels_${PANELS}"
SEED="${SEED:-42}"
TARGET="${TARGET:-RET_CS}"
SELECT="${SELECT:-ic_gated}"
# default inputs follow the panel tree: core7 panels carry the core7 features
if [ "$PANELS" = "core7" ]; then
  INPUTS="${INPUTS:-RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21}"
else
  INPUTS="${INPUTS:-RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER}"
fi
ARM="${ARM:-${TARGET}_${SELECT}}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
NTS="${NTS:-200}"
TSL="${TSL:-40}"
NVS="${NVS:-5}"
NODE_TYPES="${NODE_TYPES:-simple UGRNN MGU GRU delta LSTM}"
# population / training knobs (defaults = frozen PRIMARY.md values)
ELITE="${ELITE:-8}"
GENPOP="${GENPOP:-5}"
REPOP="${REPOP:-50}"
PER_ALPHA="${PER_ALPHA:-0.6}"
PER_LAMBDA="${PER_LAMBDA:-0.007}"
BP="${BP:-10}"
ROUNDS="${ROUNDS:-1}"
# SPAN=eval  -> scored span 2020-01-01..2024-12-31 (default, unchanged)
# SPAN=tune16 -> scored span 2016-01-01..2019-12-31 (Amendment 6 clock:
#                NTW 511/493/501/536 for sets 1-4, 201 generations)
SPAN="${SPAN:-eval}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

if [ "$SPAN" = "tune16" ]; then
  if [ "$TSL" != "40" ]; then echo "tune16 clock is defined for TSL=40 only"; exit 1; fi
  case "$SET" in
    set1) NTW=511; TOTGEN=201 ;;
    set2) NTW=493; TOTGEN=201 ;;
    set3) NTW=501; TOTGEN=201 ;;
    set4) NTW=536; TOTGEN=201 ;;
    *) echo "unknown set $SET"; exit 1 ;;
  esac
elif [ "$TSL" = "15" ]; then
  # L=15, step 5, V=2: NTW = (row(2020-01-02) - (L-s))/s - V - 50
  case "$SET" in
    set1) NTW=720; TOTGEN=301 ;;
    set2) NTW=703; TOTGEN=300 ;;
    set3) NTW=710; TOTGEN=300 ;;
    set4) NTW=746; TOTGEN=300 ;;
    *) echo "unknown set $SET"; exit 1 ;;
  esac
else
  case "$SET" in
    set1) NTW=712; TOTGEN=301 ;;
    set2) NTW=695; TOTGEN=300 ;;
    set3) NTW=702; TOTGEN=300 ;;
    set4) NTW=738; TOTGEN=300 ;;
    *) echo "unknown set $SET"; exit 1 ;;
  esac
fi

OUT="/anvil/scratch/x-jchang5/results_v2/${ARM}/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_${PANELS}/"*.csv | grep -v panel_)

echo "arm target=$TARGET selection=$SELECT $SET seed=$SEED -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names $INPUTS \
  --output_parameter_names "$TARGET" \
  --number_islands 8 --bp_iterations "$BP" --num_mutations 1 \
  --time_series_length "$TSL" --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets "$NVS" --num_training_sets $NTS \
  --get_train_data_by PER --per_alpha "$PER_ALPHA" --per_lambda "$PER_LAMBDA" --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation "$ROUNDS" \
  --speciation_method onenas --repopulation_frequency "$REPOP" \
  --generated_population_size "$GENPOP" --elite_population_size "$ELITE" \
  --total_generation "$TOTGEN" \
  --selection_metric "$SELECT" --ic_ewma_halflife 8 --ic_gate_factor 1.5 \
  --max_pred_sd_ratio 3.0 \
  --possible_node_types $NODE_TYPES \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  $EXTRA_FLAGS \
  --output_directory "$OUT"
