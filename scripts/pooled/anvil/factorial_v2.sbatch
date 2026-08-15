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
#SBATCH -t 00:25:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
PANELS="${PANELS:-v2}"   # v2 | core7 — selects panels_<PANELS>/set<N>_<PANELS>
DATA="/anvil/scratch/x-jchang5/panels_${PANELS}"
SEED="${SEED:-42}"
TARGET="${TARGET:-RET_CS}"
SELECT="${SELECT:-ic_gated}"
INPUTS="${INPUTS:-RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER}"
ARM="${ARM:-${TARGET}_${SELECT}}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
NTS="${NTS:-200}"
TSL="${TSL:-40}"
NVS="${NVS:-5}"
NODE_TYPES="${NODE_TYPES:-simple UGRNN MGU GRU delta LSTM}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

if [ "$TSL" = "15" ]; then
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
  --number_islands 8 --bp_iterations 10 --num_mutations 1 \
  --time_series_length "$TSL" --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets "$NVS" --num_training_sets $NTS \
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation 1 \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --selection_metric "$SELECT" --ic_ewma_halflife 8 --ic_gate_factor 1.5 \
  --max_pred_sd_ratio 3.0 \
  --possible_node_types $NODE_TYPES \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  $EXTRA_FLAGS \
  --output_directory "$OUT"
