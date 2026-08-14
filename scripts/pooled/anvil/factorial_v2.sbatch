#!/bin/bash -l
# Two-factor decomposition on the v2 panels: does the gain come from the
# cross-sectional TARGET, from IC-based SELECTION, or both?
#   sbatch --array=1-4 --export=ALL,TARGET=RET,SELECT=mse       factorial_v2.sbatch   # A: data fixes only
#   sbatch --array=1-4 --export=ALL,TARGET=RET_CS,SELECT=mse    factorial_v2.sbatch   # B: + CS target
#   sbatch --array=1-4 --export=ALL,TARGET=RET_CS,SELECT=ic_gated factorial_v2.sbatch # C: + IC selection
#SBATCH -J onenas_fact
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 41
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
DATA="/anvil/scratch/x-jchang5/panels_v2"
SEED="${SEED:-42}"
TARGET="${TARGET:-RET_CS}"
SELECT="${SELECT:-ic_gated}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

OUT="/anvil/scratch/x-jchang5/results_v2/${TARGET}_${SELECT}/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_v2/"*.csv | grep -v panel_)

echo "arm target=$TARGET selection=$SELECT $SET seed=$SEED -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER \
  --output_parameter_names "$TARGET" \
  --number_islands 8 --bp_iterations 10 --num_mutations 1 \
  --time_series_length 40 --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets 5 --num_training_sets 200 \
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation 1 \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --selection_metric "$SELECT" --ic_ewma_halflife 8 --ic_gate_factor 1.5 \
  --max_pred_sd_ratio 3.0 \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUT"
