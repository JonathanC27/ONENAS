#!/bin/bash -l
# Exploratory start-date run: the frozen PRIMARY.md primary (PER, NTS=2000,
# core7) with the weekly generation clock set so live trading begins at the
# first 2022 trading day (2022-01-03) and runs through 2024. The training
# pool enters with the full 2000-2021 history (NTW 813/796/803/838 vs 712/...
# for the 2020 start). Purpose: compare a FRESH 2022 start against the
# 2022-24 slice of the 2020-started runs (does accumulated evolutionary
# state matter?). 2022-24 lies inside the disclosed development span;
# descriptive/exploratory only, no pre-registered endpoint.
#
#   for SEED in 42 43 44; do
#     sbatch --array=1-4 --export=ALL,SEED=$SEED start22.sbatch
#   done
#SBATCH -J onenas_s22
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
DATA="/anvil/scratch/x-jchang5/panels_core7"
SEED="${SEED:-42}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

case "$SET" in
  set1) NTW=813 ;;
  set2) NTW=796 ;;
  set3) NTW=803 ;;
  set4) NTW=838 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac
TOTGEN=150

OUT="/anvil/scratch/x-jchang5/results_v2/C7_E_start22/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_core7/"*.csv | grep -v panel_)

echo "start22: frozen primary, $SET seed=$SEED NTW=$NTW -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21 \
  --output_parameter_names RET_CS \
  --number_islands 8 --bp_iterations 10 --num_mutations 1 \
  --time_series_length 40 --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets 5 --num_training_sets 2000 \
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation 1 \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --selection_metric mse \
  --max_pred_sd_ratio 3.0 \
  --write_elite_predictions \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUT"
