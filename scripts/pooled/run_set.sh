#!/bin/bash
# Run the pooled online ONE-NAS primary configuration on one prepared panel.
# Usage: run_set.sh <panel_dir> <num_training_windows> <total_generation> <output_dir> [np] [seed]
set -euo pipefail
PANEL_DIR=$1; NTW=$2; TOTGEN=$3; OUTDIR=$4; NP=${5:-6}; SEED=${6:-42}
BUILD_DIR="$(cd "$(dirname "$0")/../.." && pwd)/build"
mkdir -p "$OUTDIR"
FILES=$(ls "$PANEL_DIR"/*.csv | grep -v panel_)
exec mpirun -np "$NP" "$BUILD_DIR/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER \
  --output_parameter_names RET \
  --number_islands 8 --bp_iterations 10 --num_mutations 1 \
  --time_series_length 40 --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets 5 --num_training_sets 200 \
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8 \
  --online_series_seed "$SEED" --rounds_per_generation ${ROUNDS:-1} \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUTDIR"
