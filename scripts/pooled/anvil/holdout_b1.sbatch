#!/bin/bash -l
# B1: the one-shot confirmatory holdout campaign (PRIMARY_HOLDOUT.md).
#
# The frozen PRIMARY.md ac58a56 primary, unchanged, with the weekly generation
# clock started ~5 years earlier so the scored stream covers 2015-01-02 ..
# 2024-12-31 (warmup/burn-in through 2014).  The 2015-2019 segment is the
# pre-registered one-look holdout; 2020-2024 comes out of the same runs as a
# fresh replication segment.  NOTHING here may change without a signed
# amendment: no scoring-side improvement, no selection change, no new flags.
#
# DO NOT SUBMIT before (a) the holdout amendment is signed off and (b) the
# Aug 19 pilot-config freeze is committed (see PRIMARY_HOLDOUT.md).
#
#   for SEED in 42 43 44; do
#     sbatch --array=1-4 --export=ALL,SEED=$SEED holdout_b1.sbatch
#   done
#
# NTW derivation (verified against the 2020 values 712/695/702/738):
#   NTW = (row(first scored date) - (L - s)) / s - V - 50,  L=40 s=5 V=5
#   first scored date 2015-01-02 -> set1 460, set2 443, set3 450, set4 486
#   generations to cover 2015-01-02 .. 2024-12-31: 503 for every set
#SBATCH -J onenas_hold15
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:30:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

ONENAS="$HOME/ONENAS"
DATA="/anvil/scratch/x-jchang5/panels_core7"
SEED="${SEED:-42}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

case "$SET" in
  set1) NTW=460 ;;
  set2) NTW=443 ;;
  set3) NTW=450 ;;
  set4) NTW=486 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac
TOTGEN=503

OUT="/anvil/scratch/x-jchang5/results_v2/C7_E_holdout15/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_core7/"*.csv | grep -v panel_)

echo "B1 holdout: frozen ac58a56 primary, $SET seed=$SEED NTW=$NTW -> $OUT"
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
