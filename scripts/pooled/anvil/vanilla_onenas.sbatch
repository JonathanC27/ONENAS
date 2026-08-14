#!/bin/bash -l
# Vanilla ONE-NAS (published mechanism: Uniform sampling, MSE selection) on the
# pooled stock panels — ACCESS Anvil version.
#
# This is the internal-control arm: identical structure to the primary run
# (pooled panel, weekly sliding clock, same islands/populations) with ONLY the
# sampler switched to Uniform. Submit as an array over sets 1-4:
#   sbatch --array=1-4 vanilla_onenas.sbatch
# Seeds: override with --export=SEED=43 (default 42).
#
# BEFORE FIRST USE, fill in:
#   -A : your ACCESS allocation (e.g. cis2XXXXX — `myallocations` shows it)
#   ONENAS : path to your cloned fork on Anvil ($HOME or $PROJECT space)
#   DATA : path to the uploaded *_clean panel directories

#SBATCH -J onenas_vanilla
#SBATCH -A YOUR_ALLOCATION_HERE
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH -o %x_set%a_%j.out
#SBATCH -e %x_set%a_%j.err
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL_HERE

module purge
module load gcc openmpi

ONENAS="$HOME/ONENAS"                    # your fork, built on Anvil
DATA="$HOME/panels"                      # contains set1_clean .. set4_clean
SEED="${SEED:-42}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"

# per-set burn-in and generation counts (computed from each panel's common start)
case "$SET" in
  set1) NTW=712; TOTGEN=301 ;;
  set2) NTW=695; TOTGEN=300 ;;
  set3) NTW=702; TOTGEN=300 ;;
  set4) NTW=738; TOTGEN=300 ;;
  *) echo "unknown set $SET"; exit 1 ;;
esac

OUT="$ONENAS/results/vanilla_uniform/${SET}_seed${SEED}"
mkdir -p "$OUT"
FILES=$(ls "$DATA/${SET}_clean/"*.csv | grep -v panel_)

echo "vanilla ONE-NAS (Uniform) on $SET, seed $SEED -> $OUT"
time srun --mpi=pmi2 "$ONENAS/build/mpi/onenas_mpi" \
  --training_filenames $FILES \
  --pooled_panel --time_offset 1 \
  --input_parameter_names RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER \
  --output_parameter_names RET \
  --number_islands 8 --bp_iterations 10 --num_mutations 1 \
  --time_series_length 40 --window_step 5 \
  --num_training_windows "$NTW" --num_validation_sets 5 --num_training_sets 200 \
  --get_train_data_by Uniform \
  --online_series_seed "$SEED" --rounds_per_generation 1 \
  --speciation_method onenas --repopulation_frequency 50 \
  --generated_population_size 5 --elite_population_size 8 \
  --total_generation "$TOTGEN" \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --normalize none --compare_with_naive --control_size_method none \
  --std_message_level ERROR --file_message_level ERROR \
  --output_directory "$OUT"
