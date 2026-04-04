#!/bin/bash -l
# Catastrophic Forgetting CONTROL: ONE-NAS + SlidingWindow (No Replay)
#
# Uses SLURM job arrays to run 5 trials in parallel.
# Submit with: sbatch scripts/experiments/catastrophic_forgetting_control.sh

#SBATCH -J cf_onenas_sliding
#SBATCH -A cis251123
#SBATCH -o cf_onenas_sliding_%A_%a.output
#SBATCH -e cf_onenas_sliding_%A_%a.error
#SBATCH --mail-user=jchang1@ucvts.org
#SBATCH --mail-type=ALL
#SBATCH -t 4:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=22
#SBATCH -p wholenode
#SBATCH --array=0-4

set -euo pipefail

module --force purge
module load gcc
module load cmake
module load openmpi
module load libtiff

ONENAS="$HOME/ONENAS"
DATA_DIR="$ONENAS/datasets/2020_wind_engine"
TRIAL=${SLURM_ARRAY_TASK_ID}

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

TRAINING_FILES_A="\
${DATA_DIR}/turbine_R80711_2017-2020_5.csv \
${DATA_DIR}/turbine_R80711_2017-2020_6.csv \
${DATA_DIR}/turbine_R80711_2017-2020_7.csv \
${DATA_DIR}/turbine_R80711_2017-2020_8.csv \
${DATA_DIR}/turbine_R80711_2017-2020_9.csv \
${DATA_DIR}/turbine_R80711_2017-2020_10.csv \
${DATA_DIR}/turbine_R80711_2017-2020_11.csv \
${DATA_DIR}/turbine_R80711_2017-2020_12.csv \
${DATA_DIR}/turbine_R80711_2017-2020_13.csv \
${DATA_DIR}/turbine_R80711_2017-2020_14.csv \
${DATA_DIR}/turbine_R80711_2017-2020_15.csv \
${DATA_DIR}/turbine_R80711_2017-2020_16.csv \
${DATA_DIR}/turbine_R80711_2017-2020_17.csv \
${DATA_DIR}/turbine_R80711_2017-2020_18.csv"

TRAINING_FILES_B="\
${DATA_DIR}/turbine_R80711_2017-2020_1.csv \
${DATA_DIR}/turbine_R80711_2017-2020_2.csv \
${DATA_DIR}/turbine_R80711_2017-2020_25.csv \
${DATA_DIR}/turbine_R80711_2017-2020_26.csv \
${DATA_DIR}/turbine_R80711_2017-2020_27.csv \
${DATA_DIR}/turbine_R80711_2017-2020_28.csv \
${DATA_DIR}/turbine_R80711_2017-2020_29.csv \
${DATA_DIR}/turbine_R80711_2017-2020_31.csv"

EVAL_FILES_A="${DATA_DIR}/turbine_R80711_2017-2020_19.csv"
EVAL_FILES_B="${DATA_DIR}/turbine_R80711_2017-2020_30.csv"

TOTAL_GENERATIONS=1250
EVAL_FREQUENCY=5

exp_name="$ONENAS/results/catastrophic_forgetting_control/$TRIAL"
mkdir -p "$exp_name"
echo "Running Catastrophic Forgetting CONTROL (trial $TRIAL) - SlidingWindow (no replay)"
echo "Results will be saved to: $exp_name"

mpirun -np $SLURM_NTASKS "$ONENAS/build/mpi/catastrophic_forgetting_mpi" \
--training_filenames_a $TRAINING_FILES_A \
--training_filenames_b $TRAINING_FILES_B \
--eval_filenames_a $EVAL_FILES_A \
--eval_filenames_b $EVAL_FILES_B \
--total_generations $TOTAL_GENERATIONS \
--eval_frequency $EVAL_FREQUENCY \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 20 \
--bp_iterations 10 \
--output_directory "$exp_name" \
--num_mutations 1 \
--time_series_length 25 \
--num_validation_sets 100 \
--num_training_sets 100 \
--get_train_data_by SlidingWindow \
--speciation_method onenas \
--repopulation_frequency 200 \
--generated_population_size 10 \
--elite_population_size 5 \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--normalize min_max \
--compare_with_naive \
--control_size_method reduce_add_mutation \
--std_message_level INFO \
--file_message_level INFO

echo "Trial $TRIAL complete. Check $exp_name/forgetting_results.csv for results."
