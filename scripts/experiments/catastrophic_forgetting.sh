#!/bin/sh
# Catastrophic Forgetting Experiment: Train A -> Train B -> Evaluate A
#
# This script measures whether ONE-NAS suffers from catastrophic forgetting
# when the data distribution shifts. It trains on Distribution A, then
# continues the same evolutionary process on Distribution B, then evaluates
# the resulting best model back on A.
#
# Usage:
#   cd build && cmake .. && make catastrophic_forgetting_mpi
#   cd .. && bash scripts/experiments/catastrophic_forgetting.sh
#
# You must set TRAINING_FILES_A, TRAINING_FILES_B, and optionally EVAL_FILES_A
# below to point to your data files.

cd build

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

# Distribution A: e.g., first half of wind turbine data (turbines 1-15)
TRAINING_FILES_A="../datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-5].csv"

# Distribution B: e.g., second half of wind turbine data (turbines 16-31)
TRAINING_FILES_B="../datasets/2020_wind_engine/turbine_R80711_2017-2020_1[6-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv"

# Eval files for A (can be same as training or a held-out set)
EVAL_FILES_A="$TRAINING_FILES_A"

NUM_PROCS=22
GENERATIONS_A=50
GENERATIONS_B=50
EVAL_FREQUENCY=5

for i in {0..4}
do

exp_name="../results/catastrophic_forgetting/$i"
mkdir -p $exp_name
echo "Running Catastrophic Forgetting Experiment (trial $i)"
echo "Results will be saved to: $exp_name"

mpirun -np $NUM_PROCS ./mpi/catastrophic_forgetting_mpi \
--training_filenames_a $TRAINING_FILES_A \
--training_filenames_b $TRAINING_FILES_B \
--eval_filenames_a $EVAL_FILES_A \
--generations_a $GENERATIONS_A \
--generations_b $GENERATIONS_B \
--eval_frequency $EVAL_FREQUENCY \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 20 \
--bp_iterations 10 \
--output_directory $exp_name \
--num_mutations 1 \
--time_series_length 40 \
--num_validation_sets 25 \
--num_training_sets 300 \
--get_train_data_by PER \
--speciation_method onenas \
--repopulation_frequency 50 \
--generated_population_size 10 \
--elite_population_size 10 \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--normalize min_max \
--compare_with_naive \
--control_size_method reduce_add_mutation \
--std_message_level INFO \
--file_message_level INFO \
--per_alpha 0.6 \
--per_lambda 0.01 \
--per_epsilon 1e-8

echo "Trial $i complete. Check $exp_name/forgetting_results.csv for results."

done

echo "All trials complete."
