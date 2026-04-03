#!/bin/sh
# Catastrophic Forgetting LSTM CONTROL Baseline: Sliding Window (No Replay)
#
# This is the control condition for the LSTM baseline. It uses SlidingWindow
# training (only the most recent 100 episodes) instead of Uniform random replay.
# This matches the protocol of catastrophic_forgetting_control.sh but with a
# fixed LSTM instead of ONE-NAS.
#
# Without replay, once the distribution shifts from summer to winter, the LSTM
# should catastrophically forget summer knowledge.
#
# Usage:
#   cd build && cmake .. && make catastrophic_forgetting_lstm_mpi
#   cd .. && bash scripts/experiments/catastrophic_forgetting_lstm_control.sh

cd build

DATA_DIR="../datasets/2020_wind_engine"

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

# Distribution A (Summer): training files 5-18
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

# Distribution B (Winter): files 1, 2, 25-29, 31
TRAINING_FILES_B="\
${DATA_DIR}/turbine_R80711_2017-2020_1.csv \
${DATA_DIR}/turbine_R80711_2017-2020_2.csv \
${DATA_DIR}/turbine_R80711_2017-2020_25.csv \
${DATA_DIR}/turbine_R80711_2017-2020_26.csv \
${DATA_DIR}/turbine_R80711_2017-2020_27.csv \
${DATA_DIR}/turbine_R80711_2017-2020_28.csv \
${DATA_DIR}/turbine_R80711_2017-2020_29.csv \
${DATA_DIR}/turbine_R80711_2017-2020_31.csv"

# Held-out eval sets
EVAL_FILES_A="${DATA_DIR}/turbine_R80711_2017-2020_19.csv"
EVAL_FILES_B="${DATA_DIR}/turbine_R80711_2017-2020_30.csv"

# Single process — no evolution needed for LSTM baseline
NUM_PROCS=1
TOTAL_GENERATIONS=1250
EVAL_FREQUENCY=5

for i in {0..9}
do

exp_name="../results/catastrophic_forgetting_lstm_control/$i"
mkdir -p $exp_name
echo "Running LSTM Control Catastrophic Forgetting Experiment (trial $i) - SlidingWindow (no replay)"
echo "Results will be saved to: $exp_name"

mpirun -np $NUM_PROCS ./mpi/catastrophic_forgetting_lstm_mpi \
--training_filenames_a $TRAINING_FILES_A \
--training_filenames_b $TRAINING_FILES_B \
--eval_filenames_a $EVAL_FILES_A \
--eval_filenames_b $EVAL_FILES_B \
--total_generations $TOTAL_GENERATIONS \
--eval_frequency $EVAL_FREQUENCY \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--bp_iterations 10 \
--output_directory $exp_name \
--time_series_length 25 \
--num_validation_sets 100 \
--num_training_sets 100 \
--get_train_data_by SlidingWindow \
--normalize min_max \
--num_hidden_layers 1 \
--num_hidden_nodes 22 \
--rnn_type lstm \
--max_recurrent_depth 10 \
--learning_rate 0.001 \
--seed $i \
--std_message_level INFO \
--file_message_level INFO

echo "Trial $i complete. Check $exp_name/forgetting_results.csv for results."

done

echo "All LSTM control trials complete."
