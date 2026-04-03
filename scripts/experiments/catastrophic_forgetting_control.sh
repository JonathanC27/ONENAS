#!/bin/sh
# Catastrophic Forgetting CONTROL Experiment: Sliding Window (No Replay)
#
# This is the control condition for the catastrophic forgetting experiment.
# It uses SlidingWindow training (only the most recent 100 episodes) instead
# of Uniform random replay from all historical episodes. Without replay,
# once the distribution shifts from summer to winter, the model should lose
# summer knowledge -- demonstrating that forgetting DOES occur without replay.
#
# Compare results against catastrophic_forgetting.sh (Uniform replay) to
# show that ONE-NAS's random historical replay is what prevents forgetting.
#
# Distribution A (Summer): May-Sep data (files 5-18, file 19 held out for eval)
#   - Train: 14 files (~822 episodes at length 25)
#   - Eval:  file 19 (held-out summer data)
#
# Distribution B (Winter): Nov-Mar data (files 1, 2, 25-29, 31; file 30 held out for eval)
#   - Train: 8 files (~632 episodes at length 25)
#   - Eval:  file 30 (held-out winter data)
#
# Unified stream: ~1454 total episodes (A: 822, B: 632)
# num_training_sets=100 to ensure enough generations for convergence:
#   Phase boundary: gen ~722 (822 - 100)
#   SlidingWindow flushes A completely by gen ~922 (722 + 100 + 100)
#   Max generations: ~1253 (1454 - 100 - 100 - 1)
#
# Usage:
#   cd build && cmake .. && make catastrophic_forgetting_mpi
#   cd .. && bash scripts/experiments/catastrophic_forgetting_control.sh

cd build

DATA_DIR="../datasets/2020_wind_engine"

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

# Distribution A (Summer): training files 5-18
# 14 files (~822 episodes at length 25)
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

# Distribution B (Winter): files 1, 2, 25-29, 31 (Nov-Mar)
# 8 files (~632 episodes at length 25)
TRAINING_FILES_B="\
${DATA_DIR}/turbine_R80711_2017-2020_1.csv \
${DATA_DIR}/turbine_R80711_2017-2020_2.csv \
${DATA_DIR}/turbine_R80711_2017-2020_25.csv \
${DATA_DIR}/turbine_R80711_2017-2020_26.csv \
${DATA_DIR}/turbine_R80711_2017-2020_27.csv \
${DATA_DIR}/turbine_R80711_2017-2020_28.csv \
${DATA_DIR}/turbine_R80711_2017-2020_29.csv \
${DATA_DIR}/turbine_R80711_2017-2020_31.csv"

# Held-out eval sets (never seen during training)
EVAL_FILES_A="${DATA_DIR}/turbine_R80711_2017-2020_19.csv"
EVAL_FILES_B="${DATA_DIR}/turbine_R80711_2017-2020_30.csv"

NUM_PROCS=22
TOTAL_GENERATIONS=1250
EVAL_FREQUENCY=5

for i in {0..9}
do

exp_name="../results/catastrophic_forgetting_control/$i"
mkdir -p $exp_name
echo "Running Catastrophic Forgetting CONTROL Experiment (trial $i) - SlidingWindow (no replay)"
echo "Results will be saved to: $exp_name"

mpirun -np $NUM_PROCS ./mpi/catastrophic_forgetting_mpi \
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
--output_directory $exp_name \
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
--seed $i \
--std_message_level INFO \
--file_message_level INFO

echo "Trial $i complete. Check $exp_name/forgetting_results.csv for results."

done

echo "All control trials complete."
