#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <fstream>
#include <iostream>
using std::ofstream;
using std::ios;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "common/files.hxx"
#include "mpi.h"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "time_series/online_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

vector<string> arguments;

// CSV file objects for logging
ofstream forgetting_results_csv;
ofstream fitness_log_csv;
string output_directory;

// Evaluation data for Distribution A (held-out summer)
vector<vector<vector<double>>> eval_inputs_a;
vector<vector<vector<double>>> eval_outputs_a;

// Evaluation data for Distribution B (held-out winter)
vector<vector<vector<double>>> eval_inputs_b;
vector<vector<vector<double>>> eval_outputs_b;

// Unified stream: A episodes [0..num_episodes_a-1], B episodes [num_episodes_a..]
vector<vector<vector<double>>> time_series_inputs;
vector<vector<vector<double>>> time_series_outputs;
int32_t num_episodes_a = 0;
int32_t phase_boundary_gen = 0;

void initialize_csv_files() {
    mkpath(output_directory.c_str(), 0777);

    string stats_dir = output_directory + "/stats";
    mkpath(stats_dir.c_str(), 0777);

    string forgetting_csv_path = output_directory + "/forgetting_results.csv";
    forgetting_results_csv.open(forgetting_csv_path.c_str(), ios::out);
    if (!forgetting_results_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", forgetting_csv_path.c_str());
        return;
    }
    forgetting_results_csv << "generation,phase,mse_on_a,mae_on_a,mse_on_b,mae_on_b,best_validation_mse\n";

    string fitness_csv_path = output_directory + "/fitness_log.csv";
    fitness_log_csv.open(fitness_csv_path.c_str(), ios::out);
    if (!fitness_log_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", fitness_csv_path.c_str());
        return;
    }
    fitness_log_csv << "generation,bp_epochs,time,best_val_mae,best_val_mse\n";

    Log::info("CSV files initialized successfully\n");
}

void close_csv_files() {
    if (forgetting_results_csv.is_open()) forgetting_results_csv.close();
    if (fitness_log_csv.is_open()) fitness_log_csv.close();
    Log::info("CSV files closed\n");
}

void evaluate_on_distributions(
    RNN_Genome* genome, const vector<double>& best_params,
    int32_t generation, const string& phase,
    double best_val_mse
) {
    double mse_on_a = genome->get_mse(best_params, eval_inputs_a, eval_outputs_a);
    double mae_on_a = genome->get_mae(best_params, eval_inputs_a, eval_outputs_a);
    double mse_on_b = genome->get_mse(best_params, eval_inputs_b, eval_outputs_b);
    double mae_on_b = genome->get_mae(best_params, eval_inputs_b, eval_outputs_b);

    Log::info("=== Evaluation at generation %d (phase %s) ===\n", generation, phase.c_str());
    Log::info("MSE on A (summer): %lf, MAE on A: %lf\n", mse_on_a, mae_on_a);
    Log::info("MSE on B (winter): %lf, MAE on B: %lf\n", mse_on_b, mae_on_b);
    Log::info("Best validation MSE: %lf\n", best_val_mse);

    if (forgetting_results_csv.is_open()) {
        forgetting_results_csv << generation << "," << phase << ","
                               << mse_on_a << "," << mae_on_a << ","
                               << mse_on_b << "," << mae_on_b << ","
                               << best_val_mse << "\n";
        forgetting_results_csv.flush();
    }
}

void populate_current_time_series_data(
    OnlineSeries* online_series,
    const vector<int32_t>& train_index,
    const vector<int32_t>& validation_index,
    vector<vector<vector<double>>>& current_training_inputs,
    vector<vector<vector<double>>>& current_training_outputs,
    vector<vector<vector<double>>>& current_validation_inputs,
    vector<vector<vector<double>>>& current_validation_outputs
) {
    for (int32_t i = 0; i < (int32_t)train_index.size(); i++) {
        int32_t episode_id = train_index[i];
        TimeSeriesEpisode* episode = online_series->get_episode(episode_id);
        if (episode != nullptr) {
            current_training_inputs.push_back(episode->get_inputs());
            current_training_outputs.push_back(episode->get_outputs());
        } else if (episode_id < (int32_t)time_series_inputs.size()) {
            current_training_inputs.push_back(time_series_inputs[episode_id]);
            current_training_outputs.push_back(time_series_outputs[episode_id]);
        } else {
            Log::error("Episode ID %d out of bounds\n", episode_id);
        }
    }

    for (int32_t i = 0; i < (int32_t)validation_index.size(); i++) {
        int32_t episode_id = validation_index[i];
        TimeSeriesEpisode* episode = online_series->get_episode(episode_id);
        if (episode != nullptr) {
            current_validation_inputs.push_back(episode->get_inputs());
            current_validation_outputs.push_back(episode->get_outputs());
        } else if (episode_id < (int32_t)time_series_inputs.size()) {
            current_validation_inputs.push_back(time_series_inputs[episode_id]);
            current_validation_outputs.push_back(time_series_outputs[episode_id]);
        } else {
            Log::error("Validation episode ID %d out of bounds\n", episode_id);
        }
    }
}

/**
 * Build a modified argument vector that replaces --training_filenames with the given filenames.
 */
vector<string> build_args_with_filenames(const vector<string>& base_args, const string& filename_arg, const vector<string>& filenames) {
    vector<string> new_args;

    bool skip_values = false;
    for (size_t i = 0; i < base_args.size(); i++) {
        if (base_args[i] == "--training_filenames") {
            skip_values = true;
            continue;
        }
        if (skip_values) {
            if (base_args[i].substr(0, 2) == "--") {
                skip_values = false;
                new_args.push_back(base_args[i]);
            }
            continue;
        }
        new_args.push_back(base_args[i]);
    }

    new_args.push_back(filename_arg);
    for (const string& f : filenames) {
        new_args.push_back(f);
    }

    return new_args;
}

int main(int argc, char** argv) {
    std::cout << "Starting Catastrophic Forgetting LSTM Baseline MPI Program" << std::endl;
    MPI_Init(&argc, &argv);

    int32_t rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
    std::cout << "MPI rank " << rank << " of " << max_rank << std::endl;

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + std::to_string(rank));
    Log::restrict_to_rank(0);

    // Only rank 0 does work for the LSTM baseline (no evolution, single genome)
    if (rank != 0) {
        Log::clear_rank_restriction();
        Log::set_id("main_" + std::to_string(rank));
        Log::debug("rank %d idle (LSTM baseline is single-process), waiting for finalize\n", rank);
        Log::release_id("main_" + std::to_string(rank));
        MPI_Finalize();
        return 0;
    }

    // Parse parameters
    get_argument(arguments, "--output_directory", true, output_directory);

    int32_t total_generations = 1000;
    int32_t eval_frequency = 5;
    int32_t bp_iterations = 10;
    get_argument(arguments, "--total_generations", true, total_generations);
    get_argument(arguments, "--eval_frequency", false, eval_frequency);
    get_argument(arguments, "--bp_iterations", false, bp_iterations);

    Log::info("Output directory: %s\n", output_directory.c_str());
    Log::info("Total generations: %d, Eval frequency: %d\n", total_generations, eval_frequency);

    // Parse filenames for distributions A, B, and eval sets
    vector<string> training_filenames_a;
    vector<string> training_filenames_b;
    vector<string> eval_filenames_a;
    vector<string> eval_filenames_b;
    get_argument_vector(arguments, "--training_filenames_a", true, training_filenames_a);
    get_argument_vector(arguments, "--training_filenames_b", true, training_filenames_b);
    get_argument_vector(arguments, "--eval_filenames_a", false, eval_filenames_a);
    get_argument_vector(arguments, "--eval_filenames_b", false, eval_filenames_b);

    if (eval_filenames_a.empty()) {
        eval_filenames_a = training_filenames_a;
        Log::info("No --eval_filenames_a provided, using training filenames A for evaluation\n");
    }
    if (eval_filenames_b.empty()) {
        eval_filenames_b = training_filenames_b;
        Log::info("No --eval_filenames_b provided, using training filenames B for evaluation\n");
    }

    Log::info("Distribution A training files: %d\n", (int32_t)training_filenames_a.size());
    Log::info("Distribution B training files: %d\n", (int32_t)training_filenames_b.size());
    Log::info("Distribution A eval files: %d\n", (int32_t)eval_filenames_a.size());
    Log::info("Distribution B eval files: %d\n", (int32_t)eval_filenames_b.size());

    // --- Load Distribution A data ---
    vector<string> args_a = build_args_with_filenames(arguments, "--training_filenames", training_filenames_a);

    TimeSeriesSets* time_series_sets_a = TimeSeriesSets::generate_from_arguments(args_a);
    vector<vector<vector<double>>> inputs_a, outputs_a;
    slice_online_time_series(args_a, time_series_sets_a, inputs_a, outputs_a);
    Log::info("Distribution A - inputs shape: %d, %d, %d\n",
              (int32_t)inputs_a.size(), (int32_t)inputs_a[0].size(), (int32_t)inputs_a[0][0].size());

    // Get normalization bounds from A
    map<string, double> norm_mins = time_series_sets_a->get_normalize_mins();
    map<string, double> norm_maxs = time_series_sets_a->get_normalize_maxs();
    string norm_type = time_series_sets_a->get_normalize_type();

    // --- Load Distribution B data with A's normalization bounds ---
    vector<string> args_b = build_args_with_filenames(arguments, "--training_filenames", training_filenames_b);

    vector<string> args_b_custom;
    for (size_t i = 0; i < args_b.size(); i++) {
        if (args_b[i] == "--normalize") {
            i++; // skip value
            continue;
        }
        args_b_custom.push_back(args_b[i]);
    }
    args_b_custom.push_back("--normalize");
    args_b_custom.push_back("none");

    TimeSeriesSets* time_series_sets_b = TimeSeriesSets::generate_from_arguments(args_b_custom);

    if (norm_type == "min_max" && !norm_mins.empty()) {
        time_series_sets_b->normalize_min_max(norm_mins, norm_maxs);
        Log::info("Applied Distribution A's min_max normalization bounds to Distribution B\n");
    } else if (norm_type != "none") {
        Log::warning("Normalization type '%s' - B will use its own normalization. Consider using min_max.\n", norm_type.c_str());
        delete time_series_sets_b;
        time_series_sets_b = TimeSeriesSets::generate_from_arguments(args_b);
    }

    vector<vector<vector<double>>> inputs_b, outputs_b;
    slice_online_time_series(args_b, time_series_sets_b, inputs_b, outputs_b);
    Log::info("Distribution B - inputs shape: %d, %d, %d\n",
              (int32_t)inputs_b.size(), (int32_t)inputs_b[0].size(), (int32_t)inputs_b[0][0].size());

    // --- Concatenate A and B into unified stream ---
    num_episodes_a = inputs_a.size();
    for (auto& ep : inputs_a)  time_series_inputs.push_back(std::move(ep));
    for (auto& ep : outputs_a) time_series_outputs.push_back(std::move(ep));
    for (auto& ep : inputs_b)  time_series_inputs.push_back(std::move(ep));
    for (auto& ep : outputs_b) time_series_outputs.push_back(std::move(ep));
    Log::info("Unified stream: %d total episodes (A: %d, B: %d)\n",
              (int32_t)time_series_inputs.size(), num_episodes_a, (int32_t)inputs_b.size());

    // --- Load Distribution A evaluation data ---
    vector<string> input_param_names = time_series_sets_a->get_input_parameter_names();
    vector<string> output_param_names = time_series_sets_a->get_output_parameter_names();

    TimeSeriesSets* time_series_sets_eval = TimeSeriesSets::generate_test(
        eval_filenames_a, input_param_names, output_param_names
    );

    if (norm_type == "min_max" && !norm_mins.empty()) {
        time_series_sets_eval->normalize_min_max(norm_mins, norm_maxs);
    }

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);
    time_series_sets_eval->export_test_series(time_offset, eval_inputs_a, eval_outputs_a);

    int32_t time_series_length = 25;
    get_argument(arguments, "--time_series_length", true, time_series_length);
    slice_input_data(eval_inputs_a, eval_outputs_a, time_series_length);
    Log::info("Distribution A eval - inputs shape: %d, %d, %d\n",
              (int32_t)eval_inputs_a.size(), (int32_t)eval_inputs_a[0].size(), (int32_t)eval_inputs_a[0][0].size());

    // --- Load Distribution B evaluation data ---
    TimeSeriesSets* time_series_sets_eval_b = TimeSeriesSets::generate_test(
        eval_filenames_b, input_param_names, output_param_names
    );

    if (norm_type == "min_max" && !norm_mins.empty()) {
        time_series_sets_eval_b->normalize_min_max(norm_mins, norm_maxs);
    }

    time_series_sets_eval_b->export_test_series(time_offset, eval_inputs_b, eval_outputs_b);
    slice_input_data(eval_inputs_b, eval_outputs_b, time_series_length);
    Log::info("Distribution B eval - inputs shape: %d, %d, %d\n",
              (int32_t)eval_inputs_b.size(), (int32_t)eval_inputs_b[0].size(), (int32_t)eval_inputs_b[0][0].size());

    // --- Compute phase boundary and create OnlineSeries ---
    int32_t num_training_sets = 300;
    int32_t num_validation_sets = 25;
    get_argument(arguments, "--num_training_sets", false, num_training_sets);
    get_argument(arguments, "--num_validation_sets", false, num_validation_sets);

    phase_boundary_gen = num_episodes_a - num_training_sets;
    Log::info("Phase boundary at generation %d (num_episodes_a=%d - num_training_sets=%d)\n",
              phase_boundary_gen, num_episodes_a, num_training_sets);

    int32_t total_episodes = time_series_inputs.size();
    int32_t max_possible_gen = total_episodes - num_training_sets - num_validation_sets - 1;
    if (total_generations > max_possible_gen) {
        Log::warning("total_generations (%d) > max possible (%d), clamping\n", total_generations, max_possible_gen);
        total_generations = max_possible_gen;
    }

    OnlineSeries* online_series = new OnlineSeries(total_episodes, arguments);
    online_series->initialize_episodes(time_series_inputs, time_series_outputs);

    // Initialize stratified bins from source file counts
    int32_t num_episodes_b = total_episodes - num_episodes_a;
    online_series->initialize_bins(num_episodes_a, (int32_t)training_filenames_a.size(),
                                   num_episodes_b, (int32_t)training_filenames_b.size());

    online_series->print_episode_stats();

    // --- Setup weight update ---
    WeightUpdate* weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);
    Log::major_divider(Log::INFO, "Created weight update method!");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);
    Log::major_divider(Log::INFO, "Created weight rules!");

    // --- Create fixed LSTM genome ---
    int32_t num_hidden_layers = 1;
    int32_t num_hidden_nodes = input_param_names.size(); // match paper: hidden size = input size
    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--num_hidden_layers", false, num_hidden_layers);
    get_argument(arguments, "--num_hidden_nodes", false, num_hidden_nodes);
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    string rnn_type = "lstm";
    get_argument(arguments, "--rnn_type", false, rnn_type);

    RNN_Genome* genome = nullptr;
    if (rnn_type == "lstm") {
        genome = create_lstm(input_param_names, num_hidden_layers, num_hidden_nodes,
                             output_param_names, max_recurrent_depth, weight_rules);
    } else if (rnn_type == "gru") {
        genome = create_gru(input_param_names, num_hidden_layers, num_hidden_nodes,
                            output_param_names, max_recurrent_depth, weight_rules);
    } else if (rnn_type == "mgu") {
        genome = create_mgu(input_param_names, num_hidden_layers, num_hidden_nodes,
                            output_param_names, max_recurrent_depth, weight_rules);
    } else if (rnn_type == "ugrnn") {
        genome = create_ugrnn(input_param_names, num_hidden_layers, num_hidden_nodes,
                              output_param_names, max_recurrent_depth, weight_rules);
    } else if (rnn_type == "delta") {
        genome = create_delta(input_param_names, num_hidden_layers, num_hidden_nodes,
                              output_param_names, max_recurrent_depth, weight_rules);
    } else {
        Log::fatal("Unknown --rnn_type: '%s'. Supported: lstm, gru, mgu, ugrnn, delta\n", rnn_type.c_str());
        exit(1);
    }

    Log::info("Created %s genome with %d hidden layer(s), %d hidden nodes, max_recurrent_depth=%d\n",
              rnn_type.c_str(), num_hidden_layers, num_hidden_nodes, max_recurrent_depth);

    // Initialize with random weights and set BP iterations
    genome->initialize_randomly(weight_rules);
    genome->set_bp_iterations(bp_iterations);

    // Get initial parameters as our starting point
    vector<double> best_params = genome->get_best_parameters();

    Log::info("LSTM genome has %d parameters\n", (int32_t)best_params.size());

    // --- Initialize CSV files ---
    initialize_csv_files();

    // --- Main generation loop ---
    auto experiment_start = std::chrono::steady_clock::now();

    for (int32_t gen = 0; gen < total_generations; gen++) {
        auto gen_start = std::chrono::steady_clock::now();
        string phase = (gen <= phase_boundary_gen) ? "pure_A" : "mixed_AB";

        online_series->set_current_index(gen);

        Log::info("Generation %d (Phase %s)\n", gen, phase.c_str());

        if (gen == phase_boundary_gen) {
            Log::info("=== PHASE BOUNDARY: B episodes entering training window ===\n");
        }

        // Get training and validation indices
        vector<int32_t> train_index;
        online_series->get_training_index(train_index);

        vector<int32_t> validation_index;
        online_series->get_validation_index(validation_index);

        // Populate data
        vector<vector<vector<double>>> current_training_inputs;
        vector<vector<vector<double>>> current_training_outputs;
        vector<vector<vector<double>>> current_validation_inputs;
        vector<vector<vector<double>>> current_validation_outputs;

        populate_current_time_series_data(
            online_series, train_index, validation_index,
            current_training_inputs, current_training_outputs,
            current_validation_inputs, current_validation_outputs
        );

        // Set the genome's weights to the current best before continuing training
        genome->set_weights(best_params);
        genome->set_initial_parameters(best_params);

        // Train via backpropagation (continues from current weights)
        genome->backpropagate_stochastic(
            current_training_inputs, current_training_outputs,
            current_validation_inputs, current_validation_outputs,
            weight_update_method
        );

        // Always carry forward latest trained weights (true continual learning)
        double gen_val_mse = genome->get_best_validation_mse();
        best_params = genome->get_best_parameters();
        Log::info("Generation %d: validation MSE: %lf\n", gen, gen_val_mse);

        // Log fitness
        auto gen_end = std::chrono::steady_clock::now();
        double gen_time = std::chrono::duration<double>(gen_end - gen_start).count();
        double gen_val_mae = genome->get_best_validation_mae();

        if (fitness_log_csv.is_open()) {
            fitness_log_csv << gen << "," << bp_iterations << "," << gen_time << ","
                           << gen_val_mae << "," << gen_val_mse << "\n";
            fitness_log_csv.flush();
        }

        // Evaluate on both distributions at regular intervals throughout training.
        // This captures the full learning curve during pure-A and any forgetting during mixed-AB.
        bool should_eval = (gen % eval_frequency == 0) || (gen == total_generations - 1);

        if (should_eval) {
            evaluate_on_distributions(genome, best_params, gen, phase, gen_val_mse);
        }

        Log::info("Generation %d finished (%.1fs)\n", gen, gen_time);
    }

    auto experiment_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(experiment_end - experiment_start).count();
    Log::info("Experiment complete. Total time: %.1f seconds\n", total_time);

    // Save final genome
    string genome_path = output_directory + "/best_lstm_genome.bin";
    char* genome_str;
    int32_t genome_length;
    genome->write_to_array(&genome_str, genome_length);
    ofstream genome_file(genome_path, ios::binary);
    if (genome_file.is_open()) {
        genome_file.write(genome_str, genome_length);
        genome_file.close();
        Log::info("Saved final genome to %s\n", genome_path.c_str());
    }
    free(genome_str);

    // Cleanup
    close_csv_files();

    delete genome;
    delete online_series;
    delete weight_update_method;
    delete weight_rules;

    delete time_series_sets_a;
    delete time_series_sets_b;
    delete time_series_sets_eval;
    delete time_series_sets_eval_b;

    Log::set_id("main_" + std::to_string(rank));
    Log::release_id("main_" + std::to_string(rank));
    MPI_Finalize();

    return 0;
}
