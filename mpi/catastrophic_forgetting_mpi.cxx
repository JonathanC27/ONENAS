#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <fstream>
#include <iostream>
using std::ofstream;
using std::ios;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "common/files.hxx"
#include "onenas/onenas.hxx"
#include "onenas/onenas_island_speciation_strategy.hxx"
#include "mpi.h"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "time_series/online_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

#define WORK_REQUEST_TAG  1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG        3
#define TERMINATE_TAG     4

mutex onenas_mutex;

vector<string> arguments;

ONENAS* onenas;
WeightUpdate* weight_update_method;

bool finished = false;

// CSV file objects for logging
ofstream training_indices_csv;
ofstream validation_test_indices_csv;
ofstream forgetting_results_csv;
string output_directory;

// Data for both distributions
vector<vector<vector<double>>> time_series_inputs_a;
vector<vector<vector<double>>> time_series_outputs_a;
vector<vector<vector<double>>> time_series_inputs_b;
vector<vector<vector<double>>> time_series_outputs_b;

// Evaluation data for Distribution A
vector<vector<vector<double>>> eval_inputs_a;
vector<vector<vector<double>>> eval_outputs_a;

// Global fallback arrays used by populate_current_time_series_data
vector<vector<vector<double>>> time_series_inputs;
vector<vector<vector<double>>> time_series_outputs;

vector<int32_t> time_series_index;
int32_t generated_population_size;
int32_t number_islands;
int32_t total_generation;

bool has_generated_enough_genomes(int32_t current_generated_genomes) {
    OneNasIslandSpeciationStrategy* onenas_strategy =
        dynamic_cast<OneNasIslandSpeciationStrategy*>(onenas->get_speciation_strategy());

    int32_t current_generated_population_size = generated_population_size;
    if (onenas_strategy != nullptr) {
        current_generated_population_size = onenas_strategy->get_generated_population_size();
        if (current_generated_population_size != generated_population_size) {
            Log::info("Master: Generated population size updated from %d to %d\n",
                     generated_population_size, current_generated_population_size);
        }
    }

    return current_generated_genomes >= current_generated_population_size * number_islands;
}

string get_stats_directory() {
    return output_directory + "/stats";
}

void initialize_csv_files() {
    mkpath(output_directory.c_str(), 0777);

    string stats_dir = get_stats_directory();
    mkpath(stats_dir.c_str(), 0777);

    string training_csv_path = stats_dir + "/training_indices.csv";
    training_indices_csv.open(training_csv_path.c_str(), ios::out);
    if (!training_indices_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", training_csv_path.c_str());
        return;
    }
    training_indices_csv << "genome_id,generation,training_indices\n";

    string validation_csv_path = stats_dir + "/validation_test_indices.csv";
    validation_test_indices_csv.open(validation_csv_path.c_str(), ios::out);
    if (!validation_test_indices_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", validation_csv_path.c_str());
        return;
    }
    validation_test_indices_csv << "generation,validation_indices,test_index\n";

    // Initialize forgetting results CSV
    string forgetting_csv_path = output_directory + "/forgetting_results.csv";
    forgetting_results_csv.open(forgetting_csv_path.c_str(), ios::out);
    if (!forgetting_results_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", forgetting_csv_path.c_str());
        return;
    }
    forgetting_results_csv << "generation,phase,mse_on_a,mae_on_a,best_validation_mse\n";

    Log::info("CSV files initialized successfully in %s\n", stats_dir.c_str());
}

void close_csv_files() {
    if (training_indices_csv.is_open()) {
        training_indices_csv.close();
    }
    if (validation_test_indices_csv.is_open()) {
        validation_test_indices_csv.close();
    }
    if (forgetting_results_csv.is_open()) {
        forgetting_results_csv.close();
    }
    Log::info("CSV files closed\n");
}

void write_training_indices_to_csv(int32_t genome_id, int32_t generation, const vector<int32_t>& training_indices) {
    if (!training_indices_csv.is_open()) {
        Log::error("Training indices CSV file is not open\n");
        return;
    }

    training_indices_csv << genome_id << "," << generation << ",\"";
    for (size_t i = 0; i < training_indices.size(); i++) {
        if (i > 0) training_indices_csv << ";";
        training_indices_csv << training_indices[i];
    }
    training_indices_csv << "\"\n";
    training_indices_csv.flush();
}

void write_validation_test_indices_to_csv(int32_t generation, const vector<int32_t>& validation_indices, int32_t test_index) {
    if (!validation_test_indices_csv.is_open()) {
        Log::error("Validation/test indices CSV file is not open\n");
        return;
    }

    validation_test_indices_csv << generation << ",\"";
    for (size_t i = 0; i < validation_indices.size(); i++) {
        if (i > 0) validation_test_indices_csv << ";";
        validation_test_indices_csv << validation_indices[i];
    }
    validation_test_indices_csv << "\"," << test_index << "\n";
    validation_test_indices_csv.flush();
}

void send_work_request(int32_t target) {
    int32_t work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int32_t source) {
    MPI_Status status;
    int32_t work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int32_t source) {
    MPI_Status status;
    int32_t length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int32_t length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void send_genome_to(int32_t target, RNN_Genome* genome) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int32_t target) {
    int32_t terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int32_t source) {
    MPI_Status status;
    int32_t terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
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
            Log::debug("Worker: training episode ID: %d\n", episode_id);
        } else {
            Log::warning("Episode ID %d not found, falling back to legacy method\n", episode_id);
            if (episode_id < (int32_t)time_series_inputs.size()) {
                current_training_inputs.push_back(time_series_inputs[episode_id]);
                current_training_outputs.push_back(time_series_outputs[episode_id]);
                Log::debug("Worker: training legacy index: %d\n", episode_id);
            } else {
                Log::error("Episode ID %d out of bounds for both episodes and legacy data\n", episode_id);
            }
        }
    }

    for (int32_t i = 0; i < (int32_t)validation_index.size(); i++) {
        int32_t episode_id = validation_index[i];
        TimeSeriesEpisode* episode = online_series->get_episode(episode_id);
        if (episode != nullptr) {
            current_validation_inputs.push_back(episode->get_inputs());
            current_validation_outputs.push_back(episode->get_outputs());
            Log::debug("Worker: validation episode ID: %d\n", episode_id);
        } else {
            Log::warning("Episode ID %d not found for validation, falling back to legacy method\n", episode_id);
            if (episode_id < (int32_t)time_series_inputs.size()) {
                current_validation_inputs.push_back(time_series_inputs[episode_id]);
                current_validation_outputs.push_back(time_series_outputs[episode_id]);
                Log::debug("Worker: validation legacy index: %d\n", episode_id);
            } else {
                Log::error("Episode ID %d out of bounds for both episodes and legacy data\n", episode_id);
            }
        }
    }
}

void populate_test_and_validation_data(
    OnlineSeries* online_series,
    int32_t test_index,
    const vector<int32_t>& validation_index,
    vector<vector<vector<double>>>& current_test_inputs,
    vector<vector<vector<double>>>& current_test_outputs,
    vector<vector<vector<double>>>& current_validation_inputs,
    vector<vector<vector<double>>>& current_validation_outputs
) {
    TimeSeriesEpisode* test_episode = online_series->get_episode(test_index);
    if (test_episode != nullptr) {
        current_test_inputs.push_back(test_episode->get_inputs());
        current_test_outputs.push_back(test_episode->get_outputs());
    } else {
        Log::error("Test episode ID %d not found in episodes\n", test_index);
        exit(1);
    }

    for (int32_t i = 0; i < (int32_t)validation_index.size(); i++) {
        int32_t episode_id = validation_index[i];
        TimeSeriesEpisode* val_episode = online_series->get_episode(episode_id);
        if (val_episode != nullptr) {
            current_validation_inputs.push_back(val_episode->get_inputs());
            current_validation_outputs.push_back(val_episode->get_outputs());
            Log::debug("validation episode ID: %d\n", episode_id);
        } else {
            Log::error("Validation episode ID %d not found in episodes\n", episode_id);
            exit(1);
        }
    }
    Log::info("Current testing episode ID: %d\n", test_index);
}

void master(int32_t max_rank, OnlineSeries* online_series, int32_t current_generation) {
    Log::debug("MAX int32_t: %d\n", numeric_limits<int32_t>::max());

    int32_t terminates_sent = 0;
    int32_t generated_genome = 0;
    int32_t evaluated_genome = 0;
    while (true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);
            if (!has_generated_enough_genomes(generated_genome)) {
                onenas_mutex.lock();
                RNN_Genome *genome = onenas->generate_genome();
                onenas_mutex.unlock();

                if (genome != NULL) {
                    vector<int32_t> master_training_index;
                    online_series->get_training_index(master_training_index);

                    int32_t generation_id = genome->get_generation_id();
                    genome->set_training_indices(master_training_index);

                    write_training_indices_to_csv(generation_id, current_generation, master_training_index);

                    Log::info("Master: Generated %d training indices for genome %d, sending to worker: %d\n",
                             master_training_index.size(), generation_id, source);
                    Log::debug("sending genome to: %d\n", source);
                    send_genome_to(source, genome);

                    delete genome;
                    generated_genome++;
                } else {
                    Log::fatal("Returned NULL genome from generate genome function, this should never happen!\n");
                    exit(1);
                }
            } else {
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::info("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    Log::debug("Ending genome, generated genome is %d, evaluated genome is %d\n", generated_genome, evaluated_genome);
                    return;
                }
            }

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome* genome = receive_genome_from(source);

            onenas_mutex.lock();
            onenas->insert_genome(genome);
            onenas_mutex.unlock();

            delete genome;
            evaluated_genome++;
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int32_t rank, OnlineSeries* online_series) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int32_t tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::info("worker %d received genome!\n", rank);
            RNN_Genome* genome = receive_genome_from(0);

            vector<vector<vector<double>>> current_training_inputs;
            vector<vector<vector<double>>> current_training_outputs;
            vector<vector<vector<double>>> current_validation_inputs;
            vector<vector<vector<double>>> current_validation_outputs;

            vector<int32_t> train_index = genome->get_training_indices();
            vector<int32_t> validation_index;

            online_series->get_validation_index(validation_index);

            Log::info("Worker %d: Using %d training indices provided by master for genome %d\n",
                     rank, train_index.size(), genome->get_generation_id());

            populate_current_time_series_data(
                online_series, train_index, validation_index,
                current_training_inputs, current_training_outputs,
                current_validation_inputs, current_validation_outputs
            );

            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(current_training_inputs, current_training_outputs, current_validation_inputs, current_validation_outputs, weight_update_method);
            genome->evaluate_online(current_validation_inputs, current_validation_outputs);
            Log::release_id(log_id);

            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    Log::release_id("worker_" + to_string(rank));
}

/**
 * Evaluate the best genome on Distribution A data and log results.
 */
void evaluate_on_distribution_a(
    int32_t generation, const string& phase,
    const vector<vector<vector<double>>>& a_eval_inputs,
    const vector<vector<vector<double>>>& a_eval_outputs
) {
    RNN_Genome* best_genome = onenas->get_best_genome();
    if (best_genome == NULL) {
        Log::warning("No best genome available for evaluation at generation %d\n", generation);
        return;
    }

    vector<double> best_params = best_genome->get_best_parameters();
    double mse_on_a = best_genome->get_mse(best_params, a_eval_inputs, a_eval_outputs);
    double mae_on_a = best_genome->get_mae(best_params, a_eval_inputs, a_eval_outputs);
    double best_val_mse = best_genome->get_best_validation_mse();

    Log::info("=== Evaluation on Distribution A at generation %d (phase %s) ===\n", generation, phase.c_str());
    Log::info("MSE on A: %lf, MAE on A: %lf, Best validation MSE: %lf\n", mse_on_a, mae_on_a, best_val_mse);

    if (forgetting_results_csv.is_open()) {
        forgetting_results_csv << generation << "," << phase << ","
                               << mse_on_a << "," << mae_on_a << "," << best_val_mse << "\n";
        forgetting_results_csv.flush();
    }
}

/**
 * Build a modified argument vector that replaces --training_filenames with the given filenames.
 * This allows reusing generate_from_arguments with different file sets.
 */
vector<string> build_args_with_filenames(const vector<string>& base_args, const string& filename_arg, const vector<string>& filenames) {
    vector<string> new_args;

    // Copy all arguments except --training_filenames and its values
    bool skip_values = false;
    for (size_t i = 0; i < base_args.size(); i++) {
        if (base_args[i] == "--training_filenames") {
            skip_values = true;
            continue;
        }
        if (skip_values) {
            // Skip values until we hit another -- argument
            if (base_args[i].substr(0, 2) == "--") {
                skip_values = false;
                new_args.push_back(base_args[i]);
            }
            continue;
        }
        new_args.push_back(base_args[i]);
    }

    // Add the new filename argument
    new_args.push_back(filename_arg);
    for (const string& f : filenames) {
        new_args.push_back(f);
    }

    return new_args;
}

int main(int argc, char** argv) {
    std::cout << "Starting Catastrophic Forgetting MPI Program" << std::endl;
    MPI_Init(&argc, &argv);

    int32_t rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
    std::cout << "MPI rank " << rank << " of " << max_rank << std::endl;

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);

    // Load required parameters
    get_argument(arguments, "--number_islands", true, number_islands);
    get_argument(arguments, "--generated_population_size", true, generated_population_size);
    get_argument(arguments, "--output_directory", true, output_directory);

    // Load catastrophic forgetting specific parameters
    int32_t generations_a = 50;
    int32_t generations_b = 50;
    int32_t eval_frequency = 1;
    get_argument(arguments, "--generations_a", true, generations_a);
    get_argument(arguments, "--generations_b", true, generations_b);
    get_argument(arguments, "--eval_frequency", false, eval_frequency);

    Log::info("Output directory: %s\n", output_directory.c_str());
    Log::info("Generations A: %d, Generations B: %d, Eval frequency: %d\n", generations_a, generations_b, eval_frequency);

    // Parse filenames for distributions A, B, and A eval
    vector<string> training_filenames_a;
    vector<string> training_filenames_b;
    vector<string> eval_filenames_a;
    get_argument_vector(arguments, "--training_filenames_a", true, training_filenames_a);
    get_argument_vector(arguments, "--training_filenames_b", true, training_filenames_b);
    get_argument_vector(arguments, "--eval_filenames_a", false, eval_filenames_a);

    // If no separate eval filenames provided, use the training filenames for A
    if (eval_filenames_a.empty()) {
        eval_filenames_a = training_filenames_a;
        Log::info("No --eval_filenames_a provided, using training filenames A for evaluation\n");
    }

    Log::info("Distribution A training files: %d\n", (int32_t)training_filenames_a.size());
    Log::info("Distribution B training files: %d\n", (int32_t)training_filenames_b.size());
    Log::info("Distribution A eval files: %d\n", (int32_t)eval_filenames_a.size());

    // --- Load Distribution A data ---
    // Build arguments with A's training filenames
    vector<string> args_a = build_args_with_filenames(arguments, "--training_filenames", training_filenames_a);

    TimeSeriesSets* time_series_sets_a = TimeSeriesSets::generate_from_arguments(args_a);
    slice_online_time_series(args_a, time_series_sets_a, time_series_inputs_a, time_series_outputs_a);
    Log::info("Distribution A - inputs shape: %d, %d, %d\n",
              time_series_inputs_a.size(), time_series_inputs_a[0].size(), time_series_inputs_a[0][0].size());

    // Get normalization bounds from A (which includes A+B data range ideally, but we use A's bounds)
    map<string, double> norm_mins = time_series_sets_a->get_normalize_mins();
    map<string, double> norm_maxs = time_series_sets_a->get_normalize_maxs();
    string norm_type = time_series_sets_a->get_normalize_type();

    // --- Load Distribution B data with A's normalization bounds ---
    vector<string> args_b = build_args_with_filenames(arguments, "--training_filenames", training_filenames_b);

    // For B, we need to load but apply A's normalization. We create a modified arguments
    // vector that disables auto-normalization, then manually apply A's bounds.
    // However, for simplicity and correctness, we load B with --normalize none
    // and then normalize with A's bounds.
    // Replace the --normalize argument with "none" for B loading
    vector<string> args_b_no_norm;
    for (size_t i = 0; i < args_b.size(); i++) {
        args_b_no_norm.push_back(args_b[i]);
        if (args_b[i] == "--normalize" && i + 1 < args_b.size()) {
            args_b_no_norm.push_back("none");
            i++; // skip original normalize value
        }
    }

    // Actually, we need to override normalize. Let's just replace the normalize arg.
    // Re-do: remove --normalize and its value, add --normalize none
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

    // Now apply A's normalization bounds to B
    if (norm_type == "min_max" && !norm_mins.empty()) {
        time_series_sets_b->normalize_min_max(norm_mins, norm_maxs);
        Log::info("Applied Distribution A's min_max normalization bounds to Distribution B\n");
    } else if (norm_type != "none") {
        Log::warning("Normalization type '%s' - B will use its own normalization. Consider using min_max.\n", norm_type.c_str());
        // For avg_std_dev, we'd need a similar override - for now, just re-load with auto-norm
        delete time_series_sets_b;
        time_series_sets_b = TimeSeriesSets::generate_from_arguments(args_b);
    }

    slice_online_time_series(args_b, time_series_sets_b, time_series_inputs_b, time_series_outputs_b);
    Log::info("Distribution B - inputs shape: %d, %d, %d\n",
              time_series_inputs_b.size(), time_series_inputs_b[0].size(), time_series_inputs_b[0][0].size());

    // --- Load Distribution A evaluation data ---
    vector<string> input_param_names = time_series_sets_a->get_input_parameter_names();
    vector<string> output_param_names = time_series_sets_a->get_output_parameter_names();

    TimeSeriesSets* time_series_sets_eval = TimeSeriesSets::generate_test(
        eval_filenames_a, input_param_names, output_param_names
    );

    // Apply A's normalization to eval data
    if (norm_type == "min_max" && !norm_mins.empty()) {
        time_series_sets_eval->normalize_min_max(norm_mins, norm_maxs);
    }

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);
    time_series_sets_eval->export_test_series(time_offset, eval_inputs_a, eval_outputs_a);
    Log::info("Distribution A eval - inputs shape: %d, %d, %d\n",
              eval_inputs_a.size(), eval_inputs_a[0].size(), eval_inputs_a[0][0].size());

    // --- Create OnlineSeries for A and B ---
    int32_t num_sets_a = time_series_inputs_a.size();
    int32_t num_sets_b = time_series_inputs_b.size();
    Log::info("Distribution A episodes: %d, Distribution B episodes: %d\n", num_sets_a, num_sets_b);

    // Validate that we have enough episodes for the requested generations
    if (generations_a > num_sets_a) {
        Log::warning("generations_a (%d) > num episodes A (%d), clamping to %d\n", generations_a, num_sets_a, num_sets_a);
        generations_a = num_sets_a;
    }
    if (generations_b > num_sets_b) {
        Log::warning("generations_b (%d) > num episodes B (%d), clamping to %d\n", generations_b, num_sets_b, num_sets_b);
        generations_b = num_sets_b;
    }

    OnlineSeries* online_series_a = new OnlineSeries(num_sets_a, arguments);
    OnlineSeries* online_series_b = new OnlineSeries(num_sets_b, arguments);

    // Initialize episodes for A and B
    online_series_a->initialize_episodes(time_series_inputs_a, time_series_outputs_a);
    online_series_a->print_episode_stats();

    online_series_b->initialize_episodes(time_series_inputs_b, time_series_outputs_b);
    online_series_b->print_episode_stats();

    total_generation = generations_a + generations_b;

    // --- Setup weight update and seed genome ---
    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);
    Log::major_divider(Log::INFO, "Created weight update method!");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);
    if (weight_rules == NULL) {
        Log::fatal("ERROR: Failed to create weight rules\n");
        exit(1);
    }
    Log::major_divider(Log::INFO, "Created weight rules!");

    // Use A's time_series_sets for seed genome creation (defines input/output structure)
    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets_a, weight_rules);
    Log::major_divider(Log::INFO, "Created seed genome!");

    Log::clear_rank_restriction();

    if (rank == 0) {
        onenas = generate_onenas_from_arguments(arguments, time_series_sets_a, weight_rules, seed_genome);
        Log::major_divider(Log::INFO, "Created ONENAS!");

        initialize_csv_files();
    }

    // --- Main two-phase generation loop ---
    for (int32_t gen = 0; gen < total_generation; gen++) {
        // Determine current phase and online series
        OnlineSeries* current_online_series;
        string current_phase;
        int32_t phase_gen;

        if (gen < generations_a) {
            current_online_series = online_series_a;
            current_phase = "A";
            phase_gen = gen;
        } else {
            current_online_series = online_series_b;
            current_phase = "B";
            phase_gen = gen - generations_a;
        }

        // Set the global fallback arrays for the current phase
        if (current_phase == "A") {
            time_series_inputs = time_series_inputs_a;
            time_series_outputs = time_series_outputs_a;
        } else {
            time_series_inputs = time_series_inputs_b;
            time_series_outputs = time_series_outputs_b;
        }

        current_online_series->set_current_index(phase_gen);

        if (rank == 0) {
            Log::major_divider(Log::INFO, "New generation");
            Log::info("Generation %d (Phase %s, phase_gen %d)\n", gen, current_phase.c_str(), phase_gen);
            Log::log_memory_usage("Generation " + std::to_string(gen) + " start");

            // Evaluate on A at phase boundary (start of Phase B)
            if (gen == generations_a) {
                Log::info("=== PHASE BOUNDARY: Switching from A to B ===\n");
                evaluate_on_distribution_a(gen, "boundary", eval_inputs_a, eval_outputs_a);
            }

            master(max_rank, current_online_series, gen);
        } else {
            worker(rank, current_online_series);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            Log::minor_divider(Log::INFO);
            vector<int32_t> validation_index;
            current_online_series->get_validation_index(validation_index);
            int32_t test_index = current_online_series->get_test_index();

            write_validation_test_indices_to_csv(gen, validation_index, test_index);

            vector<vector<vector<double>>> current_test_inputs;
            vector<vector<vector<double>>> current_test_outputs;
            vector<vector<vector<double>>> current_validation_inputs;
            vector<vector<vector<double>>> current_validation_outputs;

            populate_test_and_validation_data(
                current_online_series, test_index, validation_index,
                current_test_inputs, current_test_outputs,
                current_validation_inputs, current_validation_outputs
            );

            // Finalize generation
            OneNasIslandSpeciationStrategy* onenas_strategy =
                dynamic_cast<OneNasIslandSpeciationStrategy*>(onenas->get_speciation_strategy());
            vector<RNN_Genome*> elite_genomes;
            if (onenas_strategy != nullptr) {
                elite_genomes = onenas_strategy->finalize_generation_with_genomes(
                    gen, current_validation_inputs, current_validation_outputs,
                    current_test_inputs, current_test_outputs);
            } else {
                Log::error("Failed to cast speciation strategy to OneNasIslandSpeciationStrategy\n");
                onenas->finalize_generation(gen, current_validation_inputs, current_validation_outputs,
                                           current_test_inputs, current_test_outputs);
            }

            Log::info("=== Generation %d Finalization Complete ===\n", gen);
            Log::info("Received %d elite genomes from finalize_generation\n", (int32_t)elite_genomes.size());

            // Update episode priorities (PER)
            if (current_online_series->get_training_method().compare("PER") == 0) {
                Log::info("Training method is PER - updating episode priorities with elite genomes\n");
                current_online_series->update_episode_priorities(elite_genomes, gen);
                current_online_series->write_priorities_to_csv(gen, get_stats_directory());
            }

            for (RNN_Genome* genome : elite_genomes) {
                if (genome != NULL) {
                    delete genome;
                }
            }
            elite_genomes.clear();

            onenas->update_log();

            // Evaluate on A at key checkpoints
            bool should_eval = false;

            // End of Phase A (baseline)
            if (gen == generations_a - 1) {
                should_eval = true;
            }

            // During Phase B: evaluate at eval_frequency intervals
            if (gen >= generations_a && (gen - generations_a) % eval_frequency == 0) {
                should_eval = true;
            }

            // End of Phase B (final)
            if (gen == total_generation - 1) {
                should_eval = true;
            }

            if (should_eval) {
                evaluate_on_distribution_a(gen, current_phase, eval_inputs_a, eval_outputs_a);
            }

            Log::log_memory_usage("Generation " + std::to_string(gen) + " end");
            Log::info("Generation %d finished\n", gen);
        }
    }

    // Cleanup
    if (rank == 0) {
        close_csv_files();

        Log::log_memory_usage("Before cleanup");
        delete onenas;
        delete online_series_a;
        delete online_series_b;
        delete weight_update_method;
        Log::log_memory_usage("After cleanup");
    }

    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets_a;
    delete time_series_sets_b;
    delete time_series_sets_eval;

    time_series_inputs_a.clear();
    time_series_outputs_a.clear();
    time_series_inputs_b.clear();
    time_series_outputs_b.clear();
    eval_inputs_a.clear();
    eval_outputs_a.clear();
    time_series_inputs.clear();
    time_series_outputs.clear();

    return 0;
}
