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
string output_directory;

vector<vector<vector<double> > > time_series_inputs;
vector<vector<vector<double> > > time_series_outputs;
vector<int32_t> time_series_index;
int32_t generated_population_size;
int32_t number_islands;
int32_t total_generation;

/**
 * Checks if enough genomes have been generated for the current generation
 * 
 * @param current_generated_genomes The number of genomes generated so far
 * @return true if enough genomes have been generated, false otherwise
 */
bool has_generated_enough_genomes(int32_t current_generated_genomes) {
    // Get the current generated_population_size from the speciation strategy
    // in case it has been modified during execution
    OneNasIslandSpeciationStrategy* onenas_strategy = 
        dynamic_cast<OneNasIslandSpeciationStrategy*>(onenas->get_speciation_strategy());
    
    int32_t current_generated_population_size = generated_population_size; // Default fallback
    if (onenas_strategy != nullptr) {
        current_generated_population_size = onenas_strategy->get_generated_population_size();
        if (current_generated_population_size != generated_population_size) {
            Log::info("Master: Generated population size updated from %d to %d\n", 
                     generated_population_size, current_generated_population_size);
        }
    }
    
    return current_generated_genomes >= current_generated_population_size * number_islands;
}

/**
 * Get stats directory path
 */
string get_stats_directory() {
    return output_directory + "/stats";
}

/**
 * Initialize CSV files for logging training indices and validation/test indices
 */
void initialize_csv_files() {
    // Create output directory if it doesn't exist
    mkpath(output_directory.c_str(), 0777);
    
    // Create stats subdirectory
    string stats_dir = get_stats_directory();
    mkpath(stats_dir.c_str(), 0777);
    
    // Initialize training indices CSV file in stats directory
    string training_csv_path = stats_dir + "/training_indices.csv";
    training_indices_csv.open(training_csv_path.c_str(), ios::out);
    if (!training_indices_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", training_csv_path.c_str());
        return;
    }
    training_indices_csv << "genome_id,generation,training_indices\n";
    
    // Initialize validation/test indices CSV file in stats directory
    string validation_csv_path = stats_dir + "/validation_test_indices.csv";
    validation_test_indices_csv.open(validation_csv_path.c_str(), ios::out);
    if (!validation_test_indices_csv.is_open()) {
        Log::error("Failed to open %s for writing\n", validation_csv_path.c_str());
        return;
    }
    validation_test_indices_csv << "generation,validation_indices,test_index\n";
    
    Log::info("CSV files initialized successfully in %s\n", stats_dir.c_str());
}

/**
 * Close CSV files
 */
void close_csv_files() {
    if (training_indices_csv.is_open()) {
        training_indices_csv.close();
    }
    if (validation_test_indices_csv.is_open()) {
        validation_test_indices_csv.close();
    }
    Log::info("CSV files closed\n");
}

/**
 * Write training indices for a genome to CSV
 */
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
    training_indices_csv.flush(); // Ensure data is written immediately
}

/**
 * Write validation and test indices for a generation to CSV
 */
void write_validation_test_indices_to_csv(int32_t generation, const vector<int32_t>& validation_indices, const vector<int32_t>& test_indices) {
    if (!validation_test_indices_csv.is_open()) {
        Log::error("Validation/test indices CSV file is not open\n");
        return;
    }

    validation_test_indices_csv << generation << ",\"";
    for (size_t i = 0; i < validation_indices.size(); i++) {
        if (i > 0) validation_test_indices_csv << ";";
        validation_test_indices_csv << validation_indices[i];
    }
    validation_test_indices_csv << "\",";
    for (size_t i = 0; i < test_indices.size(); i++) {
        if (i > 0) validation_test_indices_csv << ";";
        validation_test_indices_csv << test_indices[i];
    }
    validation_test_indices_csv << "\n";
    validation_test_indices_csv.flush(); // Ensure data is written immediately
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
    // train_index contains episode IDs (original time series indices)
    for (int32_t i = 0; i < (int32_t)train_index.size(); i++) {
        int32_t episode_id = train_index[i];  // This is the original episode ID
        TimeSeriesEpisode* episode = online_series->get_episode(episode_id);
        if (episode != nullptr) {
            current_training_inputs.push_back(episode->get_inputs());
            current_training_outputs.push_back(episode->get_outputs());
            Log::debug("Worker: training episode ID: %d\n", episode_id);
        } else {
            Log::warning("Episode ID %d not found, falling back to legacy method\n", episode_id);
            // Fallback to legacy method using original time series arrays
            if (episode_id < (int32_t)time_series_inputs.size()) {
                current_training_inputs.push_back(time_series_inputs[episode_id]);
                current_training_outputs.push_back(time_series_outputs[episode_id]);
                Log::debug("Worker: training legacy index: %d\n", episode_id);
            } else {
                Log::error("Episode ID %d out of bounds for both episodes and legacy data\n", episode_id);
            }
        }
    }
    
    // validation_index contains episode IDs (original time series indices)
    for (int32_t i = 0; i < (int32_t)validation_index.size(); i++) {
        int32_t episode_id = validation_index[i];  // This is the original episode ID
        TimeSeriesEpisode* episode = online_series->get_episode(episode_id);
        if (episode != nullptr) {
            current_validation_inputs.push_back(episode->get_inputs());
            current_validation_outputs.push_back(episode->get_outputs());
            Log::debug("Worker: validation episode ID: %d\n", episode_id);
        } else {
            Log::warning("Episode ID %d not found for validation, falling back to legacy method\n", episode_id);  
            // Fallback to legacy method using original time series arrays
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
    const vector<int32_t>& test_indices,
    const vector<int32_t>& validation_index,
    vector<vector<vector<double>>>& current_test_inputs,
    vector<vector<vector<double>>>& current_test_outputs,
    vector<vector<vector<double>>>& current_validation_inputs,
    vector<vector<vector<double>>>& current_validation_outputs
) {
    // test_indices contains episode IDs - a single episode in the default mode, or one
    // episode per stock (all at the same window) in pooled panel mode
    for (int32_t i = 0; i < (int32_t)test_indices.size(); i++) {
        TimeSeriesEpisode* test_episode = online_series->get_episode(test_indices[i]);
        if (test_episode != nullptr) {
            current_test_inputs.push_back(test_episode->get_inputs());
            current_test_outputs.push_back(test_episode->get_outputs());
        } else {
            Log::error("Test episode ID %d not found in episodes\n", test_indices[i]);
            exit(1);
        }
    }

    // validation_index contains episode IDs (original time series indices)
    for (int32_t i = 0; i < (int32_t)validation_index.size(); i++) {
        int32_t episode_id = validation_index[i];  // This is the original episode ID
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
    Log::info("Current testing episode ID(s): %d test episodes, first ID %d\n", (int32_t)test_indices.size(), test_indices.empty() ? -1 : test_indices[0]);
}

void master(int32_t max_rank, OnlineSeries* online_series, int32_t current_generation) {
    // the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX int32_t: %d\n", numeric_limits<int32_t>::max());

    int32_t terminates_sent = 0;
    int32_t generated_genome = 0;
    int32_t evaluated_genome = 0;
    while (true) {
        // wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);
            if (!has_generated_enough_genomes(generated_genome)) {
                onenas_mutex.lock();
                RNN_Genome *genome = onenas->generate_genome();
                onenas_mutex.unlock();

                if (genome != NULL) {
                    // Master generates training indices using PER system
                    vector<int32_t> master_training_index;
                    online_series->get_training_index(master_training_index);
                    
                    // Attach training indices to genome before sending to worker
                    int32_t generation_id = genome->get_generation_id();
                    genome->set_training_indices(master_training_index);
                    
                    // Write training indices to CSV
                    write_training_indices_to_csv(generation_id, current_generation, master_training_index);
                    
                    Log::info("Master: Generated %d training indices for genome %d, sending to worker: %d\n", 
                             master_training_index.size(), generation_id, source);
                    Log::debug("sending genome to: %d\n", source);
                    send_genome_to(source, genome);

                    //delete this genome as it will not be used again
                    delete genome;
                    generated_genome ++;
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

            // Training history was already recorded when genome was generated
            // No need to extract and re-add training indices here

            onenas_mutex.lock();
            onenas->insert_genome(genome);
            onenas_mutex.unlock();

            // delete the genome as it won't be used again, a copy was inserted
            delete genome;
            evaluated_genome++;
            // this genome will be deleted if/when removed from population
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

            vector< vector< vector<double> > > current_training_inputs;
            vector< vector< vector<double> > > current_training_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            // Use training indices provided by master (attached to genome)
            vector<int32_t> train_index = genome->get_training_indices();
            vector<int32_t> validation_index;

            // Still generate validation indices locally (these don't need score-based prioritization)
            online_series->get_validation_index(validation_index);

            Log::info("Worker %d: Using %d training indices provided by master for genome %d\n", 
                     rank, train_index.size(), genome->get_generation_id());

            // Use episode-based data population if available, otherwise use legacy method
            populate_current_time_series_data(
                online_series, train_index, validation_index, 
                current_training_inputs, current_training_outputs, 
                current_validation_inputs, current_validation_outputs
            );

            //have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(current_training_inputs, current_training_outputs, current_validation_inputs, current_validation_outputs, weight_update_method);
            genome->evaluate_online(current_validation_inputs, current_validation_outputs);
            Log::release_id(log_id);

            // Training indices were provided by master and used for training
            // No training history tracking needed with PER system

            // go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

/**
 * Write normalized and sliced time series data to separate CSV files
 * The data will be in normalized form (0-1 range) to match prediction files
 */
void write_sliced_files(const vector<vector<vector<double>>>& inputs, 
                       const vector<vector<vector<double>>>& outputs,
                       const vector<string>& input_parameter_names,
                       const vector<string>& output_parameter_names,
                       const string& base_directory) {
    
    // Create the directory if it doesn't exist
    mkpath(base_directory.c_str(), 0777);
    
    Log::info("Writing %d sliced files to directory: %s\n", inputs.size(), base_directory.c_str());
    
    for (int32_t slice_idx = 0; slice_idx < (int32_t)inputs.size(); slice_idx++) {
        string filename = base_directory + "/generation_" + to_string(slice_idx) + ".csv";
        ofstream outfile(filename);
        
        if (!outfile.is_open()) {
            Log::error("Failed to open file %s for writing\n", filename.c_str());
            continue;
        }
        
        // Write header with only input parameter names (output parameters are already in input)
        bool first_column = true;
        for (const string& param : input_parameter_names) {
            if (!first_column) outfile << ",";
            outfile << param;
            first_column = false;
        }
        outfile << "\n";
        
        // Get the number of time steps (should be the same for inputs and outputs)
        int32_t time_steps = inputs[slice_idx][0].size();
        int32_t num_input_params = inputs[slice_idx].size();
        // int32_t num_output_params = outputs[slice_idx].size();
        
        // Write data rows (each row is a time step) - only input values since output parameters are already in input
        for (int32_t t = 0; t < time_steps; t++) {
            bool first_value = true;
            
            // Write only input values for this time step (normalized)
            for (int32_t param = 0; param < num_input_params; param++) {
                if (!first_value) outfile << ",";
                outfile << inputs[slice_idx][param][t];
                first_value = false;
            }
            
            outfile << "\n";
        }
        
        outfile.close();
        Log::info("Written sliced file: %s with %d time steps (normalized values)\n", filename.c_str(), time_steps);
    }
}

int main(int argc, char** argv) {
    std::cout << "Starting ONENAS MPI Program" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "MPI initialized" << std::endl;
    int32_t rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
    std::cout << "MPI got rank " << rank << " and max rank " << max_rank << std::endl;
    arguments = vector<string>(argv, argv + argc);
    std::cout << "Received arguments!" << std::endl;

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);
    std::cout << "Initialized log!" << std::endl;

    // Load required parameters
    get_argument(arguments, "--number_islands", true, number_islands);
    get_argument(arguments, "--generated_population_size", true, generated_population_size);
    get_argument(arguments, "--output_directory", true, output_directory);

    // Gens-per-tick knob: run R rounds of generate+train+insert per data tick (window advance).
    // The window clock (set_current_index) and elite finalization keep their once-per-tick cadence.
    int32_t rounds_per_generation = 1;
    get_argument(arguments, "--rounds_per_generation", false, rounds_per_generation);
    if (rounds_per_generation < 1) {
        Log::fatal("--rounds_per_generation must be >= 1, got %d\n", rounds_per_generation);
        exit(1);
    }
    Log::info("Rounds per generation (genome-generation rounds per data tick): %d\n", rounds_per_generation);

    // Log::info("ONENAS will generate %d genomes per generation\n", generated_population_size * number_islands);
    Log::info("Output directory: %s\n", output_directory.c_str());

    TimeSeriesSets* time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    slice_online_time_series(
        arguments, time_series_sets, time_series_inputs, time_series_outputs
    );
    Log::major_divider(Log::INFO, "Sliced time series!");
    Log::info("Time series inputs shape: %d, %d, %d \n", time_series_inputs.size(), time_series_inputs[0].size(), time_series_inputs[0][0].size());
    Log::info("Time series outputs shape: %d, %d, %d \n", time_series_outputs.size(), time_series_outputs[0].size(), time_series_outputs[0][0].size());
    
    // Check if user wants to write sliced files and write them if requested
    if (argument_exists(arguments, "--write_sliced_files") && rank == 0) {
        string sliced_files_directory = output_directory + "/sliced_data";
        vector<string> input_parameter_names = time_series_sets->get_input_parameter_names();
        vector<string> output_parameter_names = time_series_sets->get_output_parameter_names();
        
        write_sliced_files(time_series_inputs, time_series_outputs, 
                          input_parameter_names, output_parameter_names, 
                          sliced_files_directory);
        Log::info("Sliced files written to: %s (normalized values)\n", sliced_files_directory.c_str());
    }
    
    int32_t num_sets = time_series_inputs.size();
    Log::info("Time series number of sets after slicing: %d\n", num_sets);
    OnlineSeries* online_series = new OnlineSeries(num_sets, arguments);

    int32_t max_generation = online_series->get_max_generation(); // default to the number of sets
    total_generation = max_generation;
    get_argument(arguments, "--total_generation", false, total_generation); 
    if (total_generation > max_generation) {
        Log::error("Total generation is greater than the number of sets, setting total generation to %d\n", max_generation);
        total_generation = max_generation;
    }
    Log::info("Total generation is set to: %d\n", total_generation);

    
    // Initialize episode management system
    Log::info("Initializing episode management system\n");
    online_series->initialize_episodes(time_series_inputs, time_series_outputs);
    online_series->print_episode_stats();
    
    // Optional: Clear global vectors to save memory if episodes are being used
    // time_series_inputs.clear();
    // time_series_outputs.clear();
    Log::info("Episode management initialization complete\n");

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);
    Log::major_divider(Log::INFO, "Created weight update method!");

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);
    if (weight_rules == NULL) {
        Log::fatal("ERROR in onenas mpi: Failed to create weight rules, this should not happen\n");
        exit(1);
    }
    Log::major_divider(Log::INFO, "Created weight rules!");

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);
    Log::major_divider(Log::INFO, "Created seed genome!");

    Log::clear_rank_restriction();

    if (rank == 0) {
        onenas = generate_onenas_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);
        Log::major_divider(Log::INFO, "Created ONENAS!");
        
        // Initialize CSV files for logging (only on master process)
        initialize_csv_files();
    }

    for (int32_t  current_generation = 0; current_generation < total_generation; current_generation ++) {
        // The window clock advances once per generation (data tick), regardless of how many
        // genome-generation rounds are run within the tick.
        online_series->set_current_index(current_generation);

        // Run R rounds of generate+train+insert per data tick. Each round is a full
        // master/worker exchange (the master terminates the workers at the end of a round, so
        // both sides re-enter their loops for the next round). Validation/test indices and
        // elite finalization happen ONCE per tick, after all R rounds.
        for (int32_t round = 0; round < rounds_per_generation; round++) {
            if (rank ==0) {
                Log::major_divider(Log::INFO, "New generation");
                Log::info("Current generation: %d (round %d of %d)\n", current_generation, round + 1, rounds_per_generation);

                // Track memory usage at start of generation
                Log::log_memory_usage("Generation " + std::to_string(current_generation) + " start");

                master(max_rank, online_series, current_generation);
            } else {
                worker(rank, online_series);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0) {
            Log::minor_divider(Log::INFO);
            vector <int32_t> validation_index;
            online_series->get_validation_index(validation_index);
            vector<int32_t> test_indices;
            online_series->get_test_indices(test_indices);

            // Write validation and test indices to CSV
            write_validation_test_indices_to_csv(current_generation, validation_index, test_indices);

            vector< vector< vector<double> > > current_test_inputs;
            vector< vector< vector<double> > > current_test_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            // Populate test and validation data from episodes
            populate_test_and_validation_data(
                online_series, test_indices, validation_index,
                current_test_inputs, current_test_outputs,
                current_validation_inputs, current_validation_outputs
            );

            // Finalize generation and get elite genomes for PER updates
            OneNasIslandSpeciationStrategy* onenas_strategy = dynamic_cast<OneNasIslandSpeciationStrategy*>(onenas->get_speciation_strategy());
            vector<RNN_Genome*> elite_genomes;
            if (onenas_strategy != nullptr) {
                elite_genomes = onenas_strategy->finalize_generation_with_genomes(current_generation, current_validation_inputs, current_validation_outputs, current_test_inputs, current_test_outputs);
            } else {
                Log::error("Failed to cast speciation strategy to OneNasIslandSpeciationStrategy\n");
                // Fallback to original method
                onenas->finalize_generation(current_generation, current_validation_inputs, current_validation_outputs, current_test_inputs, current_test_outputs);
            }
            
            Log::info("=== MPI Generation %d Finalization Complete ===\n", current_generation);
            Log::info("Received %d elite genomes from finalize_generation\n", (int32_t)elite_genomes.size());
            
            // Update episode priorities using all elite genomes (only for PER method)
            if (online_series->get_training_method().compare("PER") == 0) {
                Log::info("Training method is PER - updating episode priorities with elite genomes\n");
                online_series->update_episode_priorities(elite_genomes, current_generation);
                
                // Write priority statistics to CSV (PER method only). This dump is one row
                // per episode, so on a pooled panel it is several MB per generation; write it
                // periodically rather than every generation to keep the diagnostic without
                // saturating shared-filesystem I/O.
                if (current_generation % 25 == 0) {
                    online_series->write_priorities_to_csv(current_generation, get_stats_directory());
                }
            } else {
                Log::debug("Training method is %s - skipping priority updates\n", online_series->get_training_method().c_str());
            }
            
            // Clean up elite genome copies (they were created in finalize_generation)
            for (RNN_Genome* genome : elite_genomes) {
                if (genome != NULL) {
                    delete genome;
                }
            }
            elite_genomes.clear();
            
            Log::info("MPI Generation %d priority update complete\n", current_generation);
            
            onenas->update_log();
            
            // Track memory usage at end of generation
            Log::log_memory_usage("Generation " + std::to_string(current_generation) + " end");
            
            Log::info("Generation %d finished\n", current_generation);
        }
        
    }
    
    // Close CSV files (only on master process)
    if (rank == 0) {
        close_csv_files();
        
        // Clean up memory on master process
        Log::log_memory_usage("Before cleanup");
        delete onenas;
        delete online_series;
        delete weight_update_method;
        Log::log_memory_usage("After cleanup");
    }
    
    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    
    // Clear global vectors to free memory
    time_series_inputs.clear();
    time_series_outputs.clear();
    time_series_index.clear();
    
    return 0;
}
