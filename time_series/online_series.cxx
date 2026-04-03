#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <iostream>
using std::ios;

#include <sys/stat.h>

using std::min;

#include <algorithm> 
using std::shuffle;
using std::min_element;

#include <random>
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;

#include <unordered_set>
using std::unordered_set;

#include <cmath>
using std::pow;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"

#include "online_series.hxx"

OnlineSeries::OnlineSeries(const int32_t _total_num_sets,const vector<string> &arguments)
    : rng(random_device{}())
{
    total_num_sets = _total_num_sets;
    current_index = 0;
    get_online_arguments(arguments);
    num_test_sets = 1;
    // Initialize episodes vector
    episodes.reserve(total_num_sets);

    // Allow deterministic seeding for reproducibility across trials.
    // When --seed is provided, the RNG produces identical sequences for
    // identical seeds, enabling paired comparisons across conditions.
    int32_t seed = -1;
    if (get_argument(arguments, "--seed", false, seed) && seed >= 0) {
        rng.seed(seed);
        Log::info("OnlineSeries: RNG seeded deterministically with seed=%d\n", seed);
    } else {
        Log::info("OnlineSeries: RNG seeded from hardware entropy (non-deterministic)\n");
    }
}

OnlineSeries::~OnlineSeries() {
    // Clean up episodes manually
    for (int32_t i = 0; i < (int32_t)episodes.size(); i++) {
        if (episodes[i] != NULL) {
            delete episodes[i];
            episodes[i] = NULL;
        }
    }
    episodes.clear();
}

void OnlineSeries::get_online_arguments(const vector<string> &arguments) {
    get_argument(arguments, "--num_validation_sets", true, num_validation_sets);
    get_argument(arguments, "--num_training_sets", true, num_training_sets);
    get_argument(arguments, "--get_train_data_by", true, get_training_data_method);
    
    // PER parameters
    per_alpha = 0.6; // default prioritization strength
    get_argument(arguments, "--per_alpha", false, per_alpha);
    
    per_lambda = 0.01; // default temporal decay rate
    get_argument(arguments, "--per_lambda", false, per_lambda);
    
    per_epsilon = 1e-8; // default small constant for priority calculation
    get_argument(arguments, "--per_epsilon", false, per_epsilon);
}

void OnlineSeries::set_current_index(int32_t _current_gen) {
    //current index is the begining of validation index
    current_index = _current_gen + num_training_sets;
    Log::debug("current generation is %d, current index is %d\n", _current_gen, current_index);
}

void OnlineSeries::shuffle_data() {
    // current index is the end of available training index
    // avalibale_training_index contains episode IDs (original time series indices)
    avalibale_training_index.clear();

    for (int32_t i = 0; i < current_index; i++) {
        avalibale_training_index.push_back(i);  // i is the episode ID (original time series index)
    }

    shuffle(avalibale_training_index.begin(), avalibale_training_index.end(), rng);
}

void OnlineSeries::uniform_random_sample_index(vector<int32_t>& training_index) {
    shuffle_data();
    training_index.clear();
    for (int32_t i = 0; i < num_training_sets; i++) {
        training_index.push_back(avalibale_training_index[i]);
    }
}

void OnlineSeries::sliding_window_sample_index(vector<int32_t>& training_index) {
    // Only sample from the most recent num_training_sets episodes (no historical replay).
    // This serves as a control to demonstrate that forgetting occurs without replay.
    int32_t window_start = std::max(0, current_index - num_training_sets);

    vector<int32_t> window_episodes;
    for (int32_t i = window_start; i < current_index; i++) {
        window_episodes.push_back(i);
    }

    shuffle(window_episodes.begin(), window_episodes.end(), rng);

    training_index.clear();
    int32_t sample_count = std::min(num_training_sets, (int32_t)window_episodes.size());
    for (int32_t i = 0; i < sample_count; i++) {
        training_index.push_back(window_episodes[i]);
    }
}

void OnlineSeries::prioritized_experience_replay(vector<int32_t>& training_index) {
    shuffle_data();
    training_index.clear();

    int32_t current_generation = current_index - num_training_sets; // Calculate current generation

    // Calculate priorities for all available episodes
    vector<double> priorities;
    priorities.reserve(avalibale_training_index.size());
    
    for (int32_t episode_id : avalibale_training_index) {
        TimeSeriesEpisode* episode = get_episode(episode_id);
        if (episode != NULL) {
            double priority = episode->calculate_priority(current_generation, per_alpha, per_lambda, per_epsilon);
            priorities.push_back(priority);
        } else {
            // Fallback priority for episodes not found
            priorities.push_back(1.0);
            Log::warning("Episode %d not found, using default priority\n", episode_id);
        }
    }

    // Apply alpha exponentiation for sampling probability: P(i) = p_i^alpha / sum(p_j^alpha)
    vector<double> sampling_weights;
    sampling_weights.reserve(priorities.size());
    
    for (double priority : priorities) {
        double weight = pow(priority, per_alpha);
        sampling_weights.push_back(weight);
    }

    Log::info("PER: Using priority-based sampling with alpha=%.3f, lambda=%.3f\n", per_alpha, per_lambda);

    // Use the class member RNG for reproducibility
    std::discrete_distribution<> dist(sampling_weights.begin(), sampling_weights.end());

    // Sample without replacement
    std::unordered_set<int32_t> seen;
    while ((int32_t)training_index.size() < num_training_sets) {
        int32_t sampled_idx = dist(rng);  // Index into avalibale_training_index
        int32_t actual_episode_id = avalibale_training_index[sampled_idx];
        
        if (seen.find(actual_episode_id) == seen.end()) {
            training_index.push_back(actual_episode_id);
            seen.insert(actual_episode_id);
            
            // Debug info for first few selections
            if (training_index.size() <= 3) {
                Log::debug("Selected episode %d (priority: %.6f, weight: %.6f)\n", 
                          actual_episode_id, priorities[sampled_idx], sampling_weights[sampled_idx]);
            }
        }
    }
}

vector<int32_t> OnlineSeries::get_training_index(vector<int32_t>& training_index) {

    if (get_training_data_method.compare("Uniform") == 0) {
        Log::info("getting historical data with uniform random sampling\n");
        uniform_random_sample_index(training_index);
    } else if (get_training_data_method.compare("SlidingWindow") == 0) {
        Log::info("getting historical data with sliding window (most recent episodes only)\n");
        sliding_window_sample_index(training_index);
    } else if (get_training_data_method.compare("PER") == 0) {
        Log::info("getting historical data with Prioritized Experience Replay (PER)\n");
        prioritized_experience_replay(training_index);
    } else {
        Log::error("Invalid training data method: %s\n", get_training_data_method.c_str());
        exit(1);
    }

    return training_index;
}

vector< int32_t > OnlineSeries::get_validation_index(vector<int32_t>& validation_index) {
    validation_index.clear();
    for (int32_t i = 0; i < num_validation_sets; i++) {
        validation_index.push_back(current_index + i);
    }
    return validation_index;
}

int32_t OnlineSeries::get_test_index() {
    return current_index + num_validation_sets;
}

void OnlineSeries::update_episode_priorities(const vector<RNN_Genome*>& elite_genomes, int32_t current_generation) {
    // Skip priority updates for non-PER methods
    if (get_training_data_method.compare("PER") != 0) {
        Log::debug("Training data method is '%s' - skipping priority updates\n", get_training_data_method.c_str());
        return;
    }
    
    if (elite_genomes.empty()) {
        Log::warning("PER: No elite genomes provided, cannot update episode priorities\n");
        return;
    }
    
    Log::info("PER: Processing %d elite genomes for priority updates\n", (int32_t)elite_genomes.size());
    
    // Calculate statistics from all elite genomes
    double total_mse = 0.0;
    double best_mse = 1e10;
    double worst_mse = 0.0;
    int32_t valid_genomes = 0;
    
    for (const RNN_Genome* genome : elite_genomes) {
        if (genome != NULL) {
            double mse = genome->get_best_validation_mse();
            total_mse += mse;
            best_mse = std::min(best_mse, mse);
            worst_mse = std::max(worst_mse, mse);
            valid_genomes++;
            
            // Log first few genomes for debugging
            if (valid_genomes <= 3) {
                Log::debug("PER: Elite genome %d: ID=%d, MSE=%.6f\n", 
                          valid_genomes, genome->get_generation_id(), mse);
            }
        }
    }
    
    if (valid_genomes == 0) {
        Log::warning("PER: No valid elite genomes found\n");
        return;
    }
    
    double avg_mse = total_mse / valid_genomes;
    
    Log::info("PER: Elite genome statistics - Count: %d, Best MSE: %.6f, Avg MSE: %.6f, Worst MSE: %.6f\n",
             valid_genomes, best_mse, avg_mse, worst_mse);
    
    // Update ALL episodes that are newly available this generation
    // In online learning, episode i becomes available at generation i
    int32_t new_episode_id = current_generation;
    
    if (new_episode_id < total_num_sets) {
        TimeSeriesEpisode* new_episode = get_episode(new_episode_id);
        if (new_episode != NULL) {
            double old_mse = new_episode->get_validation_mse();
            
            // Use best MSE from elite genomes as the episode's initial performance estimate
            new_episode->set_validation_mse(best_mse);
            
            // Make sure availability generation is set correctly
            if (new_episode->get_availability_generation() != current_generation) {
                new_episode->set_availability_generation(current_generation);
            }
            
            Log::info("PER: Updated new episode %d - MSE: %.6f -> %.6f, availability_gen: %d\n", 
                     new_episode_id, old_mse, best_mse, current_generation);
            
            // Calculate and log new priority for this episode
            double new_priority = new_episode->calculate_priority(current_generation, per_alpha, per_lambda, per_epsilon);
            Log::info("PER: Episode %d initial priority: %.6f (MSE: %.6f, avail_gen: %d, current_gen: %d)\n",
                     new_episode_id, new_priority, best_mse, current_generation, current_generation);
        } else {
            Log::warning("PER: New episode %d not found for priority update\n", new_episode_id);
        }
    }
    
    // OPTION: Also update episodes that were likely used for training this generation
    // This would be episodes from the most recent training batch
    // We can estimate which episodes were used based on the current training selection
    
    // Get the episodes that would have been selected for training in this generation
    vector<int32_t> likely_training_episodes;
    int32_t training_start = std::max(0, current_generation - num_training_sets);
    int32_t training_end = current_generation;
    
    int32_t episodes_updated = 0;
    for (int32_t episode_id = training_start; episode_id < training_end && episode_id < total_num_sets; episode_id++) {
        TimeSeriesEpisode* episode = get_episode(episode_id);
        if (episode != NULL) {
            // Update MSE based on how this episode might have contributed to the elite genomes
            // Use average MSE for episodes that were used in training
            double old_mse = episode->get_validation_mse();
            
            // More conservative update - blend old and new MSE values
            double blended_mse = 0.7 * old_mse + 0.3 * avg_mse;
            episode->set_validation_mse(blended_mse);
            
            episodes_updated++;
            
            if (episodes_updated <= 3) {  // Log first few updates
                Log::debug("PER: Updated training episode %d - MSE: %.6f -> %.6f\n", 
                          episode_id, old_mse, blended_mse);
            }
        }
    }
    
    if (episodes_updated > 0) {
        Log::info("PER: Updated %d training episodes with blended MSE (avg elite MSE: %.6f)\n", 
                 episodes_updated, avg_mse);
    }
    
    // Log some statistics about current episode priorities
    log_priority_statistics(current_generation);
}

void OnlineSeries::log_priority_statistics(int32_t current_generation) {
    if (get_training_data_method.compare("PER") != 0) return;
    
    Log::info("PER: Priority statistics for generation %d:\n", current_generation);
    
    double total_priority = 0.0;
    double min_priority = 1e10;
    double max_priority = 0.0;
    int32_t available_episodes = 0;
    
    // Calculate statistics for available episodes only
    int32_t current_index_gen = current_generation + num_training_sets;
    
    for (int32_t episode_id = 0; episode_id < current_index_gen && episode_id < total_num_sets; episode_id++) {
        TimeSeriesEpisode* episode = get_episode(episode_id);
        if (episode != NULL) {
            double priority = episode->calculate_priority(current_generation, per_alpha, per_lambda, per_epsilon);
            total_priority += priority;
            min_priority = std::min(min_priority, priority);
            max_priority = std::max(max_priority, priority);
            available_episodes++;
            
            // Log detailed info for first few episodes
            if (episode_id < 3) {
                Log::info("PER:   Episode %d: MSE=%.6f, priority=%.6f, avail_gen=%d\n",
                         episode_id, episode->get_validation_mse(), priority, episode->get_availability_generation());
            }
        }
    }
    
    if (available_episodes > 0) {
        double avg_priority = total_priority / available_episodes;
        Log::info("PER: Available episodes: %d, Avg priority: %.6f, Min: %.6f, Max: %.6f\n",
                 available_episodes, avg_priority, min_priority, max_priority);
    }
}

void OnlineSeries::write_priorities_to_csv(int32_t generation, const string& stats_directory) {
    // Skip CSV writing for non-PER methods
    if (get_training_data_method.compare("PER") != 0) {
        Log::debug("Training data method is '%s' - skipping priority CSV writing\n", get_training_data_method.c_str());
        return;
    }
    
    string csv_file_path = stats_directory + "/episode_priorities.csv";
    
    // Check if file exists to determine if we need to write header
    bool file_exists = false;
    {
        std::ifstream test_file(csv_file_path);
        file_exists = test_file.good();
    }
    
    ofstream csv_file(csv_file_path, ios::app);
    
    if (!csv_file.is_open()) {
        Log::error("Failed to open %s for writing\n", csv_file_path.c_str());
        return;
    }
    
    // Write header if file is new
    if (!file_exists) {
        csv_file << "generation";
        for (int32_t episode_id = 0; episode_id < total_num_sets; episode_id++) {
            csv_file << ",episode_" << (episode_id + 1) << "_mse,episode_" << (episode_id + 1) << "_priority";
        }
        csv_file << "\n";
    }
    
    // Write generation number as first column
    csv_file << generation;
    
    int32_t current_generation = generation;
    
    // Write MSE and priority for all episodes
    for (int32_t episode_id = 0; episode_id < total_num_sets; episode_id++) {
        TimeSeriesEpisode* episode = get_episode(episode_id);
        if (episode != NULL) {
            double mse = episode->get_validation_mse();
            double priority = episode->calculate_priority(current_generation, per_alpha, per_lambda, per_epsilon);
            csv_file << "," << mse << "," << priority;
        } else {
            csv_file << ",1.0,1.0"; // Default values
        }
    }
    
    csv_file << "\n";
    csv_file.close();
    
    Log::info("Written priorities for generation %d to %s\n", generation, csv_file_path.c_str());
}

// Episode management methods

void OnlineSeries::add_episode(TimeSeriesEpisode* episode) {
    episodes.push_back(episode);
}

void OnlineSeries::initialize_episodes(const vector<vector<vector<double>>>& inputs, const vector<vector<vector<double>>>& outputs) {
    // Clean up any existing episodes first
    for (int32_t i = 0; i < (int32_t)episodes.size(); i++) {
        if (episodes[i] != NULL) {
            delete episodes[i];
            episodes[i] = NULL;
        }
    }
    episodes.clear();
    
    int32_t num_episodes = min(inputs.size(), outputs.size());
    
    for (int32_t i = 0; i < num_episodes; i++) {
        TimeSeriesEpisode* episode = new TimeSeriesEpisode(i, inputs[i], outputs[i]);
        // Set availability generation - episodes become available when they can be used for training
        episode->set_availability_generation(i);
        // Initialize with default MSE - will be updated when genomes are evaluated
        episode->set_validation_mse(1.0);
        episodes.push_back(episode);
    }
    
    Log::info("Initialized %d episodes with PER priority system\n", num_episodes);
}

TimeSeriesEpisode* OnlineSeries::get_episode(int32_t episode_id) {
    if (episode_id >= 0 && episode_id < (int32_t)episodes.size() && episodes[episode_id] != NULL) {
        return episodes[episode_id];
    }
    return NULL;
}

void OnlineSeries::print_episode_stats() {
    Log::info("Episode Statistics (PER System):\n");
    Log::info("Total episodes: %d\n", (int32_t)episodes.size());
    Log::info("PER Parameters: alpha=%.3f, lambda=%.3f, epsilon=%.8f\n", per_alpha, per_lambda, per_epsilon);
    for (int32_t i = 0; i < min(5, (int32_t)episodes.size()); i++) {
        if (episodes[i] != NULL) {
            episodes[i]->print_stats();
        }
    }
}

int32_t OnlineSeries::get_max_generation() {
    int32_t max_generation = total_num_sets - num_training_sets - num_validation_sets - num_test_sets;
    return max_generation;
}