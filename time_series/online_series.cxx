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
using std::sort;

#include <map>
using std::map;

#include <random>
using std::mt19937;
using std::random_device;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "online_series.hxx"

OnlineSeries::OnlineSeries(const int32_t _total_num_sets, const vector<string> &arguments)
    : rng(random_device{}())
{
    total_num_sets = _total_num_sets;
    current_index = 0;
    num_bins = 0;
    get_online_arguments(arguments);
    num_test_sets = 1;
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
}

void OnlineSeries::set_current_index(int32_t _current_gen) {
    //current index is the beginning of validation index
    current_index = _current_gen + num_training_sets;
    Log::debug("current generation is %d, current index is %d\n", _current_gen, current_index);
}

void OnlineSeries::shuffle_data() {
    avalibale_training_index.clear();

    for (int32_t i = 0; i < current_index; i++) {
        avalibale_training_index.push_back(i);
    }

    shuffle(avalibale_training_index.begin(), avalibale_training_index.end(), rng);
}

void OnlineSeries::uniform_random_sample_index(vector<int32_t>& training_index) {
    shuffle_data();
    training_index.clear();
    int32_t sample_count = std::min(num_training_sets, (int32_t)avalibale_training_index.size());
    for (int32_t i = 0; i < sample_count; i++) {
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

void OnlineSeries::stratified_sample_index(vector<int32_t>& training_index) {
    training_index.clear();

    // Group available episodes (0 to current_index-1) by bin_id
    map<int32_t, vector<int32_t>> bin_to_episodes;
    for (int32_t i = 0; i < current_index; i++) {
        TimeSeriesEpisode* episode = get_episode(i);
        if (episode != NULL) {
            bin_to_episodes[episode->get_bin_id()].push_back(i);
        }
    }

    int32_t total_available = current_index;
    int32_t total_to_sample = std::min(num_training_sets, total_available);

    if (bin_to_episodes.empty()) {
        Log::warning("Stratified: No bins available, falling back to uniform\n");
        uniform_random_sample_index(training_index);
        return;
    }

    // Proportional allocation with largest-remainder method (Hamilton's method).
    // This guarantees that each bin's allocation differs from its exact
    // proportional share by at most one sample, which is the mathematically
    // optimal rounding for proportional representation.
    struct BinAllocation {
        int32_t bin_id;
        int32_t allocation;
        double remainder;
    };

    vector<BinAllocation> allocations;
    int32_t allocated_so_far = 0;

    for (auto& pair : bin_to_episodes) {
        int32_t bin_id = pair.first;
        int32_t bin_size = (int32_t)pair.second.size();
        double exact = (double)bin_size / total_available * total_to_sample;
        int32_t floor_alloc = (int32_t)exact;
        double rem = exact - floor_alloc;

        // Cap allocation at bin size (can't sample more than available)
        floor_alloc = std::min(floor_alloc, bin_size);

        allocations.push_back({bin_id, floor_alloc, rem});
        allocated_so_far += floor_alloc;
    }

    // Distribute remaining slots by largest remainder
    int32_t remaining = total_to_sample - allocated_so_far;
    if (remaining > 0) {
        // Sort by remainder descending
        sort(allocations.begin(), allocations.end(),
             [](const BinAllocation& a, const BinAllocation& b) {
                 return a.remainder > b.remainder;
             });

        for (int32_t i = 0; i < remaining && i < (int32_t)allocations.size(); i++) {
            int32_t bin_id = allocations[i].bin_id;
            int32_t bin_size = (int32_t)bin_to_episodes[bin_id].size();
            if (allocations[i].allocation < bin_size) {
                allocations[i].allocation++;
            }
        }
    }

    // Sample from each bin with random shuffling
    for (auto& alloc : allocations) {
        vector<int32_t>& candidates = bin_to_episodes[alloc.bin_id];
        shuffle(candidates.begin(), candidates.end(), rng);
        int32_t to_take = std::min(alloc.allocation, (int32_t)candidates.size());
        for (int32_t i = 0; i < to_take; i++) {
            training_index.push_back(candidates[i]);
        }

        if (to_take > 0) {
            Log::debug("Stratified: bin %d -> %d/%d episodes sampled (of %d available)\n",
                       alloc.bin_id, to_take, alloc.allocation, (int32_t)candidates.size());
        }
    }

    // Final shuffle so training order is randomized across bins
    shuffle(training_index.begin(), training_index.end(), rng);

    Log::info("Stratified: sampled %d episodes from %d bins (%d available)\n",
              (int32_t)training_index.size(), (int32_t)bin_to_episodes.size(), total_available);
}

void OnlineSeries::initialize_bins(int32_t num_episodes_a, int32_t num_files_a, int32_t num_episodes_b, int32_t num_files_b) {
    num_bins = num_files_a + num_files_b;

    // Assign bins to Distribution A episodes.
    // Episodes are contiguous per source file after slice_online_time_series,
    // so equal division approximates the per-file boundaries.
    for (int32_t i = 0; i < num_episodes_a && i < (int32_t)episodes.size(); i++) {
        int32_t bin = (int32_t)((int64_t)i * num_files_a / num_episodes_a);
        bin = std::min(bin, num_files_a - 1);
        episodes[i]->set_bin_id(bin);
    }

    // Assign bins to Distribution B episodes (bins numbered num_files_a .. num_bins-1)
    for (int32_t i = num_episodes_a; i < (int32_t)episodes.size(); i++) {
        int32_t idx = i - num_episodes_a;
        int32_t bin = num_files_a + (int32_t)((int64_t)idx * num_files_b / num_episodes_b);
        bin = std::min(bin, num_bins - 1);
        episodes[i]->set_bin_id(bin);
    }

    // Log bin assignments for reproducibility
    map<int32_t, int32_t> bin_counts;
    for (int32_t i = 0; i < (int32_t)episodes.size(); i++) {
        if (episodes[i] != NULL) {
            bin_counts[episodes[i]->get_bin_id()]++;
        }
    }
    Log::info("Stratified bins initialized: %d total bins (%d from A, %d from B)\n",
              num_bins, num_files_a, num_files_b);
    for (auto& pair : bin_counts) {
        Log::info("  Bin %d: %d episodes\n", pair.first, pair.second);
    }
}

vector<int32_t> OnlineSeries::get_training_index(vector<int32_t>& training_index) {

    if (get_training_data_method.compare("Uniform") == 0) {
        Log::info("getting historical data with uniform random sampling\n");
        uniform_random_sample_index(training_index);
    } else if (get_training_data_method.compare("SlidingWindow") == 0) {
        Log::info("getting historical data with sliding window (most recent episodes only)\n");
        sliding_window_sample_index(training_index);
    } else if (get_training_data_method.compare("Stratified") == 0) {
        Log::info("getting historical data with stratified replay\n");
        stratified_sample_index(training_index);
    } else {
        Log::error("Invalid training data method: %s\n", get_training_data_method.c_str());
        exit(1);
    }

    return training_index;
}

vector<int32_t> OnlineSeries::get_validation_index(vector<int32_t>& validation_index) {
    validation_index.clear();
    for (int32_t i = 0; i < num_validation_sets; i++) {
        validation_index.push_back(current_index + i);
    }
    return validation_index;
}

int32_t OnlineSeries::get_test_index() {
    return current_index + num_validation_sets;
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
        episodes.push_back(episode);
    }

    Log::info("Initialized %d episodes\n", num_episodes);
}

TimeSeriesEpisode* OnlineSeries::get_episode(int32_t episode_id) {
    if (episode_id >= 0 && episode_id < (int32_t)episodes.size() && episodes[episode_id] != NULL) {
        return episodes[episode_id];
    }
    return NULL;
}

void OnlineSeries::print_episode_stats() {
    Log::info("Episode Statistics:\n");
    Log::info("Total episodes: %d\n", (int32_t)episodes.size());
    Log::info("Training method: %s\n", get_training_data_method.c_str());
    if (num_bins > 0) {
        Log::info("Stratified bins: %d\n", num_bins);
    }
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
