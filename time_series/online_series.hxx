#ifndef ONLINE_SERIES_HXX
#define ONLINE_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

#include <unordered_set>
using std::unordered_set;

#include <random>
using std::normal_distribution;
using std::default_random_engine;
using std::mt19937;
using std::random_device;

#include "time_series_episode.hxx"

// Forward declarations
class RNN_Genome;

class OnlineSeries {
    private:
        // Episode management
        vector<TimeSeriesEpisode*> episodes;
        
        // Core configuration
        int32_t total_num_sets;
        int32_t sequence_length;
        int32_t current_index; // current index = current_generation + num_training_sets
        vector< int32_t > avalibale_training_index;
        int32_t num_training_sets;
        int32_t num_validation_sets;
        int32_t num_test_sets;
        string get_training_data_method;
        
        // RNG for sampling (seeded once from hardware entropy)
        mt19937 rng;

        // Stratified replay: total number of bins
        int32_t num_bins;
        
    public:
        OnlineSeries(int32_t _num_sets, const vector<string> &arguments);
        ~OnlineSeries();

        // Episode management methods
        void add_episode(TimeSeriesEpisode* episode);
        void initialize_episodes(const vector<vector<vector<double>>>& inputs, const vector<vector<vector<double>>>& outputs);
        TimeSeriesEpisode* get_episode(int32_t episode_id);
        void print_episode_stats();
        
        // Core sampling methods
        void shuffle_data();
        void uniform_random_sample_index(vector<int32_t>& training_index);
        void sliding_window_sample_index(vector<int32_t>& training_index);
        void stratified_sample_index(vector<int32_t>& training_index);
        void set_current_index(int32_t _current_gen);
        void get_online_arguments(const vector<string> &arguments);

        // Stratified replay bin initialization
        void initialize_bins(int32_t num_episodes_a, int32_t num_files_a, int32_t num_episodes_b, int32_t num_files_b);

        // Core interface methods
        vector<int32_t> get_training_index(vector<int32_t>& training_index);
        vector<int32_t> get_validation_index(vector<int32_t>& validation_index);
        int32_t get_test_index();

        // Getter for training data method
        string get_training_method() const { return get_training_data_method; }

        int32_t get_max_generation();
};

#endif