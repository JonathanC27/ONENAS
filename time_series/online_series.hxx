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

#include "time_series_episode.hxx"

// Forward declarations
class RNN_Genome;

class OnlineSeries {
    private:
        // Episode management - PER approach
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

        // Pooled panel mode: input is S date-aligned, equal-length series (stocks).
        // Episode identity = (stock s, window w); flat episode id = s * num_windows + w.
        // Windows at the same index across stocks are contemporaneous.
        bool pooled_panel;
        int32_t num_stocks;            // S: number of panel series (training files)
        int32_t num_windows;           // W: windows per stock
        int32_t num_training_windows;  // initial burn-in pool, in windows (replaces num_training_sets' role in set_current_index)
        
        // PER parameters
        double per_alpha;    // prioritization strength [0, 1]
        double per_lambda;   // temporal decay rate
        double per_epsilon;  // small constant for priority calculation
        
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
        void prioritized_experience_replay(vector<int32_t>& training_index);
        void set_current_index(int32_t _current_gen);
        void get_online_arguments(const vector<string> &arguments);
        
        // Core interface methods
        vector<int32_t> get_training_index(vector<int32_t>& training_index);
        vector< int32_t > get_validation_index(vector<int32_t>& validation_index);
        int32_t get_test_index();
        void get_test_indices(vector<int32_t>& test_indices);
        bool is_pooled_panel() const { return pooled_panel; }
        int32_t get_num_stocks() const { return num_stocks; }
        
        // PER priority system methods
        void update_episode_priorities(const vector<RNN_Genome*>& elite_genomes, int32_t current_generation);
        void write_priorities_to_csv(int32_t generation, const string& stats_directory);
        void log_priority_statistics(int32_t current_generation);
        
        // Getter for training data method
        string get_training_method() const { return get_training_data_method; }

        int32_t get_max_generation();
};

#endif