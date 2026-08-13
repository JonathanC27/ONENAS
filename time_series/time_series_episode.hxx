#ifndef TIME_SERIES_EPISODE_HXX
#define TIME_SERIES_EPISODE_HXX

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

class TimeSeriesEpisode {
   private:
    int32_t episode_id; // episode id is the index of the episode in the original time series - this NEVER changes
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;
    
    // PER-based priority system
    double validation_mse;  // MSE used for priority calculation
    int32_t availability_generation;  // generation when this episode first became available

    // Pooled panel identity: episode = (stock, window). In non-pooled mode
    // window_index == episode_id and stock_index == 0.
    int32_t window_index;
    int32_t stock_index;

    // Memory management
    bool is_loaded;

   public:
    // Constructors
    TimeSeriesEpisode(int32_t id);
    TimeSeriesEpisode(int32_t id, const vector<vector<double>>& inputs, const vector<vector<double>>& outputs);
    
    // Destructor
    ~TimeSeriesEpisode();
    
    // Data access methods
    const vector<vector<double>>& get_inputs();
    const vector<vector<double>>& get_outputs();
    void set_data(const vector<vector<double>>& inputs, const vector<vector<double>>& outputs);
    
    // Priority system methods
    void set_validation_mse(double mse);
    double get_validation_mse() const;
    void set_availability_generation(int32_t generation);
    int32_t get_availability_generation() const;
    void set_window_index(int32_t window);
    int32_t get_window_index() const;
    void set_stock_index(int32_t stock);
    int32_t get_stock_index() const;
    double calculate_priority(int32_t current_generation, double alpha = 0.6, double lambda = 0.01, double epsilon = 1e-8) const;
    
    // Memory management
    bool is_data_loaded() const;
    void ensure_loaded();  // Make sure data is in memory
    
    // Episode identification
    int32_t get_episode_id() const;
    
    // For debugging
    void print_stats() const;
};

#endif 