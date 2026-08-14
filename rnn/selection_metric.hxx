#ifndef SELECTION_METRIC_HXX
#define SELECTION_METRIC_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

/**
 * Which quantity genome selection (elite ranking, global-best selection) optimizes.
 *
 *   SELECTION_MSE       rank by validation MSE. Historical behavior, and the default.
 *   SELECTION_IC        rank by the EWMA of the mean daily cross-sectional Spearman rank IC
 *                       computed over the validation windows. Pooled panel mode only.
 *   SELECTION_IC_GATED  same as SELECTION_IC, but a genome is only eligible if its validation
 *                       MSE is within --ic_gate_factor of the best MSE in its island; ineligible
 *                       genomes sort after every eligible one.
 *
 * Motivation: on a pooled daily stock panel the validation MSE is ~99% the market's realized
 * variance, so ranking by MSE is very nearly ranking by noise. The cross-sectional rank IC only
 * measures whether a genome orders the stocks correctly on a given day, which is the quantity a
 * cross-sectional strategy actually trades on.
 */
enum SelectionMetric { SELECTION_MSE = 0, SELECTION_IC = 1, SELECTION_IC_GATED = 2 };

/**
 * Process-wide selection settings, parsed once from the command line.
 *
 * This is a global (like the weight update method) because RNN_Genome::get_fitness() is called
 * from deep inside the population/island machinery which has no path to per-run configuration.
 * Every MPI rank parses the same argv, so master and workers agree. The defaults reproduce the
 * historical MSE-only behavior exactly, so binaries that never call initialize_from_arguments()
 * (EXAMM and friends) are unaffected.
 */
class SelectionConfig {
   private:
    static SelectionMetric metric;
    static bool pooled_panel;
    static int32_t num_stocks;
    static int32_t ic_ewma_halflife;
    static double ic_gate_factor;

   public:
    /**
     * Parses --selection_metric / --ic_ewma_halflife / --ic_gate_factor.
     *
     * \param arguments the full argv
     * \param _pooled_panel whether --pooled_panel is in effect
     * \param _num_stocks the panel width S (1 when not pooled)
     *
     * Fatal error if an IC metric is requested without --pooled_panel: the cross-section that the
     * IC is computed over only exists on a panel.
     */
    static void initialize_from_arguments(const vector<string>& arguments, bool _pooled_panel, int32_t _num_stocks);

    static SelectionMetric get_metric() { return metric; }

    /** true when the selection metric itself is IC-based. */
    static bool uses_ic() { return metric == SELECTION_IC || metric == SELECTION_IC_GATED; }

    /** true when ineligible (high-MSE) genomes must sort last. */
    static bool gates_by_mse() { return metric == SELECTION_IC_GATED; }

    /**
     * true when a cross-sectional IC can be formed at all. IC is computed (and logged) whenever
     * the panel is wide enough, even under MSE selection, so runs always carry the diagnostic.
     */
    static bool ic_available() { return pooled_panel && num_stocks >= 3; }

    static bool is_pooled_panel() { return pooled_panel; }
    static int32_t get_num_stocks() { return num_stocks; }
    static int32_t get_ic_ewma_halflife() { return ic_ewma_halflife; }
    static double get_ic_gate_factor() { return ic_gate_factor; }

    /** EWMA smoothing factor derived from the half-life in generations. */
    static double get_ic_ewma_alpha();

    static string get_metric_name();
};

/**
 * Mean daily cross-sectional Spearman rank IC over a pooled-panel validation set.
 *
 * \param predictions  [series][output][timestep], as returned by RNN_Genome::get_predictions
 * \param expected     [series][output][timestep], the validation targets
 * \param num_stocks   panel width S
 * \param num_cross_sections out-param: how many (window, timestep) cross-sections contributed
 *
 * The validation series are laid out window-major / stock-minor (OnlineSeries::get_validation_index
 * pushes all S stocks for window 0, then all S stocks for window 1, ...), so series index
 * v = window * S + stock and every group of S consecutive series is one contemporaneous window.
 * For each (window, timestep) the S predicted and S expected values form a cross-section; the
 * Spearman rank correlation of that cross-section is averaged over all of them.
 *
 * Cross-sections where either side has no rank variation (all ties) are skipped -- the correlation
 * is undefined there. Returns NAN if nothing contributed.
 */
double cross_sectional_rank_ic(
    const vector<vector<vector<double> > >& predictions, const vector<vector<vector<double> > >& expected,
    int32_t num_stocks, int32_t& num_cross_sections
);

/** Spearman rank correlation between two equal-length samples. NAN when either side is constant. */
double spearman_rank_correlation(const vector<double>& a, const vector<double>& b);

#endif
