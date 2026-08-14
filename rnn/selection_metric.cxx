#include "selection_metric.hxx"

#include <algorithm>
using std::sort;

#include <cmath>
using std::fabs;
using std::pow;
using std::sqrt;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

SelectionMetric SelectionConfig::metric = SELECTION_MSE;
bool SelectionConfig::pooled_panel = false;
int32_t SelectionConfig::num_stocks = 1;
int32_t SelectionConfig::ic_ewma_halflife = 8;
double SelectionConfig::ic_gate_factor = 1.5;

void SelectionConfig::initialize_from_arguments(
    const vector<string>& arguments, bool _pooled_panel, int32_t _num_stocks
) {
    pooled_panel = _pooled_panel;
    num_stocks = _num_stocks;

    string metric_string = "mse";
    get_argument(arguments, "--selection_metric", false, metric_string);

    if (metric_string.compare("mse") == 0) {
        metric = SELECTION_MSE;
    } else if (metric_string.compare("ic") == 0) {
        metric = SELECTION_IC;
    } else if (metric_string.compare("ic_gated") == 0) {
        metric = SELECTION_IC_GATED;
    } else {
        Log::fatal(
            "--selection_metric must be one of 'mse', 'ic' or 'ic_gated', got '%s'\n", metric_string.c_str()
        );
        exit(1);
    }

    if (uses_ic() && !pooled_panel) {
        Log::fatal(
            "--selection_metric %s requires --pooled_panel. The cross-sectional rank IC is a "
            "correlation computed ACROSS the panel's stocks at each timestep, so it is only defined "
            "when the run supplies S date-aligned series (pooled panel mode). Either add "
            "--pooled_panel (with the matching --num_training_windows / --window_step arguments) or "
            "use --selection_metric mse.\n",
            metric_string.c_str()
        );
        exit(1);
    }

    if (uses_ic() && num_stocks < 3) {
        Log::fatal(
            "--selection_metric %s needs at least 3 panel series to form a rank cross-section, but "
            "only %d --training_filenames were given\n",
            metric_string.c_str(), num_stocks
        );
        exit(1);
    }

    get_argument(arguments, "--ic_ewma_halflife", false, ic_ewma_halflife);
    if (ic_ewma_halflife < 1) {
        Log::fatal("--ic_ewma_halflife must be >= 1 generation, got %d\n", ic_ewma_halflife);
        exit(1);
    }

    get_argument(arguments, "--ic_gate_factor", false, ic_gate_factor);
    if (ic_gate_factor < 1.0) {
        Log::fatal(
            "--ic_gate_factor must be >= 1.0 (it is a multiple of the best MSE in the island), got %f\n",
            ic_gate_factor
        );
        exit(1);
    }

    Log::info("Genome selection metric: %s\n", get_metric_name().c_str());
    if (ic_available()) {
        Log::info(
            "Cross-sectional rank IC will be computed over %d panel series (EWMA half-life %d "
            "generations, alpha %f)\n",
            num_stocks, ic_ewma_halflife, get_ic_ewma_alpha()
        );
    } else if (pooled_panel) {
        Log::info(
            "Pooled panel has only %d series, too narrow for a cross-sectional IC; MSE only\n", num_stocks
        );
    }
    if (gates_by_mse()) {
        Log::info(
            "IC gating enabled: a genome is only eligible if its validation MSE is within %fx the "
            "best MSE in its island\n",
            ic_gate_factor
        );
    }
}

double SelectionConfig::get_ic_ewma_alpha() {
    // half-life h in generations -> weight of the observation alpha such that (1-alpha)^h = 0.5
    return 1.0 - pow(0.5, 1.0 / (double) ic_ewma_halflife);
}

string SelectionConfig::get_metric_name() {
    switch (metric) {
        case SELECTION_IC: return "ic (EWMA of mean daily cross-sectional Spearman rank IC)";
        case SELECTION_IC_GATED: return "ic_gated (IC, restricted to genomes near the island's best MSE)";
        default: return "mse (validation mean squared error)";
    }
}

/** Average ranks (1-based), ties share the mean of the tied positions. */
static void rank_with_ties(const vector<double>& values, vector<double>& ranks) {
    int32_t n = (int32_t) values.size();
    vector<int32_t> order(n);
    for (int32_t i = 0; i < n; i++) order[i] = i;
    sort(order.begin(), order.end(), [&values](int32_t a, int32_t b) { return values[a] < values[b]; });

    ranks.assign(n, 0.0);
    int32_t i = 0;
    while (i < n) {
        int32_t j = i;
        while (j + 1 < n && values[order[j + 1]] == values[order[i]]) j++;
        // positions i..j (0-based) are tied -> average 1-based rank
        double average_rank = ((double) (i + j) / 2.0) + 1.0;
        for (int32_t k = i; k <= j; k++) ranks[order[k]] = average_rank;
        i = j + 1;
    }
}

double spearman_rank_correlation(const vector<double>& a, const vector<double>& b) {
    int32_t n = (int32_t) a.size();
    if (n < 3 || (int32_t) b.size() != n) return NAN;

    vector<double> rank_a;
    vector<double> rank_b;
    rank_with_ties(a, rank_a);
    rank_with_ties(b, rank_b);

    double mean = ((double) n + 1.0) / 2.0;  // mean of 1..n, and of any average-rank vector
    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double da = rank_a[i] - mean;
        double db = rank_b[i] - mean;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    // all values tied on one side -> rank vector is constant -> correlation undefined
    if (var_a <= 0.0 || var_b <= 0.0) return NAN;

    double correlation = cov / sqrt(var_a * var_b);
    // guard against fp drift pushing us microscopically outside [-1, 1]
    if (correlation > 1.0) correlation = 1.0;
    if (correlation < -1.0) correlation = -1.0;
    return correlation;
}

double cross_sectional_rank_ic(
    const vector<vector<vector<double> > >& predictions, const vector<vector<vector<double> > >& expected,
    int32_t num_stocks, int32_t& num_cross_sections
) {
    num_cross_sections = 0;

    int32_t num_series = (int32_t) predictions.size();
    if (num_stocks < 3 || num_series < num_stocks || (int32_t) expected.size() != num_series) return NAN;
    if (num_series % num_stocks != 0) {
        Log::warning(
            "Cross-sectional IC: %d validation series is not a multiple of the %d panel series; "
            "the trailing partial window is ignored\n",
            num_series, num_stocks
        );
    }

    int32_t num_windows = num_series / num_stocks;
    double ic_sum = 0.0;

    vector<double> predicted_cross_section((size_t) num_stocks, 0.0);
    vector<double> expected_cross_section((size_t) num_stocks, 0.0);

    for (int32_t w = 0; w < num_windows; w++) {
        // every stock in a window carries the same number of timesteps (contemporaneous windows),
        // but take the min defensively
        int32_t time_length = -1;
        bool usable = true;
        for (int32_t s = 0; s < num_stocks; s++) {
            int32_t v = w * num_stocks + s;
            if (predictions[v].empty() || expected[v].empty()) {
                usable = false;
                break;
            }
            int32_t len = (int32_t) predictions[v][0].size();
            if ((int32_t) expected[v][0].size() < len) len = (int32_t) expected[v][0].size();
            if (time_length < 0 || len < time_length) time_length = len;
        }
        if (!usable || time_length <= 0) continue;

        for (int32_t t = 0; t < time_length; t++) {
            for (int32_t s = 0; s < num_stocks; s++) {
                int32_t v = w * num_stocks + s;
                // output 0 only: a cross-sectional IC is defined per predicted quantity, and the
                // pooled panel configuration predicts a single output (the forward return)
                predicted_cross_section[s] = predictions[v][0][t];
                expected_cross_section[s] = expected[v][0][t];
            }

            double ic = spearman_rank_correlation(predicted_cross_section, expected_cross_section);
            if (!std::isnan(ic)) {
                ic_sum += ic;
                num_cross_sections++;
            }
        }
    }

    if (num_cross_sections == 0) return NAN;
    return ic_sum / (double) num_cross_sections;
}
