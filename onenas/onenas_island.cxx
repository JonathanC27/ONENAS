#include <algorithm>
using std::sort;
using std::lower_bound;
using std::upper_bound;

#include <iomanip>
using std::setw;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include <unordered_map>
using std::unordered_map;

#include "onenas_island.hxx"
// #include "rnn_ge/nome.hxx"

#include "common/log.hxx"

OneNasIsland::OneNasIsland(int32_t _id, int32_t _generated_size, int32_t _elite_size) {
    id = _id;
    generate_population_size = _generated_size;
    elite_population_size = _elite_size;
    status = INITIALIZING;
    erase_again = 0; 
    erased = false;

    elite_population = new Population(ELITE, elite_population_size, id);
    generated_population = new Population(GENERATED, generate_population_size, id);
}

OneNasIsland::OneNasIsland(int32_t _id, Population* _elite_population, int32_t _generated_size, int32_t _elite_size) : id(_id), generate_population_size(_generated_size), elite_population_size(_elite_size), elite_population(_elite_population), status(OneNasIsland::FILLED), erase_again(0), erased(false) {
    elite_population = _elite_population;
    generated_population = new Population(GENERATED, generate_population_size, id);
}

OneNasIsland::~OneNasIsland() {
    // Clean up population memory to prevent memory leaks
    if (elite_population != NULL) {
        Log::debug("Destructor: Deleting elite_population at %p for island %d\n", elite_population, id);
        delete elite_population;
        elite_population = NULL;
    }
    
    if (generated_population != NULL) {
        Log::debug("Destructor: Deleting generated_population at %p for island %d\n", generated_population, id);
        delete generated_population;
        generated_population = NULL;
    }
    
    Log::debug("OneNasIsland destructor completed for island %d\n", id);
}

RNN_Genome* OneNasIsland::get_best_genome() {
    return elite_population->get_best_genome();
}

RNN_Genome* OneNasIsland::get_worst_genome() {
    return elite_population->get_worst_genome();
}

double OneNasIsland::get_best_fitness() {
    // RNN_Genome *best_genome = elite_population->get_best_genome();
    // if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    // else return best_genome->get_fitness();
    return elite_population->get_best_fitness();
}

double OneNasIsland::get_worst_fitness() {
    // RNN_Genome *worst_genome = elite_population->get_worst_genome();
    // if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    // else return worst_genome->get_fitness();
    return elite_population->get_worst_fitness();
}

// int32_t OneNasIsland::get_max_size() {
//     return max_size;
// }

int32_t OneNasIsland::generated_size() {
    return (int32_t)generated_population->size();
}

int32_t OneNasIsland::elite_size() {
    return (int32_t)elite_population->size();
}
// after first generation, the elite population should always be full
bool OneNasIsland::elite_is_full() {
    bool filled = elite_population->is_full();
    if (filled) {
        status = OneNasIsland::FILLED;
    }
    return filled;
}
// // 
// bool OneNasIsland::generated_is_full() {
//     bool filled = generated_population->is_full();
//     if (filled) {
//         status = OneNasIsland::FILLED;
//     }
//     return filled;
// }

bool OneNasIsland::is_initializing() {
    return status == OneNasIsland::INITIALIZING;
}

// bool OneNasIsland::is_filled() {
//     return status == OneNasIsland::FILLED;
// }

bool OneNasIsland::is_repopulating() {
    return status == OneNasIsland::REPOPULATING;
}



void OneNasIsland::copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome) {
    // int32_t genome_position = size() * rng_0_1(generator);
    // *genome = genomes[genome_position]->copy();
    if (elite_size() > 0) {
        Log::info("Island %d: copying random genome from elite population\n", id);
        elite_population->copy_random_genome(rng_0_1, generator, genome);
    } else if (generated_size() > 0) {
        Log::info("Island %d: copying random genome from generated population\n", id);
        generated_population->copy_random_genome(rng_0_1, generator, genome);
    } else {
        Log::fatal("ERROR: Cannot copy random genome from island %d - both elite and generated populations are empty!\n", id);
        *genome = NULL;
    }
}

void OneNasIsland::copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2) {
    if (elite_size() >= 2) {
        Log::info("copying two random genomes from elite population\n");
        elite_population->copy_two_random_genomes(rng_0_1, generator, genome1, genome2);
    } else if (generated_size() >= 2) {
        Log::info("copying two random genomes from generated population\n");
        generated_population->copy_two_random_genomes(rng_0_1, generator, genome1, genome2);
    } else {
        Log::fatal("CANNOT copy two random genomes, elite population size is %d, generated population size is %d\n", elite_size(), generated_size());
    }

}

// void OneNasIsland::do_population_check(int line, int initial_size) {
//     if (status == OneNasIsland::FILLED && genomes.size() < max_size) {
//         Log::error("ERROR: do_population_check had issue on island.cxx line %d, status was FILLED and genomes.size() was: %d, size at beginning of insert was: %d\n", line, genomes.size(), initial_size);
//         status = OneNasIsland::INITIALIZING;
//     }
// }


//returns -1 for not inserted, otherwise the index it was inserted at
//inserts a copy of the genome, caller of the function will need to delete their
//pointer
int32_t OneNasIsland::insert_genome(RNN_Genome *genome) {
    int32_t insert_position;
    if (genome->get_genome_type() == GENERATED) {
        insert_position = generated_population->insert_genome(genome);
    } else if (genome->get_genome_type() == ELITE) {

        insert_position = elite_population->insert_genome(genome);
    } else {
        Log::fatal("genome type is %d, not inserting \n", genome->get_genome_type());
        exit(1);
    }
    if (elite_size() >= elite_population_size) {
        //the island is filled
        status = OneNasIsland::FILLED;
    }
    return insert_position;
}

void OneNasIsland::get_elite_population_ids(vector<int32_t>& good_genome_ids) {
    vector<RNN_Genome*> elite_genomes = elite_population->get_genomes();
    for (int32_t i = 0; i < (int32_t)elite_genomes.size(); i++) {
        good_genome_ids.push_back(elite_genomes[i]->get_generation_id());
    }
}

// void OneNasIsland::print(string indent) {
//     if (Log::at_level(Log::INFO)) {

//         Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());

//         for (int32_t i = 0; i < genomes.size(); i++) {
//             Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
//         }
//     }
// }

void OneNasIsland::erase_island() {
    erased_generation_id = latest_generation_id;
    elite_population->erase_population();
    generated_population->erase_population();
    erased = true;
    erase_again = 5;
    Log::info("Island %d erased!\n", id);
}

// void OneNasIsland::erase_structure_map() {
//     Log::info("Erasing the structure map in the worst performing island\n");
//     structure_map.clear();
//     Log::debug("after erase structure map size is %d\n", structure_map.size());
// }

int32_t OneNasIsland::get_erased_generation_id() {
    return erased_generation_id;
}

int32_t OneNasIsland::get_status() {
    return status;
}

void OneNasIsland::set_status(int32_t status_to_set) {
    if (status_to_set == OneNasIsland::INITIALIZING || status_to_set == OneNasIsland::FILLED || status_to_set == OneNasIsland::REPOPULATING) {
        status = status_to_set;
    } else {
        Log::error("OneNasIsland::set_status: Wrong island status to set! %d\n", status_to_set);
        exit(1);
    }
}

bool OneNasIsland::been_erased() {
    return erased;
}

vector<RNN_Genome *> OneNasIsland::get_genomes() {
    return elite_population->get_genomes();
}

void OneNasIsland::set_latest_generation_id(int32_t _latest_generation_id){
    latest_generation_id = _latest_generation_id;
}

int32_t OneNasIsland::get_erase_again_num() {
    return erase_again;
}

void OneNasIsland::set_erase_again_num() {
    erase_again--;
}

void OneNasIsland::evaluate_elite_population(const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output) {
        // if (elite_genomes.size() == 0) return;
    Log::info("Finalizing generation: Evaluating elite population on island %d\n", id);
    vector<RNN_Genome *> elite_genomes = elite_population->get_genomes();
    int32_t elite_population_size = elite_population->get_population_size();
    for (int32_t i = 0; i < elite_population_size; i++) {
        RNN_Genome* g = elite_genomes[i];
        g->evaluate_online(validation_input, validation_output);
    }

    // The gate and the exploding-prediction guard are relative to the rest of the island, so they
    // can only be resolved once every elite has been re-evaluated on this generation's data.
    apply_selection_flags();

    elite_population->sort_population("fitness");
    for (int32_t i = 0; i < elite_population_size; i++) {
        RNN_Genome* g = elite_genomes[i];
        if (SelectionConfig::ic_available()) {
            Log::info(
                "Island %d: elite genome %d fitness: %f (MSE %.8f, IC %.6f, IC EWMA %.6f%s)\n", id, i,
                g->get_fitness(), g->get_best_validation_mse(), g->get_validation_ic(), g->get_ic_ewma(),
                g->is_selection_gated() ? ", GATED" : ""
            );
        } else {
            Log::info("Island %d: elite genome %d fitness: %f\n", id, i, g->get_fitness());
        }
    }
}

/**
 * Recomputes the island-relative selection flags over the current elite population.
 *
 * Only --selection_metric ic_gated sets any flag: a genome is eligible when its validation MSE is
 * within --ic_gate_factor of the island's best MSE, and ineligible genomes are pushed behind every
 * eligible one by get_fitness(). Because the factor is >= 1 the island's own best-MSE genome is
 * always eligible, so gating can never empty the island.
 */
void OneNasIsland::apply_selection_flags() {
    vector<RNN_Genome*> genomes = elite_population->get_genomes();
    if (genomes.empty()) return;

    if (!SelectionConfig::gates_by_mse()) {
        // Not gating: make sure no stale flag survives a metric change or a promoted copy.
        for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
            if (genomes[i] != NULL) genomes[i]->set_selection_gated(false);
        }
        return;
    }

    double threshold = get_mse_gate_threshold();
    int32_t gated = 0;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        if (genomes[i] == NULL) continue;
        bool gate = genomes[i]->get_best_validation_mse() > threshold;
        genomes[i]->set_selection_gated(gate);
        if (gate) gated++;
    }

    if (gated > 0) {
        Log::info(
            "Island %d: IC gate (%.3fx best MSE = %.8f) made %d of %d elite genomes ineligible\n", id,
            SelectionConfig::get_ic_gate_factor(), threshold, gated, (int32_t)genomes.size()
        );
    }
}

/**
 * MSE ceiling for eligibility under --selection_metric ic_gated: ic_gate_factor times the best
 * validation MSE currently in the island. Returns EXAMM_MAX_DOUBLE when gating is off or when no
 * genome has a usable MSE, which makes every genome eligible.
 */
double OneNasIsland::get_mse_gate_threshold() {
    if (!SelectionConfig::gates_by_mse()) return EXAMM_MAX_DOUBLE;

    vector<RNN_Genome*> genomes = elite_population->get_genomes();
    double best_mse = EXAMM_MAX_DOUBLE;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        if (genomes[i] == NULL) continue;
        double mse = genomes[i]->get_best_validation_mse();
        if (!std::isnan(mse) && mse < best_mse) best_mse = mse;
    }

    if (best_mse >= EXAMM_MAX_DOUBLE) return EXAMM_MAX_DOUBLE;
    return best_mse * SelectionConfig::get_ic_gate_factor();
}

void OneNasIsland::select_elite_population() {
    // vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    Log::info("Finalizing generation: Selecting elite population on island %d\n", id);
    vector<RNN_Genome*> trained_genomes = generated_population->get_genomes();

    // Threshold from the elites that were just re-evaluated. Promoted genomes are flagged BEFORE
    // insertion so that insert_genome()'s keep/discard comparison already sees the gated ordering.
    double gate_threshold = get_mse_gate_threshold();

    for (int i = 0; i < (int32_t)trained_genomes.size(); i++) {
        RNN_Genome* genome_copy = trained_genomes[i]->copy();
        genome_copy->set_genome_type(ELITE);
        genome_copy->set_selection_gated(
            SelectionConfig::gates_by_mse() && genome_copy->get_best_validation_mse() > gate_threshold
        );
        elite_population->insert_genome(genome_copy);
    }

    generated_population->erase_population();

    // The promotions can move the island's best MSE, so re-derive the flags over the merged
    // population and re-sort. Both are no-ops unless gating is on.
    if (SelectionConfig::gates_by_mse()) {
        apply_selection_flags();
        elite_population->sort_population("fitness");
    }
}

void OneNasIsland::write_prediction(string filename, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    elite_population->write_prediction(filename, test_input, test_output);
}

void OneNasIsland::save_entire_population(string output_path) {
    elite_population->save_entire_population(output_path);
    generated_population->save_entire_population(output_path);
}

void OneNasIsland::generation_check() {
    //  at the end of each generation, check if the elite population is full
    // and check if the generated population is empty
    if (elite_is_full()) {
        Log::info("Generation check: Island %d elite population is full\n", id);
    } else {
        Log::error("Generation check: Island %d elite population is not full, its size is %d\n", id, elite_size());
    }
    if (generated_population->is_empty()) {
        Log::info("Generation check: Island %d generated population is empty\n", id);
    } else {
        Log::error("Generation check: Island %d generated population is not empty, its size is %d\n", id, generated_size());
    }
}