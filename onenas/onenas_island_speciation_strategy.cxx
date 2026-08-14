#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <random>

using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <fstream>
using std::ofstream;

#include "examm.hxx"
#include "rnn/rnn_genome.hxx"
#include "onenas_island_speciation_strategy.hxx"
#include "onenas.hxx"

#include "common/files.hxx"
#include "common/log.hxx"

/**
 *
 */
OneNasIslandSpeciationStrategy::OneNasIslandSpeciationStrategy(
        int32_t _number_of_islands, int32_t _generated_population_size, int32_t _elite_population_size, 
        double _mutation_rate, double _intra_island_crossover_rate,
        double _inter_island_crossover_rate, RNN_Genome *_seed_genome,
        string _island_ranking_method, string _repopulation_method,
        int32_t _repopulation_frequency, int32_t _num_mutations, int32_t _repopulation_mutations,
        int32_t _islands_to_exterminate, bool _repeat_extinction, string _output_directory,
        string _control_size_method, bool _compare_with_naive, bool _write_elite_predictions
        ) :
                        generation_island(0),
                        number_of_islands(_number_of_islands),
                        generated_population_size(_generated_population_size),
                        elite_population_size(_elite_population_size),
                        mutation_rate(_mutation_rate),
                        intra_island_crossover_rate(_intra_island_crossover_rate),
                        inter_island_crossover_rate(_inter_island_crossover_rate),
                        generated_genomes(0),
                        evaluated_genomes(0),
                        seed_genome(_seed_genome),
                        island_ranking_method(_island_ranking_method),
                        repopulation_method(_repopulation_method),
                        repopulation_frequency(_repopulation_frequency),
                        num_mutations(_num_mutations),
                        repopulation_mutations(_repopulation_mutations),
                        islands_to_exterminate(_islands_to_exterminate),
                        repeat_extinction(_repeat_extinction),
                        output_directory(_output_directory),
                        naive_better_count(0),
                        genome_better_count(0),
                        control_size_method(_control_size_method),
                        compare_with_naive(_compare_with_naive),
                        write_elite_predictions(_write_elite_predictions),
                        onenas_instance(nullptr) {
    double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
    if (rate_sum != 1.0) {
        mutation_rate = mutation_rate / rate_sum;
        intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
        inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
    }

    intra_island_crossover_rate += mutation_rate;
    inter_island_crossover_rate += intra_island_crossover_rate;
    Log::info("OneNAS Strategy: Generated population size is %d, elite population size is %d\n", generated_population_size, elite_population_size);
    Log::info("OneNAS Strategy: Mutation rate %f, inter-island crossover rate %f, intra island crossover rate %f\n", mutation_rate, inter_island_crossover_rate, intra_island_crossover_rate);
    Log::info("OneNAS Strategy: Repopulation frequency is %d, islands to exterminate is %d\n", repopulation_frequency, islands_to_exterminate);
    Log::info("OneNAS Strategy: Doing repopulation is set to %s, and it will start at generation %d\n", repopulation_frequency > 0 ? "TRUE" : "FALSE", repopulation_frequency * 2);
    if (write_elite_predictions) {
        Log::info("OneNAS Strategy: Writing per-elite test predictions to generation_<g>_elites.csv every generation\n");
    }
    //set the generation id for the initial minimal genome
    seed_genome->set_generation_id(generated_genomes);
    generated_genomes++;
    global_best_genome = NULL;
}

OneNasIslandSpeciationStrategy::~OneNasIslandSpeciationStrategy() {
    // Clean up global_best_genome memory to prevent memory leaks
    if (global_best_genome != NULL) {
        Log::debug("Destructor: Deleting global_best_genome at %p\n", global_best_genome);
        delete global_best_genome;
        global_best_genome = NULL;
    }
    
    // Clean up islands
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i] != NULL) {
            delete islands[i];
            islands[i] = NULL;
        }
    }
    islands.clear();
    
    Log::debug("OneNasIslandSpeciationStrategy destructor completed\n");
}

int32_t OneNasIslandSpeciationStrategy::get_generated_genomes() const {
    return generated_genomes;
}

int32_t OneNasIslandSpeciationStrategy::get_evaluated_genomes() const {
    return evaluated_genomes;
}

int32_t OneNasIslandSpeciationStrategy::get_generated_population_size() const {
    return generated_population_size;
}

RNN_Genome* OneNasIslandSpeciationStrategy::get_best_genome() {
    //the global_best_genome is updated every time a genome is inserted
    return global_best_genome;
}

RNN_Genome* OneNasIslandSpeciationStrategy::get_worst_genome() {
    int32_t worst_genome_island = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->elite_size() > 0) {
            double island_worst_fitness = islands[i]->get_worst_fitness();
            if (island_worst_fitness > worst_fitness) {
                worst_fitness = island_worst_fitness;
                worst_genome_island = i;
            }
        }
    }

    if (worst_genome_island < 0) {
        return NULL;
    } else {
        return islands[worst_genome_island]->get_worst_genome();
    }
}


double OneNasIslandSpeciationStrategy::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double OneNasIslandSpeciationStrategy::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

bool OneNasIslandSpeciationStrategy::islands_full() const {
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (!islands[i]->elite_is_full()) return false;
    }

    return true;
}


//this will insert a COPY, original needs to be deleted
//returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
int32_t OneNasIslandSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    // NOTE: commented out this part because global best is selected at end of generation
    //       also it is not necessary to track global best genome here
    // bool new_global_best = false;
    // if (global_best_genome == NULL) {
    //     //this is the first insert of a genome so it's the global best by default
    //     global_best_genome = genome->copy();
    //     new_global_best = true;
    // } else if (global_best_genome->get_fitness() > genome->get_fitness()) {
    //     //since we're re-setting this to a copy you need to delete it.
    //     Log::info("[DEBUG] About to delete global_best_genome at %p in insert_genome\n", global_best_genome);
    //     delete global_best_genome;
    //     global_best_genome = genome->copy();
    //     new_global_best = true;
    // }

    evaluated_genomes++;
    int32_t island = genome->get_group_id();

    int32_t insert_position = islands[island]->insert_genome(genome);


    return insert_position;
    // if (insert_position == 0) {
    //     if (new_global_best) return 0;
    //     else return 1;
    // } else {
    //     return insert_position; //will be -1 if not inserted, or > 0 if not the global best
    // }
}

int32_t OneNasIslandSpeciationStrategy::get_worst_island_by_best_genome() {
    int32_t worst_island = -1;
    double worst_best_fitness = 0;
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->elite_size() > 0) {
            if (islands[i]->get_erase_again_num() > 0) continue;
            double island_best_fitness = islands[i]->get_best_fitness();
            if (island_best_fitness > worst_best_fitness) {
                worst_best_fitness = island_best_fitness;
                worst_island = i;
            }
        }
    }
    return worst_island;
}

vector<int32_t> OneNasIslandSpeciationStrategy::rank_islands() {
    vector<int32_t> island_rank;
    int32_t temp;
    double fitness_j1, fitness_j2;
    Log::info("ranking islands\n");
    Log::info("repeat extinction: %s\n", repeat_extinction? "true":"false");
    for (int32_t i = 0; i< number_of_islands; i++){
        if (repeat_extinction) {
            island_rank.push_back(i);
        } else {
            if (islands[i] -> get_erase_again_num() == 0) {
                island_rank.push_back(i);
            }
        }
    }

    for (int32_t i = 0; i < (int32_t)island_rank.size() - 1; i++)   {
        for (int32_t j = 0; j < (int32_t)island_rank.size() - i - 1; j++)  {
            fitness_j1 = islands[island_rank[j]]->get_best_fitness();
            fitness_j2 = islands[island_rank[j+1]]->get_best_fitness();
            if (fitness_j1 < fitness_j2) {
                temp = island_rank[j];
                island_rank[j] = island_rank[j+1];
                island_rank[j+1]= temp;
            }
        }
    }
    Log::debug("island rank:\n");
    for (int32_t i = 0; i < (int32_t)island_rank.size(); i++){
        Log::debug("island: %d fitness %f\n", island_rank[i], islands[island_rank[i]]->get_best_fitness());
    }
    return island_rank;
}


RNN_Genome* OneNasIslandSpeciationStrategy::generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover, WeightRules* weight_rules) {
    //generate the genome from the next island in a round
    //robin fashion.


    Log::info("Generating genome %d for island: %d\n", generated_genomes, generation_island);
    OneNasIsland *current_island = islands[generation_island];
    RNN_Genome *new_genome = NULL;

    // Log::info("generating new genome for island[%d], island_size: %d, max_island_size: %d, mutation_rate: %lf, intra_island_crossover_rate: %lf, inter_island_crossover_rate: %lf\n", generation_island, island->size(), generated_genome_size, mutation_rate, intra_island_crossover_rate, inter_island_crossover_rate);
    if (current_island->is_initializing()) {
        Log::info("Island %d: island is initializing\n", generation_island);
        new_genome = generate_for_initializing_island(rng_0_1, generator, mutate, weight_rules);

    } else if (current_island->elite_is_full()) {
        Log::info("Island %d: island elite is full\n", generation_island);
        new_genome = generate_for_filled_island(rng_0_1, generator, mutate, crossover);

    } else if (current_island->is_repopulating()) {
        //select two other islands (non-overlapping) at random, and select genomes
        //from within those islands and generate a child via crossover
        Log::info("Island %d: island is repopulating\n", generation_island);
        new_genome = generate_for_repopulating_island(rng_0_1, generator, mutate, crossover, weight_rules);

    } else {
        Log::fatal("ERROR: island was neither initializing, repopulating or full.\n");
        Log::fatal("This should never happen!\n");

    }

    if (new_genome == NULL) {
        Log::info("Island %d: new genome is still null, regenerating\n", generation_island);
        new_genome = generate_genome(rng_0_1, generator, mutate, crossover, weight_rules);
    }
    generated_genomes++;
    new_genome->set_generation_id(generated_genomes);
    islands[generation_island]->set_latest_generation_id(generated_genomes);
    new_genome->set_group_id(generation_island);
    new_genome->set_genome_type(GENERATED);

    if (current_island->is_initializing()) {
        RNN_Genome* genome_copy = new_genome->copy();
        Log::debug("inserting genome copy!\n");
        insert_genome(genome_copy);
    }
    generation_island++;
    if (generation_island >= (int32_t) islands.size()) {
        generation_island = 0;
    }

    return new_genome;
}

RNN_Genome* OneNasIslandSpeciationStrategy::generate_for_initializing_island(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, function<void(int32_t, RNN_Genome*)>& mutate, WeightRules* weight_rules
) {
    OneNasIsland* current_island = islands[generation_island];
    RNN_Genome* new_genome = NULL;
    if (current_island->generated_size() == 0) {
        Log::info("Island %d: starting island with minimal genome\n", generation_island);
        new_genome = seed_genome->copy();
        new_genome->initialize_randomly(weight_rules);

    } else {
        Log::info("Island %d: island is initializing but not empty, mutating a random genome\n", generation_island);
        while (new_genome == NULL) {
            current_island->copy_random_genome(rng_0_1, generator, &new_genome);
            mutate(num_mutations, new_genome);
            if (new_genome->outputs_unreachable()) {
                // no path from at least one input to the outputs
                delete new_genome;
                new_genome = NULL;
            }
        }
    }
    new_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
    new_genome->best_validation_mae = EXAMM_MAX_DOUBLE;

    return new_genome;
}

RNN_Genome* OneNasIslandSpeciationStrategy::generate_for_filled_island(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, function<void(int32_t, RNN_Genome*)>& mutate,
    function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
) {
    // if we haven't filled ALL of the island populations yet, only use mutation
    // otherwise do mutation at %, crossover at %, and island crossover at %
    OneNasIsland* island = islands[generation_island];
    RNN_Genome* genome;
    double r = rng_0_1(generator);

    if (!island->elite_is_full() || r < mutation_rate) {
        Log::info("Island %d generate_for_filled_island: populating through mutation\n", generation_island);
        island->copy_random_genome(rng_0_1, generator, &genome);
        mutate(num_mutations, genome);

    } else if (r < intra_island_crossover_rate || number_filled_islands() == 1) {
        // intra-island crossover
        Log::info("Island %d generate_for_filled_island: performing intra-island crossover\n", generation_island);
        // select two distinct parent genomes in the same island
        RNN_Genome *parent1 = NULL, *parent2 = NULL;
        island->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);
        genome = crossover(parent1, parent2);
        delete parent1;
        delete parent2;
    } else {
        // get a random genome from this island
        Log::info("Island %d: island is full and is populating through inter-island crossover\n", generation_island);
        RNN_Genome* parent1 = NULL;
        island->copy_random_genome(rng_0_1, generator, &parent1);

        // select a different island randomly
        int32_t other_island_index = get_other_full_island(rng_0_1, generator, generation_island);
        Log::info("Island %d: select island: %d as parent 2 for inter-island crossover\n", generation_island, other_island_index);
        // get the best genome from the other island
        RNN_Genome* parent2 = islands[other_island_index]->get_best_genome()->copy();  // new RNN GENOME

        // swap so the first parent is the more fit parent
        if (parent1->get_fitness() > parent2->get_fitness()) {
            RNN_Genome* tmp = parent1;
            parent1 = parent2;
            parent2 = tmp;
        }
        genome = crossover(parent1, parent2);  // new RNN GENOME
        delete parent1;
        delete parent2;
    }

    if (genome->outputs_unreachable()) {
        // no path from at least one input to the outputs
        delete genome;
        genome = NULL;
    }
    return genome;
}

RNN_Genome* OneNasIslandSpeciationStrategy::generate_for_repopulating_island(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, function<void(int32_t, RNN_Genome*)>& mutate,
    function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover, WeightRules* weight_rules
) {
    Log::info("Island %d: island is repopulating \n", generation_island);
    // Island *current_island = islands[generation_island];
    RNN_Genome* new_genome = NULL;

    if (repopulation_method.compare("randomParents") == 0 || repopulation_method.compare("randomparents") == 0) {
        Log::info("Island %d: island is repopulating through random parents method!\n", generation_island);
        new_genome = parents_repopulation("randomParents", rng_0_1, generator, mutate, crossover, weight_rules);

    } else if (repopulation_method.compare("bestParents") == 0 || repopulation_method.compare("bestparents") == 0) {
        Log::info("Island %d: island is repopulating through best parents method!\n", generation_island);
        new_genome = parents_repopulation("bestParents", rng_0_1, generator, mutate, crossover, weight_rules);

    } else if (repopulation_method.compare("bestGenome") == 0 || repopulation_method.compare("bestgenome") == 0) {
        Log::info("Island %d: island is repopulating through best genome method!\n", generation_island);
        new_genome = get_global_best_genome()->copy();
        mutate(repopulation_mutations, new_genome);

    } else if (repopulation_method.compare("bestIsland") == 0 || repopulation_method.compare("bestisland") == 0) {
        // copy the best island to the worst at once
        Log::info(
            "Island %d: island is repopulating through bestIsland method! Coping the best island to the population "
            "island\n",
            generation_island
        );
        Log::info(
            "Island %d: island current size is: %d \n", generation_island,
            islands[generation_island]->get_genomes().size()
        );
        int32_t best_island_id = get_best_genome()->get_group_id();
        repopulate_by_copy_island(best_island_id, mutate);
        if (new_genome == NULL) {
            new_genome = generate_for_filled_island(rng_0_1, generator, mutate, crossover);
        }
    } else {
        Log::fatal("Wrong repopulation method: %s\n", repopulation_method.c_str());
        exit(1);
    }
    return new_genome;
}

void OneNasIslandSpeciationStrategy::repopulate_by_copy_island(
    int32_t best_island_id, function<void(int32_t, RNN_Genome*)>& mutate
) {
    vector<RNN_Genome*> best_island_genomes = islands[best_island_id]->get_genomes();
    for (int32_t i = 0; i < (int32_t) best_island_genomes.size(); i++) {
        // copy the genome from the best island
        RNN_Genome* copy = best_island_genomes[i]->copy();
        mutate(num_mutations, copy);

        generated_genomes++;
        copy->set_generation_id(generated_genomes);
        islands[generation_island]->set_latest_generation_id(generated_genomes);
        copy->set_group_id(generation_island);
        insert_genome(copy);
    }
}

void OneNasIslandSpeciationStrategy::print(string indent) const {
    // Log::info("%sIslands: \n", indent.c_str());
    // for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
    //     Log::info("%sIsland %d:\n", indent.c_str(), i);
    //     islands[i]->print(indent + "\t");
    // }
}

/**
 * Gets speciation strategy information headers for logs
 */
string OneNasIslandSpeciationStrategy::get_strategy_information_headers() const {
    string info_header = "";
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        info_header.append(",");
        info_header.append("Island_");
        info_header.append(to_string(i));
        info_header.append("_best_fitness");
        info_header.append(",");
        info_header.append("Island_");
        info_header.append(to_string(i));
        info_header.append("_worst_fitness");
    }
    return info_header;
}

/**
 * Gets speciation strategy information values for logs
 */
string OneNasIslandSpeciationStrategy::get_strategy_information_values() const {
    string info_value="";
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        double best_fitness = islands[i]->get_best_fitness();
        double worst_fitness = islands[i]->get_worst_fitness();
        info_value.append(",");
        info_value.append(to_string(best_fitness));
        info_value.append(",");
        info_value.append(to_string(worst_fitness));
    }
    return info_value;
}

RNN_Genome* OneNasIslandSpeciationStrategy::parents_repopulation(
    string method, uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
    function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover, WeightRules* weight_rules
) {
    RNN_Genome* genome = NULL;

    Log::debug("generation island: %d\n", generation_island);
    int32_t parent_island1;
    do {
        parent_island1 = (number_of_islands - 1) * rng_0_1(generator);
    } while (parent_island1 == generation_island);

    Log::debug("parent island 1: %d\n", parent_island1);
    int32_t parent_island2;
    do {
        parent_island2 = (number_of_islands - 1) * rng_0_1(generator);
    } while (parent_island2 == generation_island || parent_island2 == parent_island1);

    Log::debug("parent island 2: %d\n", parent_island2);
    RNN_Genome* parent1 = NULL;
    RNN_Genome* parent2 = NULL;

    // Track whether parents are copies that need deletion
    bool parent1_is_copy = false;
    bool parent2_is_copy = false;
    
    while (parent1 == NULL) {
        if (method.compare("randomParents") == 0) {
            islands[parent_island1]->copy_random_genome(rng_0_1, generator, &parent1);
            parent1_is_copy = true; // This is a copy that needs deletion
        } else if (method.compare("bestParents") == 0) {
            parent1 = islands[parent_island1]->get_best_genome();
            parent1_is_copy = false; // This is a reference, don't delete
        }
    }

    while (parent2 == NULL) {
        if (method.compare("randomParents") == 0) {
            islands[parent_island2]->copy_random_genome(rng_0_1, generator, &parent2);
            parent2_is_copy = true; // This is a copy that needs deletion
        } else if (method.compare("bestParents") == 0) {
            parent2 = islands[parent_island2]->get_best_genome();
            parent2_is_copy = false; // This is a reference, don't delete
        }
    }

    Log::debug(
        "current island is %d, the parent1 island is %d, parent 2 island is %d (parent1_copy: %d, parent2_copy: %d)\n", 
        generation_island, parent_island1, parent_island2, parent1_is_copy, parent2_is_copy
    );

    // swap so the first parent is the more fit parent
    if (parent1->get_fitness() > parent2->get_fitness()) {
        RNN_Genome* tmp = parent1;
        parent1 = parent2;
        parent2 = tmp;
        // Also swap the copy flags
        bool tmp_flag = parent1_is_copy;
        parent1_is_copy = parent2_is_copy;
        parent2_is_copy = tmp_flag;
    }
    genome = crossover(parent1, parent2);

    // CRITICAL FIX: Delete parent copies to prevent memory leaks
    if (parent1_is_copy) {
        Log::debug("Deleting parent1 copy at %p\n", parent1);
        delete parent1;
        parent1 = NULL;
    }
    if (parent2_is_copy) {
        Log::debug("Deleting parent2 copy at %p\n", parent2);
        delete parent2;
        parent2 = NULL;
    }

    mutate(num_mutations, genome);

    if (genome->outputs_unreachable()) {
        // no path from at least one input to the outputs
        delete genome;
        genome = generate_genome(rng_0_1, generator, mutate, crossover, weight_rules);
    }
    return genome;
}
// void OneNasIslandSpeciationStrategy::fill_island(int32_t best_island_id, function<void (int32_t, RNN_Genome*)> &mutate){
//     vector<RNN_Genome*>best_island = islands[best_island_id]->get_genomes();
//     for (int32_t i = 0; i < (int32_t)best_island.size(); i++){
//         // copy the genome from the best island
//         RNN_Genome *copy = best_island[i]->copy();
//         generated_genomes++;
//         copy->set_generation_id(generated_genomes);
//         islands[generation_island] -> set_latest_generation_id(generated_genomes);
//         copy->set_group_id(generation_island);
//         if (repopulation_mutations > 0) {
//             Log::info("Doing %d mutations to genome %d before inserted to the repopulating island\n", repopulation_mutations,copy->generation_id);
//             mutate(repopulation_mutations, copy);
//         }
//         insert_genome(copy);
//         delete copy;
//     }
// }

RNN_Genome* OneNasIslandSpeciationStrategy::get_global_best_genome(){
    return global_best_genome;
}

RNN_Genome* OneNasIslandSpeciationStrategy::select_global_best_genome() {
    RNN_Genome* best_genome = NULL;
    double best_validation_mse = EXAMM_MAX_DOUBLE;
    
    for (int32_t i = 0; i < number_of_islands; i++) {
        // Get the best genome from current island (genomes[0] of elite population)
        RNN_Genome* island_best = islands[i]->get_best_genome();
        
        if (island_best != NULL) {
            double current_mse = island_best->get_best_validation_mse();
            
            // Check if this genome has a better (smaller) validation MSE
            if (current_mse < best_validation_mse) {
                best_validation_mse = current_mse;
                best_genome = island_best;
            }
        }
    }
    
    return best_genome;
}

void OneNasIslandSpeciationStrategy::write_global_best_prediction(int32_t current_generation, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    if (global_best_genome == NULL) {
        Log::error("Cannot write predictions: global_best_genome is NULL\n");
        return;
    }

    // Get the best parameters for the global best genome
    vector<double> parameters = global_best_genome->get_best_parameters();
    if (parameters.size() <= 0) {
        Log::error("Global best genome %d best parameter size is %d\n", global_best_genome->get_generation_id(), parameters.size());
        return;
    }

    string filename = output_directory + "/generation_" + std::to_string(current_generation);

    // Get predictions for the global best genome
    vector< vector< vector<double> > > predictions = global_best_genome->get_predictions(parameters, test_input, test_output);
    
    // Calculate performance comparison between naive and genome predictions (only if comparison is enabled)
    if (compare_with_naive) {
        double naive_mse = 0.0;
        double genome_mse = 0.0;
        if (calculate_prediction_performance(predictions, test_output, naive_mse, genome_mse)) {
            update_performance_counters(current_generation, naive_mse, genome_mse);
        }
    }
    
    // Write predictions to file
    write_prediction_file(filename, predictions, test_input, test_output);
}

void OneNasIslandSpeciationStrategy::write_elite_prediction_file(int32_t current_generation, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    if (test_input.size() == 0 || test_input[0].size() == 0) {
        Log::error("Cannot write elite predictions: test input is empty\n");
        return;
    }

    string filename = output_directory + "/generation_" + std::to_string(current_generation) + "_elites.csv";
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        Log::error("Cannot write elite predictions: could not open %s\n", filename.c_str());
        return;
    }

    outfile << "island,elite_rank,stock,row,predicted\n";

    // Number of test series: 1 in the default mode, one per stock in pooled panel mode.
    // All series share the same window length (contemporaneous windows).
    int32_t num_series = (int32_t)test_input.size();
    int32_t time_length = (int32_t)test_input[0][0].size();

    int64_t rows_written = 0;
    int32_t genomes_written = 0;
    bool multi_output_warned = false;

    for (int32_t island = 0; island < number_of_islands; island++) {
        // elites are kept in fitness order by evaluate_elite_population/select_elite_population,
        // so index 0 is the best genome of the island
        vector<RNN_Genome*> elites = islands[island]->get_genomes();

        for (int32_t elite_rank = 0; elite_rank < (int32_t)elites.size(); elite_rank++) {
            RNN_Genome* genome = elites[elite_rank];
            if (genome == NULL) {
                Log::error("Elite predictions: island %d elite %d is NULL, skipping\n", island, elite_rank);
                continue;
            }

            vector<double> parameters = genome->get_best_parameters();
            if (parameters.size() <= 0) {
                Log::error("Elite predictions: island %d elite %d (genome %d) best parameter size is %d, skipping\n", island, elite_rank, genome->get_generation_id(), (int32_t)parameters.size());
                continue;
            }

            if (!multi_output_warned && genome->get_number_outputs() > 1) {
                Log::warning("Elite predictions: genomes have %d outputs, only the first output is written to %s\n", genome->get_number_outputs(), filename.c_str());
                multi_output_warned = true;
            }

            vector< vector< vector<double> > > predictions = genome->get_predictions(parameters, test_input, test_output);
            if ((int32_t)predictions.size() != num_series) {
                Log::error("Elite predictions: island %d elite %d returned %d series, expected %d, skipping\n", island, elite_rank, (int32_t)predictions.size(), num_series);
                continue;
            }

            for (int32_t n = 0; n < num_series; n++) {
                // rows mirror the data rows of generation_<g>_global_best.csv, which start
                // at timestep 1 (timestep 0 has no naive prediction), so row = j - 1
                for (int32_t j = 1; j < time_length; j++) {
                    outfile << island << "," << elite_rank << "," << n << "," << (j - 1) << "," << predictions[n][0][j] << "\n";
                    rows_written++;
                }
            }
            genomes_written++;
        }
    }

    outfile.close();

    Log::info("Elite genome predictions written to %s (%d genomes, %lld rows)\n", filename.c_str(), genomes_written, (long long)rows_written);
}

void OneNasIslandSpeciationStrategy::set_erased_islands_status() {
    for (int i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i] -> get_erase_again_num() > 0) {
            islands[i] -> set_erase_again_num();
            Log::debug("Island %d can be removed in %d rounds.\n", i, islands[i] -> get_erase_again_num());
        }
    }
}

void OneNasIslandSpeciationStrategy::initialize_population(function<void(int32_t, RNN_Genome*)>& mutate, WeightRules* weight_rules) {
    for (int32_t i = 0; i < number_of_islands; i++) {
        OneNasIsland* new_island = new OneNasIsland(i, generated_population_size, elite_population_size);
        // if (start_filled) {
        //     new_island->fill_with_mutated_genomes(seed_genome, seed_stirs, tl_epigenetic_weights, mutate);
        // }
        islands.push_back(new_island);
    }
    Log::info("OneNAS Speciation Strategy: Initialized %d islands\n", islands.size());
}

void OneNasIslandSpeciationStrategy::finalize_generation(int32_t current_generation, const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    // Just call our PER-specific method since we no longer need to extract IDs
    vector<RNN_Genome*> elite_genomes = finalize_generation_with_genomes(current_generation, validation_input, validation_output, test_input, test_output);
    
    // Clean up the elite genome copies since they're not needed by the base class interface
    for (RNN_Genome* genome : elite_genomes) {
        if (genome != NULL) {
            delete genome;
        }
    }
    
    Log::info("Base class finalize_generation: completed for generation %d\n", current_generation);
}

vector<RNN_Genome*> OneNasIslandSpeciationStrategy::finalize_generation_with_genomes(int32_t current_generation, const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    Log::info("OneNAS Speciation Strategy: Finalizing generation %d\n", current_generation);
    evaluate_elite_population(validation_input, validation_output);
    select_elite_population();
    generation_check();
    
    // Collect all elite genomes from all islands
    vector<RNN_Genome*> elite_genomes;
    for (int32_t i = 0; i < number_of_islands; i++) {
        vector<RNN_Genome*> island_elites = islands[i]->get_genomes();
        for (RNN_Genome* genome : island_elites) {
            if (genome != NULL) {
                // Create a copy of the genome for return
                elite_genomes.push_back(genome->copy());
            }
        }
    }
    
    Log::info("OneNAS: Collected %d elite genomes from %d islands\n", (int32_t)elite_genomes.size(), number_of_islands);
    
    // Safely update global_best_genome with proper memory management
    RNN_Genome* new_best_genome = select_global_best_genome();
    if (new_best_genome != NULL) {
        // Delete the old global_best_genome if it exists to prevent memory leaks
        if (global_best_genome != NULL) {
            Log::debug("Deleting previous global_best_genome at %p\n", global_best_genome);
            delete global_best_genome;
            global_best_genome = NULL;
        }
        
        // Create a copy of the new best genome to own the memory
        global_best_genome = new_best_genome->copy();
        Log::info("Updated global_best_genome to generation %d with validation MSE: %.6f\n", 
                 global_best_genome->get_generation_id(), global_best_genome->get_best_validation_mse());
    } else {
        Log::warning("No best genome found in any island - global_best_genome remains unchanged\n");
    }
    
    write_global_best_prediction(current_generation, test_input, test_output);
    if (write_elite_predictions) {
        write_elite_prediction_file(current_generation, test_input, test_output);
    }
    save_genome(global_best_genome);
    
    // Check if we should trigger network size control (only when compare_with_naive is still enabled)
    if (compare_with_naive && current_generation > 10) {
        if (genome_better_count > naive_better_count) {
            if (control_size_method.compare("none") == 0) {
                // 'none' is a strict no-op: no mutation-rate changes AND no population shrink.
                // Naive-vs-genome tracking keeps running since nothing will ever fire.
                Log::info(
                    "Generation %d: Genome consistently outperforming naive (Genome: %d > Naive: %d), but "
                    "control_size_method is 'none' - size control is a no-op (no rate changes, no population shrink)\n",
                    current_generation, genome_better_count, naive_better_count);
            } else {
                Log::info("=== PERFORMANCE THRESHOLD REACHED ===\n");
                Log::info("Generation %d: Genome consistently outperforming naive (Genome: %d > Naive: %d)\n",
                         current_generation, genome_better_count, naive_better_count);
                Log::info("Triggering network size control method: %s\n", control_size_method.c_str());

                // Apply network size control
                int32_t previous_population_size = generated_population_size;
                control_network_size(control_size_method);
                generated_population_size = (int32_t)(std::floor(generated_population_size * 0.25));
                if (generated_population_size < 1) {
                    generated_population_size = 1;
                }
                Log::info(
                    "SIZE CONTROL FIRED at generation %d: method '%s' applied, generated population size reduced "
                    "from %d to %d (one-shot)\n",
                    current_generation, control_size_method.c_str(), previous_population_size,
                    generated_population_size);

                // Disable further comparisons - this only happens once
                compare_with_naive = false;
                Log::info("Network size control applied. Disabling further naive comparisons.\n");
                Log::info("=== PERFORMANCE CONTROL ACTIVATED ===\n");
            }
        } else {
            Log::info("Generation %d: Performance tracking - Naive: %d, Genome: %d (threshold not reached)\n", 
                     current_generation, naive_better_count, genome_better_count);
        }
    } else {
        Log::info("Generation %d: Genome file saved (performance comparison disabled)\n", current_generation);
    }
    
    if (repopulation_frequency != 0) {
        do_repopulation(current_generation);
    }
    
    // Return elite genomes for priority updates
    Log::info("Returning %d elite genomes for priority updates\n", (int32_t)elite_genomes.size());
    
    return elite_genomes;
}

void OneNasIslandSpeciationStrategy::do_repopulation(int32_t current_generation) {
    if(current_generation > repopulation_frequency * 2 && current_generation % repopulation_frequency == 0 ) {
        Log::info("Current generation: %d, doing repopulation\n", current_generation);
        if (island_ranking_method.compare("EraseWorst") == 0 || island_ranking_method.compare("") == 0){
            // global_best_genome = get_best_genome()->copy();
            vector<int32_t> rank = rank_islands();
            for (int32_t i = 0; i < islands_to_exterminate; i++){
                if (rank[i] >= 0){
                    Log::info("found island: %d is the worst island\n",rank[0]);
                    islands[rank[i]]->erase_island();
                    // islands[rank[i]]->erase_structure_map();
                    islands[rank[i]]->set_status(OneNasIsland::REPOPULATING);
                }
                else Log::error("Didn't find the worst island!");
                // set this so the island would not be re-killed in 5 rounds
                if (!repeat_extinction) {
                    set_erased_islands_status();
                }
            }
        }
    }
    
}

void OneNasIslandSpeciationStrategy::set_onenas_instance(ONENAS* onenas_ref) {
    onenas_instance = onenas_ref;
}

void OneNasIslandSpeciationStrategy::control_network_size(string control_size_method) {
    if (control_size_method.compare("reduce_mutation_rate") == 0) {
        Log::info("Reducing mutation rates to control network size\n");
        mutation_rate = 0.4;
        intra_island_crossover_rate = 0.4;
        inter_island_crossover_rate = 0.2;
        
        // Normalize rates to sum to 1.0
        double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
        if (rate_sum != 1.0) {
            mutation_rate = mutation_rate / rate_sum;
            intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
            inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
        }
        
        // Update cumulative rates for generation
        intra_island_crossover_rate += mutation_rate;
        inter_island_crossover_rate += intra_island_crossover_rate;
        
        Log::info("Updated rates - Mutation: %.3f, Intra-island: %.3f, Inter-island: %.3f\n", 
                 mutation_rate, intra_island_crossover_rate, inter_island_crossover_rate);
                 
    } else if (control_size_method.compare("reduce_add_mutation") == 0) {
        Log::info("Reducing add node/edge mutation rates to control network size\n");
        if (onenas_instance != nullptr) {
            onenas_instance->reduce_add_mutation_rates(0.5);  // Reduce by factor of 0.5
        } else {
            Log::error("ONENAS instance not set. This should not happen.\n");
            exit(1);
        }
                 
    } else if (control_size_method.compare("none") == 0) {
        Log::info("No network size control method applied\n");
    } else {
        Log::warning("Unknown control_size_method: %s. No action taken.\n", control_size_method.c_str());
    }
}

// write function to save genomes to file
void OneNasIslandSpeciationStrategy::save_genome(RNN_Genome* genome) {
    string genomes_dir = output_directory + "/genomes";
    mkpath(genomes_dir.c_str(), 0777);
    genome->write_to_file(genomes_dir + "/" + "rnn_genome" + "_" + to_string(genome->get_generation_id()) + ".bin");
}

void OneNasIslandSpeciationStrategy::generation_check() {
    for (int i = 0; i < number_of_islands; i++) {
        islands[i]->generation_check();
    }
}

int32_t OneNasIslandSpeciationStrategy::number_filled_islands() {
    int32_t n_filled = 0;
    
    for (int32_t i = 0; i < number_of_islands; i++) {
        if (islands[i]->elite_is_full()) {
            n_filled++;
        }
    }
    
    return n_filled;
}

int32_t OneNasIslandSpeciationStrategy::get_other_full_island(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, int32_t first_island
) {
    // select a different island randomly
    Log::debug("other filled islands:\n");
    vector<int32_t> other_filled_islands;
    for (int32_t i = 0; i < number_of_islands; i++) {
        if (i != first_island && islands[i]->elite_is_full()) {
            other_filled_islands.push_back(i);
            Log::debug("\t %d\n", i);
        }
    }
    
    // Check if we have any other filled islands
    if (other_filled_islands.size() == 0) {
        Log::fatal("FATAL: No Other Filled Islands at this moment for inter-island crossover, this should not happen\n");
        exit(1);
        return -1; // Return error code
    }
    
    int32_t other_island = other_filled_islands[rng_0_1(generator) * other_filled_islands.size()];
    Log::debug("other island: %d\n", other_island);
    
    return other_island;
}

void OneNasIslandSpeciationStrategy::evaluate_elite_population(const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output) {
    // vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    for (int i = 0; i < number_of_islands; i++) {
        islands[i] -> evaluate_elite_population(validation_input, validation_output);
    }
}

void OneNasIslandSpeciationStrategy::select_elite_population() {
    for (int i = 0; i < number_of_islands; i++) {
        islands[i] -> select_elite_population();
    }
    
}

RNN_Genome* OneNasIslandSpeciationStrategy::get_seed_genome() {
    return seed_genome;
}

void OneNasIslandSpeciationStrategy::save_entire_population(string output_path) {
    for (int32_t i = 0; i < number_of_islands; i++) {
        islands[i]->save_entire_population(output_path);
    }
}

bool OneNasIslandSpeciationStrategy::calculate_prediction_performance(const vector< vector< vector<double> > > &predictions, const vector< vector< vector<double> > > &test_output, double &naive_mse, double &genome_mse) {
    if (global_best_genome == NULL) {
        Log::error("Cannot calculate performance: global_best_genome is NULL\n");
        return false;
    }

    int32_t num_outputs = global_best_genome->get_number_outputs();

    naive_mse = 0.0;
    genome_mse = 0.0;
    int32_t comparison_count = 0;

    // Average over all test series (a single series in the default mode, one per stock in
    // pooled panel mode)
    for (int32_t n = 0; n < (int32_t)test_output.size(); n++) {
        int32_t time_length = (int32_t)test_output[n][0].size();
        for (int32_t j = 1; j < time_length; j++) {
            for (int32_t i = 0; i < num_outputs; i++) {
                double expected = test_output[n][i][j];
                double naive_pred = test_output[n][i][j-1];  // previous timestep
                double genome_pred = predictions[n][i][j];

                double naive_error = expected - naive_pred;
                double genome_error = expected - genome_pred;

                naive_mse += naive_error * naive_error;
                genome_mse += genome_error * genome_error;
                comparison_count++;
            }
        }
    }
    
    // Calculate average MSE
    if (comparison_count > 0) {
        naive_mse /= comparison_count;
        genome_mse /= comparison_count;
        return true;
    }
    
    return false;
}

void OneNasIslandSpeciationStrategy::update_performance_counters(int32_t current_generation, double naive_mse, double genome_mse) {
    // Update counters based on which prediction method performed better
    if (naive_mse < genome_mse) {
        naive_better_count++;
        Log::info("Generation %d: Naive prediction better (MSE: %.6f vs %.6f). Counters - Naive: %d, Genome: %d\n", 
                 current_generation, naive_mse, genome_mse, naive_better_count, genome_better_count);
    } else {
        genome_better_count++;
        Log::info("Generation %d: Genome prediction better (MSE: %.6f vs %.6f). Counters - Naive: %d, Genome: %d\n", 
                 current_generation, genome_mse, naive_mse, naive_better_count, genome_better_count);
    }
}

void OneNasIslandSpeciationStrategy::write_prediction_file(const string &filename, const vector< vector< vector<double> > > &predictions, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    if (global_best_genome == NULL) {
        Log::error("Cannot write prediction file: global_best_genome is NULL\n");
        return;
    }

    int32_t num_outputs = global_best_genome->get_number_outputs();
    vector<string> output_parameter_names = global_best_genome->get_output_parameter_names();

    // Number of test series: 1 in the default mode, one per stock in pooled panel mode.
    // With a single series the file format is unchanged; with multiple series each stock
    // gets its own column group with an _s<stock> suffix on every column name.
    int32_t num_series = (int32_t)test_input.size();

    // Create output file
    ofstream outfile(filename + "_global_best.csv");
    outfile << "#";

    for (int32_t n = 0; n < num_series; n++) {
        string suffix = (num_series > 1) ? ("_s" + to_string(n)) : "";

        // Write expected output headers
        for (int32_t i = 0; i < num_outputs; i++) {
            if (n > 0 || i > 0) outfile << ",";
            outfile << "expected_" << output_parameter_names[i] << suffix;
        }

        // Write naive prediction headers (previous timestep)
        for (int32_t i = 0; i < num_outputs; i++) {
            outfile << ",";
            outfile << "naive_" << output_parameter_names[i] << suffix;
        }

        // Write global best genome prediction headers
        for (int32_t i = 0; i < num_outputs; i++) {
            outfile << ",";
            outfile << "global_best_predicted_" << output_parameter_names[i] << suffix;
        }
    }

    outfile << endl;

    // Write data rows - all series share the same window length (contemporaneous windows)
    int32_t time_length = (int32_t)test_input[0][0].size();
    for (int32_t j = 1; j < time_length; j++) {
        for (int32_t n = 0; n < num_series; n++) {
            // Write expected values
            for (int32_t i = 0; i < num_outputs; i++) {
                if (n > 0 || i > 0) outfile << ",";
                outfile << test_output[n][i][j];
            }

            // Write naive predictions (previous timestep)
            for (int32_t i = 0; i < num_outputs; i++) {
                outfile << ",";
                outfile << test_output[n][i][j-1];
            }

            // Write global best genome predictions
            for (int32_t i = 0; i < num_outputs; i++) {
                outfile << ",";
                outfile << predictions[n][i][j];
            }
        }
        outfile << endl;
    }
    outfile.close();

    Log::info("Global best genome predictions written to %s_global_best.csv\n", filename.c_str());
}