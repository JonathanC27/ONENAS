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

#include <iostream>
using std::endl;

#include <unordered_map>
using std::unordered_map;

#include "population.hxx"
#include "rnn/rnn_genome.hxx"

#include "common/log.hxx"

Population::Population(int32_t _population_type, int32_t _max_size, int32_t _island_id) :  population_type(_population_type), max_size(_max_size), island_id(_island_id) {
}

Population::~Population() {
    // Clean up all genome memory to prevent memory leaks
    Log::debug("Destructor: Cleaning up population (type: %d, island: %d) with %d genomes\n", 
               population_type, island_id, (int32_t)genomes.size());
    erase_population(); // This already properly deletes all genomes and clears structure_map
    Log::debug("Population destructor completed for island %d\n", island_id);
}

RNN_Genome* Population::get_best_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes[0];
}

RNN_Genome* Population::get_worst_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes.back();
}

double Population::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double Population::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

int32_t Population::get_max_size() const {
    return max_size;
}

int32_t Population::get_population_size() const {
    return (int32_t)genomes.size();
}

int32_t Population::size() {
    return (int32_t)genomes.size();
}

bool Population::is_full() {
    return (int32_t)genomes.size() >= max_size;
}

bool Population::is_empty() {
    return genomes.size() == 0;
}

void Population::copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome) {
    if (genomes.size() == 0) {
        Log::fatal("ERROR: Attempting to copy random genome from empty population (type: %d, island: %d)\n", population_type, island_id);
        exit(1);
    }
    int32_t genome_position = size() * rng_0_1(generator);
    
    // Additional safety check for the genome itself
    if (genome_position < 0 || genome_position >= (int32_t)genomes.size()) {
        Log::fatal("ERROR: Invalid genome position %d in population of size %d (type: %d, island: %d)\n", genome_position, genomes.size(), population_type, island_id);
        exit(1);
    }
    *genome = genomes[genome_position]->copy();
}

void Population::copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2) {
    if (genomes.size() < 2) {
        Log::fatal("ERROR: Attempting to copy two random genomes from population with less than 2 genomes (size: %d, type: %d, island: %d)\n", genomes.size(), population_type, island_id);
        *genome1 = NULL;
        *genome2 = NULL;
        return;
    }
    int32_t p1 = size() * rng_0_1(generator);
    int32_t p2 = (size() - 1) * rng_0_1(generator);
    if (p2 >= p1) p2++;

    //swap the gnomes so that the first parent is the more fit parent
    if (p1 > p2) {
        int32_t tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    *genome1 = genomes[p1]->copy();
    *genome2 = genomes[p2]->copy();
}


//returns -1 for not inserted, otherwise the index it was inserted at
//inserts a copy of the genome, caller of the function will need to delete their
//pointer
int32_t Population::insert_genome(RNN_Genome *genome) {
    // int32_t initial_size = (int32_t)genomes.size();
    Log::info("inserting genome to island %d, population %s\n", island_id, population_type == GENERATED ? "generated" : "elite");
    Log::debug("getting fitness of genome copy\n");
    double new_fitness = genome->get_fitness();

    //discard the genome if the island is full and it's fitness is worse than the worst in thte population
    if (is_full() && new_fitness > get_worst_fitness()) {
        Log::info("ignoring genome, fitness: %lf > worst for island[%d] fitness: %lf\n", new_fitness, population_type, genomes.back()->get_fitness());
        // do_population_check(__LINE__, initial_size);
        return false;
    }

    //check and see if the structural hash of the genome is in the
    //set of hashes for this population
    string structural_hash = genome->get_structural_hash();

    if (structure_map.count(structural_hash) > 0) {
        vector<RNN_Genome*> &potential_matches = structure_map.find(structural_hash)->second;

        Log::info("potential duplicate for hash '%s', had %d potential matches.\n", structural_hash.c_str(), potential_matches.size());

        for (auto potential_match = potential_matches.begin(); potential_match != potential_matches.end(); ) {
            Log::info("on potential match %d of %d\n", potential_match - potential_matches.begin(), potential_matches.size());
            if ((*potential_match)->equals(genome)) {
                if ((*potential_match)->get_fitness() > new_fitness) {
                    Log::info("REPLACING DUPLICATE GENOME, fitness of genome in search: %s, new fitness: %s\n", parse_fitness((*potential_match)->get_fitness()).c_str(), parse_fitness(genome->get_fitness()).c_str());
                    //we have an exact match for this genome in the island and its fitness is worse
                    //than the genome we're trying to remove, so remove the duplicate it from the genomes
                    //as well from the potential matches vector

                    auto duplicate_genome_iterator = lower_bound(genomes.begin(), genomes.end(), *potential_match, sort_genomes_by_fitness());
                    bool found = false;
                    for (; duplicate_genome_iterator != genomes.end(); duplicate_genome_iterator++) {
                        Log::debug("duplicate_genome_iterator: %p, (*potential_match): %p\n", (*duplicate_genome_iterator), (*potential_match));

                        if ((*duplicate_genome_iterator) == (*potential_match)) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        Log::fatal("ERROR: could not find duplicate genome even though its structural hash was in the island, this should never happen!\n");
                        exit(1);
                    }

                    Log::debug("potential_match->get_fitness(): %lf, duplicate_genome_iterator->get_fitness(): %lf, new_fitness: %lf\n", (*potential_match)->get_fitness(), (*duplicate_genome_iterator)->get_fitness(), new_fitness);

                    int32_t duplicate_genome_index = duplicate_genome_iterator - genomes.begin();
                    Log::debug("duplicate_genome_index: %d\n", duplicate_genome_index);
                    //int32_t test_index = contains(genome);
                    //Log::info("test_index: %d\n", test_index);

                    RNN_Genome *duplicate = genomes[duplicate_genome_index];

                    //Log::info("duplicate.equals(potential_match)? %d\n", duplicate->equals(*potential_match));

                    genomes.erase(genomes.begin() + duplicate_genome_index);

                    Log::debug("potential_matches.size() before erase: %d\n", potential_matches.size());

                    //erase the potential match from the structure map as well
                    //returns an iterator to next element after the deleted one so
                    //we don't need to increment it
                    potential_match = potential_matches.erase(potential_match);

                    delete duplicate;

                    Log::debug("potential_matches.size() after erase: %d\n", potential_matches.size());
                    Log::debug("structure_map[%s].size() after erase: %d\n", structural_hash.c_str(), structure_map[structural_hash].size());

                    if (potential_matches.size() == 0) {
                        Log::debug("deleting the potential_matches vector for hash '%s' because it was empty.\n", structural_hash.c_str());
                        structure_map.erase(structural_hash);
                        break; //break because this vector is now empty and deleted
                    }

                } else {
                    Log::info("island already contains a duplicate genome with a better fitness! not inserting.\n");
                    // do_population_check(__LINE__, initial_size);
                    return -1;
                }
            } else {
                //increment potential match because we didn't delete an entry (or return from the method)
                potential_match++;
            }
        }
    }

    //inorder insert the new individual
    RNN_Genome *copy = genome->copy();

    vector<double> best = copy -> get_best_parameters();
    if(best.size() != 0){
        copy->set_weights(best);
    }
    copy -> set_generation_id (genome -> get_generation_id());
    Log::debug("created copy to insert to island: %d\n", copy->get_group_id());
    auto index_iterator = upper_bound(genomes.begin(), genomes.end(), copy, sort_genomes_by_fitness());
    int32_t insert_index = index_iterator - genomes.begin();
    Log::info("inserting genome at index: %d\n", insert_index);

    if (insert_index >= max_size) {
        //if we're going to insert this at the back of the population
        //its just going to get removed anyways, so we can delete 
        //it and report it was not inserted.
        delete copy;
        // do_population_check(__LINE__, initial_size);
        return -1;
    }

    genomes.insert(index_iterator, copy);
    //calculate the index the genome was inseretd at from the iterator

    structural_hash = copy->get_structural_hash();
    //add the genome to the vector for this structural hash
    structure_map[structural_hash].push_back(copy);
    Log::debug("adding to structure_map[%s] : %p\n", structural_hash.c_str(), &copy);

    if (insert_index == 0) {
        //this was a new best genome for this island

        Log::debug("new best fitness for island: %d!\n", population_type);

        //need to set the weights for non-initial genomes so we
        //can generate a proper graphviz file.
        //
        //This used to be guarded by `get_fitness() != EXAMM_MAX_DOUBLE`, using the fitness
        //sentinel as a proxy for "this genome has trained weights". That proxy only holds while
        //fitness IS the validation MSE -- under --selection_metric ic the fitness is derived from
        //the IC instead, so a freshly generated genome could pass the guard with an empty
        //best_parameters vector and take down the whole run inside set_weights(). Test the thing
        //we actually depend on instead: that best_parameters matches this genome's weight count.
        vector<double> best_parameters = genome->get_best_parameters();
        if ((int32_t) best_parameters.size() == genome->get_number_weights()) {
            genome->set_weights(best_parameters);
        }
    }

    // Log::info("genomes.size(): %d, max_size: %d, status: %d\n", genomes.size(), max_size, status);

    if ((int32_t)genomes.size() > max_size) {
        //island was full before insert so now we need to 
        //delete the worst genome in the island.

        Log::debug("deleting worst genome\n");
        RNN_Genome *worst = genomes.back();
        genomes.pop_back();
        structural_hash = worst->get_structural_hash();

        vector<RNN_Genome*> &potential_matches = structure_map.find(structural_hash)->second;

        bool found = false;
        for (auto potential_match = potential_matches.begin(); potential_match != potential_matches.end(); ) {
            //make sure the addresses of the pointers are the same
            Log::info("checking to remove worst from structure_map - &worst: %p, &(*potential_match): %p\n", worst, (*potential_match));
            if ((*potential_match) == worst) {
                found = true;
                Log::debug("potential_matches.size() before erase: %d\n", potential_matches.size());

                //erase the potential match from the structure map as well
                potential_match = potential_matches.erase(potential_match);

                Log::debug("potential_matches.size() after erase: %d\n", potential_matches.size());
                Log::debug("structure_map[%s].size() after erase: %d\n", structural_hash.c_str(), structure_map[structural_hash].size());

                //clean up the structure_map if no genomes in the population have this hash
                if (potential_matches.size() == 0) {
                    Log::info("deleting the potential_matches vector for hash '%s' because it was empty.\n", structural_hash.c_str());
                    structure_map.erase(structural_hash);
                    break;
                }
            } else {
                potential_match++;
            }
        }

        if (!found) {
            Log::info("could not erase from structure_map[%s], genome not found! This should never happen.\n", structural_hash.c_str());
            exit(1);
        }

        delete worst;
    }

    if (insert_index >= max_size) {
        //technically we shouldn't get here but it might happen
        //if the genome's fitness == the worst fitness in the
        //island. So in this case it was not inserted to the
        //island and return -1
        // do_population_check(__LINE__, initial_size);
        return -1;
    } else {
        // do_population_check(__LINE__, initial_size);
        return insert_index;
    }
}

void Population::print(string indent) {
    if (Log::at_level(Log::INFO)) {

        Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());

        for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
            Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
        }
    }
}

vector<RNN_Genome *> Population::get_genomes() {
    return genomes;
}

void Population::erase_population() {
    RNN_Genome* genome;

    while (genomes.size() > 0) {
        genome = genomes.back();
        genomes.pop_back();
        delete genome;
    }

    erase_structure_map();
}

void Population::erase_structure_map() {
    Log::debug("Erasing the structure map in the worst performing island\n");
    structure_map.clear();
    Log::debug("after erase structure map size is %d\n", structure_map.size());
}

void Population::sort_population(string sort_by) {
    // "MSE" is the historical spelling; both sort by RNN_Genome::get_fitness(), which is the
    // validation MSE unless --selection_metric selects an IC-based metric. "fitness" is the
    // metric-agnostic spelling used by the newer call sites.
    if (sort_by.compare("MSE") == 0 || sort_by.compare("fitness") == 0) {
        sort (genomes.begin(), genomes.end(), sort_genomes_by_fitness());
    } else {
        Log::fatal("Invalid sort by parameter: %s\n", sort_by.c_str());
        exit(1);
    }
    Log::debug("sorted sequence: \n");
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        Log::debug("# %d genome fitness %f\n", i, genomes[i]->get_fitness());
    }

}

void Population::write_prediction(string filename, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    vector <vector <vector <vector<double>>>> predictions; // <genome <input sets < output parameter <value>>>>
    int32_t num_genomes = (int32_t)genomes.size();
    int32_t num_outputs = genomes[0]->get_number_outputs();
    vector <string> input_parameter_names = genomes[0]->get_input_parameter_names();
    vector <string> output_parameter_names = genomes[0]->get_output_parameter_names();
    for (int32_t i = 0; i < num_genomes; i++) {
        vector<double> parameters = genomes[i]->get_best_parameters();
        if (parameters.size() <= 0) {
            Log::error("Genome %d best parameter size is %d\n", genomes[i]->get_generation_id(), parameters.size());
        }

        predictions.push_back(genomes[i]->get_predictions(parameters, test_input, test_output));
    }
    ofstream outfile(filename + "_island_" + std::to_string(island_id) + ".csv");
    outfile << "#";

    // ofstream outfile_original(filename + "_original.csv");
    // outfile_original << "#";

    for (int32_t i = 0; i < num_outputs; i++) {
        if (i > 0) outfile << ",";
        // if (i > 0) outfile_original << ",";

        outfile << "expected_" << output_parameter_names[i];
        // outfile_original << "expected_" << output_parameter_names[i];
        Log::debug("output_parameter_names[%d]: '%s'\n", i, output_parameter_names[i].c_str());
    }
    
    for (int32_t i = 0; i < num_outputs; i++) {
        outfile << ",";
        // outfile_original << ",";

        outfile << "naive_" << output_parameter_names[i];
        // outfile_original << "naive_" << output_parameter_names[i];
        Log::debug("output_parameter_names[%d]: '%s'\n", i, output_parameter_names[i].c_str());
    }

    for (int32_t g = 0; g < num_genomes; g++) {
        for (int32_t i = 0; i < num_outputs; i++) {
            outfile << ",";
            // outfile_original << ",";

            outfile << "genome_" << g << "_predicted_" << output_parameter_names[i];
            // outfile_original << "genome_" << g << "_predicted_" << output_parameter_names[i];
            Log::debug("output_parameter_names[%d]: '%s'\n", i, output_parameter_names[i].c_str());
        }
    }

    outfile << endl;
    // outfile_original << endl;

    int32_t time_length = (int32_t)test_input[0][0].size();
    for (int32_t j = 1; j < time_length; j++) {

        for (int32_t i = 0; i < num_outputs; i++) {
            if (i > 0) outfile << ",";
            // if (i > 0) outfile_original << ",";

            outfile << test_output[0][i][j];
            // outfile_original << time_series_sets->denormalize(output_parameter_names[i], test_output[0][i][j]);
        }

        for (int32_t i = 0; i < num_outputs; i++) {
            outfile << ",";
            // outfile_original << ",";

            outfile << test_output[0][i][j-1];
            // outfile_original << time_series_sets->denormalize(output_parameter_names[i], test_output[0][i][j-1]);
        }

        for (int32_t g = 0; g < num_genomes; g++) {
            for (int32_t i = 0; i < num_outputs; i++) {
                outfile << ",";
                // outfile_original << ",";

                outfile << predictions[g][0][i][j];
                // outfile_original << time_series_sets->denormalize(output_parameter_names[i], predictions[g][0][i][j]);
            }
        }
        outfile << endl;
        // outfile_original << endl;
    }
    outfile.close();
    // outfile_original.close();
}

void Population::save_entire_population(string output_path) {
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        genomes[i]->write_to_file(output_path + "/genome_" + to_string(i) + ".bin");
        genomes[i]->write_graphviz(output_path + "/genome_" + to_string(i) + ".gv");
    }
}
