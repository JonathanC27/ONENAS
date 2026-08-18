// make_seed_genome.cxx -- build a FIXED architecture RNN_Genome and serialise it to a
// .bin that ONE-NAS can be seeded with via --genome_bin.
//
// WHY THIS EXISTS.  The control arm of the IAAI campaign must be identical to the
// ONE-NAS arm in every respect EXCEPT that the architecture is fixed and hand-picked
// rather than searched.  The cleanest way to guarantee "identical in every other
// respect" is to reuse the ONE-NAS binary itself with the search switched off
// (--num_mutations 0, crossover rates 0), because then the data ingest, the
// windowing, the per-window recurrent-state reset, the online clock, the PER replay,
// the BPTT loop, the elite selection and the tanh-bounded output are literally the
// same code paths.  ONE-NAS's DEFAULT seed genome is create_ff(inputs,0,0,outputs,0)
// -- a 14-input -> 1-output net with no hidden nodes -- so a fixed-architecture
// control needs a seed genome supplied from outside.  That is what this tool makes.
//
// The architecture is ONE fully-connected hidden layer of memory cells whose types are
// listed on the command line, drawn from exactly the vocabulary ONE-NAS searches over
// (simple / UGRNN / MGU / GRU / delta / LSTM).  No recurrent skip edges are added
// (max_recurrent_depth 0); the cells' own internal state is the recurrence, which is
// what keeps the edge count near the range evolved genomes actually reach.
//
// Build:  ~/wwtp_scripts/v3/control/build_seed.sh
// Usage:  make_seed_genome --inputs A B C ... --outputs N2O --hidden LSTM LSTM LSTM LSTM
//                          --out /path/seed.bin [--seed 42]

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "weights/weight_rules.hxx"

// The ONE-NAS --possible_node_types vocabulary, and nothing else.  Refusing anything
// outside it is deliberate: a control built from a node type the search cannot reach
// would hand the control a representational advantage.
static int32_t node_type_from_string(const string& s) {
    if (s == "simple" || s == "SIMPLE") return SIMPLE_NODE;
    if (s == "UGRNN" || s == "ugrnn") return UGRNN_NODE;
    if (s == "MGU" || s == "mgu") return MGU_NODE;
    if (s == "GRU" || s == "gru") return GRU_NODE;
    if (s == "delta" || s == "DELTA") return DELTA_NODE;
    if (s == "LSTM" || s == "lstm") return LSTM_NODE;
    Log::fatal(
        "unknown hidden node type '%s'; the control may only use the ONE-NAS search "
        "vocabulary: simple UGRNN MGU GRU delta LSTM\n", s.c_str());
    exit(1);
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    Log::initialize(arguments);
    Log::set_id("make_seed_genome");

    vector<string> input_parameter_names;
    get_argument_vector(arguments, "--inputs", true, input_parameter_names);
    vector<string> output_parameter_names;
    get_argument_vector(arguments, "--outputs", true, output_parameter_names);
    vector<string> hidden_type_names;
    get_argument_vector(arguments, "--hidden", true, hidden_type_names);
    string out_path;
    get_argument(arguments, "--out", true, out_path);
    int32_t seed = 42;
    get_argument(arguments, "--seed", false, seed);

    vector<int32_t> hidden_types;
    for (int32_t i = 0; i < (int32_t) hidden_type_names.size(); i++) {
        hidden_types.push_back(node_type_from_string(hidden_type_names[i]));
    }

    // Weight initialisation: whatever the run's --weight_initialize says.  ONE-NAS
    // re-randomises the weights of the first genome of every island anyway
    // (generate_for_initializing_island -> initialize_randomly), so the weights stored
    // in this file are never actually used for training; only the TOPOLOGY is.
    WeightRules* weight_rules = new WeightRules(arguments);

    // Stateful node factory: hand create_nn a closure that walks the requested type list
    // so a single hidden layer can mix cell types, exactly as an evolved genome can.
    int32_t next = 0;
    std::function<RNN_Node_Interface*(int32_t&, double)> make_node =
        [&](int32_t& innovation_counter, double depth) -> RNN_Node_Interface* {
            int32_t kind = hidden_types[next % (int32_t) hidden_types.size()];
            next++;
            return create_hidden_node(kind, innovation_counter, depth);
        };

    RNN_Genome* genome = create_nn(
        input_parameter_names,
        /* number_hidden_layers */ 1,
        /* number_hidden_nodes  */ (int32_t) hidden_types.size(),
        output_parameter_names,
        /* max_recurrent_depth  */ 0,
        make_node, weight_rules);

    // initialize_randomly() also calls set_best_parameters(initial_parameters), which is
    // required: get_seed_genome() -> transfer_to() does set_weights(best_parameters)
    // first thing, and RNN_Genome::write_to_file serialises best_parameters.
    genome->set_generation_id(0);
    genome->set_group_id(0);
    genome->initialize_randomly(weight_rules);
    genome->write_to_file(out_path);

    // Read it straight back and print the topology census, so the caller can put the
    // node/edge counts in the manifest without trusting this program's arithmetic.
    RNN_Genome* check = new RNN_Genome(out_path);
    cout << "wrote " << out_path << endl;
    cout << "  nodes            " << check->get_enabled_node_count() << " enabled / "
         << check->get_node_count() << " total" << endl;
    cout << "  edges            " << check->get_enabled_edge_count() << " enabled" << endl;
    cout << "  recurrent_edges  " << check->get_enabled_recurrent_edge_count() << " enabled" << endl;
    cout << "  weights          " << check->get_number_weights() << endl;
    cout << "  node census      " << check->get_node_count_str(-1) << endl;
    cout << "  edge census      " << check->get_edge_count_str(false) << endl;
    cout << "  inputs           " << input_parameter_names.size() << endl;
    cout << "  outputs          " << output_parameter_names.size() << endl;
    cout << "  hidden           ";
    for (int32_t i = 0; i < (int32_t) hidden_type_names.size(); i++) cout << hidden_type_names[i] << " ";
    cout << endl;

    return 0;
}
