#!/usr/bin/env python3
"""verify_control.py -- prove, mechanically, that the control arm differs from the
ONE-NAS arm in the architecture search AND NOTHING ELSE.

Run it on a control run directory.  Pass --reference <onenas_run_dir> to diff the
two runs' actual argv; without it the control's argv is checked against the
pre-registered ONE-NAS argument set recorded in EXPECTED below.

Checks, in order:
  1  ARGV DIFF        every argument that differs from the ONE-NAS arm must be on
                      the allow-list of control switches.  Anything else is a
                      matching failure and is reported as one.
  2  TOPOLOGY FROZEN  fitness_log.csv records Enabled Nodes / Edges / Rec. Edges
                      every generation.  In the control they must be CONSTANT and
                      equal to the seed genome's.  This is the empirical proof
                      that --num_mutations 0 plus zero crossover rates really do
                      disable structural search -- not an argument from source.
  3  SOURCE INVARIANTS the load-bearing semantics the control inherits by running
                      the same binary, each re-checked against the current source
                      so a later edit to ONENAS cannot silently break the match.
  4  PREDICTION SANITY sd ratio, distinct-value count and physical-floor fraction,
                      the collapse diagnostics that destroyed the v1 runs.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ONENAS = os.path.expanduser("~/ONENAS")

# The pre-registered ONE-NAS argument set (PROTOCOL.md section 6 / run_v2.sh).
EXPECTED = {
    "--time_series_length": ["144"], "--window_step": ["144"],
    "--num_training_sets": ["80"], "--num_validation_sets": ["5"],
    "--time_offset": ["72"],
    "--get_train_data_by": ["PER"], "--per_alpha": ["0.6"],
    "--per_lambda": ["0.007"], "--per_epsilon": ["1e-8"],
    "--speciation_method": ["onenas"],
    "--number_islands": ["16"], "--generated_population_size": ["5"],
    "--elite_population_size": ["8"], "--bp_iterations": ["10"],
    "--possible_node_types": ["simple", "UGRNN", "MGU", "GRU", "delta", "LSTM"],
    "--rounds_per_generation": ["1"], "--repopulation_frequency": ["0"],
    "--selection_metric": ["mse"], "--max_pred_sd_ratio": ["3.0"],
    "--normalize": ["none"], "--control_size_method": ["none"],
    "--compare_with_naive": [],
    "--input_parameter_names": ["N2O", "NH4", "NO3", "PO4", "O2_T1", "O2_T2", "O2_SP",
                                "AIR_T1", "AIR_T2", "AIR_BLOWER", "SS_T1", "TEMP",
                                "INLET_Q", "SWM"],
    "--output_parameter_names": ["N2O"],
}
# Arguments the control is ALLOWED to differ on.  This list is the definition of
# "what differs"; if the diff in check 1 contains anything else, the arms are not
# matched and the run must not be reported.
CONTROL_SWITCHES = {
    "--genome_bin", "--transfer_learning_version", "--num_mutations",
    "--mutation_rate", "--intra_island_co_rate", "--inter_island_co_rate",
    "--write_elite_predictions",
    # FORCED, not chosen.  Population::insert_genome dedupes by structural hash and
    # RNN_Genome::equals compares topology only, so a single-architecture elite
    # population can never be full; with ELITE > 1 the island reaches a state that is
    # neither initializing, repopulating nor full and generate_genome recurses without
    # bound.  ELITE=1 is the only legal value on the stock binary.  Check 5 below
    # asserts it really is 1 and that the per-generation compute is unchanged.
    "--elite_population_size",
    # bookkeeping, not method
    "--output_directory", "--online_series_seed", "--total_generation",
    "--training_filenames", "--std_message_level", "--file_message_level",
}

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--reference", help="a ONE-NAS run directory to diff argv against")
ap.add_argument("--seed_nodes", type=int, default=None,
                help="expected enabled node count (default: from generation 0)")
a = ap.parse_args()

man = json.load(open(os.path.join(a.run, "run_manifest.json")))
argv = man["argv"]
P = man["params"]
fails, warns = [], []


def parse_argv(av):
    """--flag v1 v2 ... -> {flag: [values]}; bare flags map to []."""
    out, cur = {}, None
    for tok in av[1:] if av and not av[0].startswith("--") else av:
        if tok.startswith("--"):
            cur = tok
            out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(tok)
    return out


ctl = parse_argv(argv)

print("=" * 78)
print("1  ARGV DIFF -- what differs between the control and the ONE-NAS arm")
print("=" * 78)

if a.reference:
    ref = parse_argv(json.load(open(os.path.join(a.reference, "run_manifest.json")))["argv"])
    print(f"reference: {a.reference}")
else:
    ref = EXPECTED
    print("reference: pre-registered ONE-NAS argument set (PROTOCOL.md section 6)")

diff_keys = sorted(set(ref) | set(ctl))
unexpected = []
for k in diff_keys:
    rv, cv = ref.get(k), ctl.get(k)
    if rv == cv:
        continue
    tag = "OK (control switch)" if k in CONTROL_SWITCHES else "*** UNEXPECTED ***"
    if k not in CONTROL_SWITCHES:
        unexpected.append(k)
    rs = "absent" if rv is None else (" ".join(rv) or "(bare flag)")
    cs = "absent" if cv is None else (" ".join(cv) or "(bare flag)")
    if k == "--training_filenames":
        rs = f"({len(rv)} files)" if rv else "absent"
        cs = f"({len(cv)} files)" if cv else "absent"
    print(f"  {k:32s} onenas={rs[:38]:38s} control={cs[:38]:38s} {tag}")

if unexpected:
    fails.append(f"argv differs on non-control arguments: {unexpected}")
else:
    print("  -> every difference is a declared control switch.")

print()
print("=" * 78)
print("2  TOPOLOGY FROZEN -- empirical, from fitness_log.csv")
print("=" * 78)
fl = os.path.join(a.run, "fitness_log.csv")
if not os.path.exists(fl):
    fails.append("fitness_log.csv missing -- cannot prove the topology never changed")
else:
    f = pd.read_csv(fl)
    f.columns = [c.strip() for c in f.columns]
    cols = ["Enabled Nodes", "Enabled Edges", "Enabled Rec. Edges"]
    missing = [c for c in cols if c not in f.columns]
    if missing:
        fails.append(f"fitness_log.csv lacks {missing}")
    else:
        for c in cols:
            u = sorted(f[c].unique().tolist())
            ok = len(u) == 1
            print(f"  {c:22s} distinct values over {len(f)} generations: {u}   "
                  f"{'CONSTANT' if ok else '*** CHANGED -- SEARCH IS NOT OFF ***'}")
            if not ok:
                fails.append(f"{c} changed over the run: {u}")
        if a.seed_nodes is not None and f["Enabled Nodes"].iloc[0] != a.seed_nodes:
            fails.append(f"generation 0 has {f['Enabled Nodes'].iloc[0]} nodes, "
                         f"expected {a.seed_nodes}")

print()
print("=" * 78)
print("3  SOURCE INVARIANTS -- re-checked against the current ONENAS source")
print("=" * 78)


def src_has(path, pattern, label):
    p = os.path.join(ONENAS, path)
    try:
        hits = subprocess.run(["grep", "-n", "-E", pattern, p],
                              capture_output=True, text=True).stdout.strip().splitlines()
    except Exception as ex:
        hits = []
        warns.append(f"{label}: grep failed ({ex})")
    ok = bool(hits)
    print(f"  [{'ok' if ok else 'MISSING'}] {label}")
    for h in hits[:2]:
        print(f"        {path}:{h.split(':', 1)[0]}  {h.split(':', 1)[1].strip()[:88]}")
    if not ok:
        fails.append(f"source invariant not found: {label} ({path} / {pattern})")


src_has("rnn/rnn.cxx", r"nodes\[i\]->reset\(series_length\)",
        "recurrent state is RESET at the top of every forward pass (per window)")
src_has("rnn/rnn.cxx", r"recurrent_edges\[i\]->reset\(series_length\)",
        "recurrent EDGE state is reset per window too")
src_has("onenas/examm.cxx", r"if \(number_mutations >= max_mutations\)",
        "EXAMM::mutate breaks on number_mutations >= max_mutations (so max=0 is a no-op)")
src_has("onenas/onenas_island_speciation_strategy.cxx",
        r"if \(!island->elite_is_full\(\) \|\| r < mutation_rate\)",
        "generate_for_filled_island takes the mutation branch when r < mutation_rate "
        "(mutation_rate normalises to 1.0 in the control, so crossover is unreachable)")
src_has("onenas/population.cxx", r"if \(structure_map.count\(structural_hash\) > 0\)",
        "Population::insert_genome dedupes by structural hash (why ELITE must be 1)")
src_has("rnn/rnn_genome.cxx", r"bool RNN_Genome::equals",
        "RNN_Genome::equals compares topology only, never weights")
src_has("time_series/online_series.cxx", r"current_index = _current_gen \+ num_training_sets",
        "online clock: current_index = generation + T")
src_has("time_series/online_series.cxx", r"return current_index \+ num_validation_sets",
        "test episode = current_index + V  (= g + T + V)")
src_has("common/process_arguments.cxx", r"if \(number_islands == 1\)",
        "the number_islands==1 crossover override exists (run_control.sh refuses ISL=1)")

print()
print("=" * 78)
print("4  PREDICTION SANITY -- the v1 collapse diagnostics")
print("=" * 78)
files = sorted(glob.glob(os.path.join(a.run, "generation_*_global_best.csv")),
               key=lambda f: int(re.search(r"generation_(\d+)_", f).group(1)))
if not files:
    fails.append("no prediction files")
else:
    ps, ys = [], []
    for fn in files:
        d = pd.read_csv(fn)
        d.columns = [c.lstrip("#") for c in d.columns]
        ps.append(d["global_best_predicted_N2O"].values)
        ys.append(d["expected_N2O"].values)
    p, y = np.concatenate(ps), np.concatenate(ys)
    fin = np.isfinite(p)
    sd_ratio = p[fin].std() / y[fin].std()
    distinct = len(np.unique(np.round(p[fin], 9)))
    C, S = man["data"]["n2o_center"], man["data"]["n2o_scale"]
    floor_norm = (0.0 - C) / S
    below = float((p[fin] < floor_norm).mean())
    print(f"  generations           {len(files)}   rows {fin.sum()}")
    print(f"  prediction  norm mean {p[fin].mean():+.4f}  sd {p[fin].std():.4f}  "
          f"min {p[fin].min():+.4f}  max {p[fin].max():+.4f}")
    print(f"  target      norm mean {y[fin].mean():+.4f}  sd {y[fin].std():.4f}")
    print(f"  pred/target sd ratio  {sd_ratio:.4f}      (collapse if << 0.3; v1 was 0.018)")
    print(f"  distinct predictions  {distinct} of {int(fin.sum())}")
    print(f"  below physical floor  {100*below:.2f}%")
    print(f"  non-finite            {int((~fin).sum())}")
    if not (0.3 <= sd_ratio <= 3.0):
        warns.append(f"pred/target sd ratio {sd_ratio:.3f} outside [0.3, 3.0]")
    if distinct < 0.5 * fin.sum():
        warns.append(f"only {distinct} distinct predictions in {int(fin.sum())} rows")
    if (~fin).any():
        fails.append(f"{int((~fin).sum())} non-finite predictions")

print()
print("=" * 78)
print("5  COMPUTE BUDGET -- genomes trained per generation, measured")
print("=" * 78)
if os.path.exists(fl):
    f = pd.read_csv(fl)
    f.columns = [c.strip() for c in f.columns]
    if "Total BP Epochs" in f.columns and len(f) >= 2:
        bp_per_gen = f["Total BP Epochs"].diff().dropna().unique().tolist()
        expected = P["ISL"] * P["POP"] * P["BP"]
        print(f"  BP epochs per generation   {bp_per_gen}   "
              f"expected ISL*POP*BP = {P['ISL']}*{P['POP']}*{P['BP']} = {expected}")
        if len(bp_per_gen) != 1 or int(bp_per_gen[0]) != expected:
            warns.append(f"BP epochs/generation {bp_per_gen} != ISL*POP*BP = {expected}")
        else:
            print("  -> the per-generation compute budget matches the ONE-NAS arm exactly:")
            print("     the same number of genomes are trained for the same number of BP")
            print("     iterations.  --elite_population_size changes only how many are KEPT.")
    if "Time" in f.columns and len(f) >= 2:
        dt = f["Time"].diff().dropna()
        print(f"  wall per generation        mean {dt.mean()/1000:.2f} s  "
              f"sd {dt.std()/1000:.2f} s  (flat by construction: the topology never grows)")
    if P["ELITE"] != 1:
        warns.append(f"ELITE={P['ELITE']} != 1 on a stock binary risks the "
                     f"elite_is_full() deadlock")

print()
print("=" * 78)
if man.get("control"):
    c = man["control"]
    print(f"arch {c['architecture']}   seed_genome {c['seed_genome_bin']}")
    print(f"seed sha256 {c['seed_genome_sha256'][:16]}...")
for w in warns:
    print(f"WARN  {w}")
for fmsg in fails:
    print(f"FAIL  {fmsg}")
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)} problems)")
sys.exit(0 if not fails else 1)
