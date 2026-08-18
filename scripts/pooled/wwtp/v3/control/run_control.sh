#!/bin/bash -l
# run_control.sh -- the FIXED-ARCHITECTURE CONTROL ARM.
#
# ============================================================================
# WHAT THIS IS
# ============================================================================
# The campaign's ONE-NAS arm claims that *architecture search* buys something.
# That claim is untestable without an arm that is identical in every respect
# EXCEPT the search.  This script is that arm.
#
# It runs THE SAME BINARY as the ONE-NAS arm (onenas_mpi) with the evolutionary
# operators switched off:
#
#   --genome_bin <fixed.bin>      hand-picked topology, supplied from outside
#   --num_mutations 0             EXAMM::mutate(0, g) enters its loop, sees
#                                 number_mutations(0) >= max_mutations(0) and
#                                 breaks before touching a single node or edge
#                                 (examm.cxx:284-291).  Topology is frozen.
#   --mutation_rate 1.0           after normalisation in
#   --intra_island_co_rate 0.0    OneNasIslandSpeciationStrategy's ctor
#   --inter_island_co_rate 0.0    (onenas_island_speciation_strategy.cxx:63-71)
#                                 the mutation branch of
#                                 generate_for_filled_island is taken with
#                                 probability 1, so crossover -- the other
#                                 operator that can change topology -- never
#                                 runs.
#   --repopulation_frequency 0    no island extinction/repopulation.
#
# NOTE: --number_islands MUST be > 1.  process_arguments.cxx:177-180 silently
# OVERRIDES intra_island_co_rate to 0.30 when number_islands == 1, which would
# re-enable crossover.  This script refuses ISL=1 for that reason.
#
# Because it is the same binary, the following are matched BY CONSTRUCTION
# rather than by reimplementation -- there is no second implementation to drift:
#   * data ingest and per-file window slicing (no window spans a segment)
#   * time_series_length L and window_step == L
#   * RECURRENT STATE RESET AT EVERY WINDOW.  RNN::forward_pass calls
#     nodes[i]->reset(series_length) for every node, edge and recurrent edge at
#     the top of every window (rnn.cxx:372-381), so at timestep j the network
#     has seen only rows wL..wL+j.  This is the property the reviews found ridge
#     violates; the control cannot violate it because it runs the same code.
#   * the online clock: pool {w < current_index}, validation [cw, cw+V),
#     test cw+V, current_index = generation + T (online_series.cxx:174, 185-212,
#     352-360)
#   * horizon semantics: --time_offset H maps input row k -> target row k+H
#   * PER recency/priority replay, --get_train_data_by PER, same alpha/lambda/eps
#   * bp_iterations per genome per generation
#   * SIMPLE_NODE output -> tanh -> hard bound (-1,1) (rnn_node.cxx:82-88)
#   * normalisation constants (read from the prep manifest, --normalize none)
#   * elite selection, selection_metric, max_pred_sd_ratio
#
# WHAT DIFFERS -- and it is only this:  the topology never changes.  Every
# genome in every island is a copy of the seed topology; only its weights are
# trained.  Island i's champion after generation g is "the hand-picked
# architecture, trained online up to g", which is exactly the control the
# reviews asked for.
#
# ============================================================================
# Required env: DATA (a prep output dir), OUT (run dir), SEED_BIN (seed genome).
# Everything else defaults to the pre-registered primary arm.
set -euo pipefail
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

PY=/apps/anvil/external/apps/conda/2024.02/bin/python3
DATA=${DATA:?set DATA to a prep output directory (v2 or v3)}
OUT=${OUT:?set OUT to the run directory}
SEED_BIN=${SEED_BIN:?set SEED_BIN to a seed genome .bin from make_seed_genome}
ARCH=${ARCH:-$(basename "$SEED_BIN" .bin)}

# --- parameters: IDENTICAL DEFAULTS TO run_v2.sh -----------------------------
L=${L:-144}
T=${T:-80}
V=${V:-5}
H=${H:-72}
ISL=${ISL:-16}
POP=${POP:-5}
# ELITE=1, NOT the ONE-NAS arm's 8.  THIS IS FORCED BY THE BINARY, not a choice.
# Population::insert_genome (population.cxx:135-210) rejects a genome whose
# STRUCTURAL HASH already appears in the population, and RNN_Genome::equals
# (rnn_genome.cxx:1517-1547) compares nodes/edges/recurrent-edges ONLY -- never
# weights.  In the control every genome is structurally identical, so an elite
# population can never hold more than one member: each insert either deletes the
# worse duplicate or is itself rejected.  With --elite_population_size 8 the
# island's status is set to FILLED while elite_is_full() stays false forever, and
# generate_genome falls through all three branches into
#   "ERROR: island was neither initializing, repopulating or full"
# and then recurses without bound (onenas_island_speciation_strategy.cxx:249-281).
# Observed directly: job 20008466 stalled at generation 7 with island elite sizes
# of 2..7 and wrote a 7.4 GB log of that message.
#
# ELITE=1 makes elite_is_full() satisfiable with the single member the dedupe
# allows, so the island is always in a legal state.  It is the degenerate-but-
# correct value: an elite population of 8 exists to preserve ARCHITECTURAL
# diversity inside an island, and the control has exactly one architecture.
# Compute is unaffected: all POP genomes per island are still generated and still
# trained for BP iterations every generation; only how many are RETAINED changes.
ELITE=${ELITE:-1}
BP=${BP:-10}
SEED=${SEED:-42}
GEN=${GEN:-}
NP=${NP:-16}
STEP=$L          # pinned to L, never a knob -- see run_v2.sh

if [ "$ELITE" -gt 1 ] && [ "${ALLOW_DUP_STRUCTURES:-0}" != "1" ]; then
  echo "FATAL: ELITE=$ELITE > 1 with a single fixed architecture deadlocks the"     >&2
  echo "       binary (structural-hash dedupe in Population::insert_genome means"   >&2
  echo "       the elite population can never be full; generate_genome then"        >&2
  echo "       recurses forever).  Use ELITE=1, or set ALLOW_DUP_STRUCTURES=1 if"   >&2
  echo "       you are running a binary whose dedupe has been disabled."            >&2
  exit 1
fi

if [ "$ISL" -lt 2 ]; then
  echo "FATAL: ISL must be >= 2.  process_arguments.cxx:177-180 overrides"        >&2
  echo "       intra_island_co_rate to 0.30 when number_islands == 1, which"      >&2
  echo "       would re-enable topology-changing crossover in the control."       >&2
  exit 1
fi
[ -f "$SEED_BIN" ] || { echo "FATAL: SEED_BIN $SEED_BIN does not exist" >&2; exit 1; }

mkdir -p "$OUT"
mapfile -t SEGFILES < "$DATA/filelist.txt"
INPUTS=(N2O NH4 NO3 PO4 O2_T1 O2_T2 O2_SP AIR_T1 AIR_T2 AIR_BLOWER SS_T1 TEMP INLET_Q SWM)

ARGS=(
  --training_filenames "${SEGFILES[@]}"
  --input_parameter_names "${INPUTS[@]}"
  --output_parameter_names N2O
  --time_offset "$H"
  --time_series_length "$L"
  --window_step "$STEP"
  --num_training_sets "$T" --num_validation_sets "$V"
  --get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8
  --speciation_method onenas
  --number_islands "$ISL" --generated_population_size "$POP"
  --elite_population_size "$ELITE"
  --bp_iterations "$BP"
  # ---- THE CONTROL: search off ---------------------------------------------
  --genome_bin "$SEED_BIN"
  --transfer_learning_version v1
  --num_mutations 0
  --mutation_rate 1.0
  --intra_island_co_rate 0.0
  --inter_island_co_rate 0.0
  # --------------------------------------------------------------------------
  --possible_node_types simple UGRNN MGU GRU delta LSTM
  --rounds_per_generation 1
  --repopulation_frequency 0
  --selection_metric mse
  --max_pred_sd_ratio 3.0
  --online_series_seed "$SEED"
  --normalize none --compare_with_naive --control_size_method none
  # per-island champion predictions, for the rank-mean ensemble of 16 networks
  --write_elite_predictions
  --std_message_level ERROR --file_message_level ERROR
  --output_directory "$OUT"
)
[ -n "$GEN" ] && ARGS+=(--total_generation "$GEN")

BIN="$HOME/ONENAS/build/mpi/onenas_mpi"

# --- the manifest, written BEFORE the run ------------------------------------
# Same schema as run_v2.sh's run_manifest.json (so score_v2.py / score_v3.py read
# it unchanged) plus a "control" block recording exactly what was frozen.
printf '%s\0' "${ARGS[@]}" > "$OUT/.argv"
DATA="$DATA" OUT="$OUT" BIN="$BIN" L="$L" T="$T" V="$V" H="$H" STEP="$STEP" \
ISL="$ISL" POP="$POP" ELITE="$ELITE" BP="$BP" SEED="$SEED" NP="$NP" \
SEED_BIN="$SEED_BIN" ARCH="$ARCH" \
"$PY" - <<'PY'
import hashlib, json, os, subprocess, time
d, out = os.environ["DATA"], os.environ["OUT"]
prep = json.load(open(os.path.join(d, "prep_meta.json")))
argv = [a.decode() for a in open(os.path.join(out, ".argv"), "rb").read().split(b"\0")[:-1]]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def git(*a):
    try:
        return subprocess.check_output(["git", "-C", os.path.expanduser("~/ONENAS")] + list(a),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

i = int
man = {
    "schema": "onenas_wwtp_run_manifest/1",
    "arm": "fixed_architecture_control",
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "params": {
        "L": i(os.environ["L"]), "T": i(os.environ["T"]), "V": i(os.environ["V"]),
        "OFFSET": i(os.environ["H"]), "window_step": i(os.environ["STEP"]),
        "ISL": i(os.environ["ISL"]), "POP": i(os.environ["POP"]),
        "ELITE": i(os.environ["ELITE"]), "BP": i(os.environ["BP"]),
        "seed": i(os.environ["SEED"]), "mpi_ranks": i(os.environ["NP"]),
        "pooled_panel": False, "sampling": "PER",
    },
    # ---- what makes this the CONTROL -----------------------------------------
    "control": {
        "architecture": os.environ["ARCH"],
        "seed_genome_bin": os.environ["SEED_BIN"],
        "seed_genome_sha256": sha(os.environ["SEED_BIN"]),
        "num_mutations": 0,
        "mutation_rate": 1.0,
        "intra_island_co_rate": 0.0,
        "inter_island_co_rate": 0.0,
        "repopulation_frequency": 0,
        "elite_population_size_differs_from_onenas": True,
        "elite_population_size_reason": (
            "Population::insert_genome dedupes by structural hash and "
            "RNN_Genome::equals compares topology only, so a single-architecture "
            "elite population cannot hold more than one member; with ELITE>1 the "
            "island never satisfies elite_is_full() and generate_genome recurses "
            "without bound.  All POP genomes per island are still generated and "
            "trained BP iterations each generation, so the per-generation compute "
            "budget is unchanged; only the number retained differs."),
        "search_disabled_because": (
            "EXAMM::mutate(0, g) breaks before any structural operator "
            "(examm.cxx:284-291); mutation_rate normalises to 1.0 so "
            "generate_for_filled_island never reaches either crossover branch "
            "(onenas_island_speciation_strategy.cxx:336-373); "
            "repopulation_frequency 0 disables island extinction."),
        "ensemble": ("rank-mean over the ISL island champions "
                     "(elite_rank == 0 of each island in generation_<g>_elites.csv); "
                     "ensemble_control.py rewrites them into global_best format"),
    },
    "alignment": {
        "test_episode_of_generation": "g + T + V",
        "episode_to_window": "windows.csv row with window_id == episode",
        "file_row_to_timestep": "data row i (0-based) of generation_<g>_global_best.csv is "
                                "episode timestep j = i+1 (the writer starts at j=1)",
        "timestep_to_target_row": "target seg_row = tgt_start_seg_row + i + 1",
        "timestep_to_issue_row": "forecast-issue seg_row = target seg_row - OFFSET",
        "verification": "expected_N2O[i] must equal index.csv n2o_norm at the target "
                        "(seg_id, seg_row) to 1e-4; the scorer refuses the generation otherwise",
        "denormalise": "raw = pred * stats.N2O.scale + stats.N2O.center",
    },
    "data": {
        "dir": d,
        "prep_meta_sha256": sha(os.path.join(d, "prep_meta.json")),
        "index_csv_sha256": sha(os.path.join(d, "index.csv")),
        "windows_csv_sha256": sha(os.path.join(d, "windows.csv")),
        "mask": prep["mask"],
        "span_start": prep["segments"][0]["t_start"],
        "span_end": prep["segments"][-1]["t_end"],
        "burn_cut_time": prep.get("burn_cut_time"),
        "first_scored_window_time": prep.get("first_scored_window_time"),
        "n_segments": prep["n_segments_used"],
        "n_windows": prep["n_windows"],
        "max_generation": prep["max_generation"],
        "n2o_center": prep["stats"]["N2O"]["center"],
        "n2o_scale": prep["stats"]["N2O"]["scale"],
        "segment_map": [{"seg_id": s["seg_id"], "file": s["file"],
                         "n_rows": s["n_rows"], "n_windows": s["n_windows"],
                         "t_start": s["t_start"], "t_end": s["t_end"]}
                        for s in prep["segments"]],
    },
    "binary": os.environ["BIN"],
    "binary_sha256": sha(os.environ["BIN"]),
    "onenas_git_commit": git("rev-parse", "HEAD"),
    "onenas_git_dirty": bool(git("status", "--porcelain")),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "argv": argv,
}
with open(os.path.join(out, "run_manifest.json"), "w") as fh:
    json.dump(man, fh, indent=2)
print(f"wrote {out}/run_manifest.json  "
      f"(CONTROL arch={man['control']['architecture']}, "
      f"{man['data']['n_segments']} segments, {man['data']['n_windows']} windows, "
      f"max_generation {man['data']['max_generation']})")
PY
rm -f "$OUT/.argv"

echo "=== CONTROL launching: arch=$ARCH L=$L T=$T V=$V H=$H step=$STEP ISL=$ISL POP=$POP ELITE=$ELITE BP=$BP seed=$SEED"
time srun --mpi=pmi2 -n "$NP" "$BIN" "${ARGS[@]}"
echo "=== prediction files: $(ls "$OUT"/generation_*_global_best.csv 2>/dev/null | wc -l)"
echo "=== elite files:      $(ls "$OUT"/generation_*_elites.csv 2>/dev/null | wc -l)"
