#!/bin/bash -l
# run_v2.sh -- launch ONE-NAS on the v2 segmented data and write a machine-readable
# record of EVERYTHING the scorer needs, BEFORE the binary starts.
#
# WHY THIS EXISTS.  In v1 the scorer took L, T, V from environment variables and
# then had to SEARCH for the row offset that made each generation's `expected`
# column match the sidecar.  For several runs no offset matched the parameters the
# sbatch advertised, so those runs could not be scored at all -- the run and the
# record of the run had drifted apart.  Here the run writes its own parameters,
# the ordered segment map, and the episode->row rule into run_manifest.json at
# launch; score_v2.py reads that file and takes NOTHING from the environment.
#
# Required env: DATA (a prep_v2 output dir), OUT (run dir).
# Everything else defaults to the pre-registered primary arm in PROTOCOL.md.
set -euo pipefail
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

PY=/apps/anvil/external/apps/conda/2024.02/bin/python3
DATA=${DATA:?set DATA to a prep_v2 output directory}
OUT=${OUT:?set OUT to the run directory}

# --- parameters (defaults = pre-registered primary arm) -----------------------
L=${L:-144}          # --time_series_length ; ALSO the window step (see below)
T=${T:-80}           # --num_training_sets  : burn-in windows AND per-generation batch
V=${V:-5}            # --num_validation_sets
H=${H:-72}           # --time_offset        : the forecast horizon, in 5-min bins
ISL=${ISL:-16}       # --number_islands
POP=${POP:-5}        # --generated_population_size
ELITE=${ELITE:-8}    # --elite_population_size
BP=${BP:-10}         # --bp_iterations
SEED=${SEED:-42}     # --online_series_seed
GEN=${GEN:-}         # --total_generation (empty => run to max_generation)
NP=${NP:-16}         # MPI ranks

# WINDOW STEP.  Pinned to L, never exposed as a knob.  In non-pooled mode the
# training pool is {episode i : i < current_index} and the validation window IS
# current_index, so any step < L makes the last training window overlap the
# validation window by L-step rows -- straight leakage, which ONE-NAS only warns
# about (process_arguments.cxx:333).  It also sets the adaptation cadence: one
# generation advances the clock by `step` rows of VALID series time, so L=144 is
# a 12 h cadence, comfortably inside the 3-7 day crossover measured for ridge.
STEP=$L

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
  --bp_iterations "$BP" --num_mutations 1
  --possible_node_types simple UGRNN MGU GRU delta LSTM
  --rounds_per_generation 1
  # repopulation_frequency 0: with a non-zero value do_repopulation indexes
  # rank[i] for i < islands_to_exterminate while rank_islands() may return a
  # shorter vector -> out-of-bounds read -> SIGSEGV (killed v1 jobs 20006592/3).
  --repopulation_frequency 0
  --selection_metric mse
  # One-sided guard: rejects genomes whose validation predictions have >3x the
  # target sd.  It does NOT catch COLLAPSE (sd -> 0), which is what v1 produced.
  # score_v2.py checks the low side explicitly.
  --max_pred_sd_ratio 3.0
  --online_series_seed "$SEED"
  --normalize none --compare_with_naive --control_size_method none
  --std_message_level ERROR --file_message_level ERROR
  --output_directory "$OUT"
)
[ -n "$GEN" ] && ARGS+=(--total_generation "$GEN")

BIN="$HOME/ONENAS/build/mpi/onenas_mpi"

# --- the manifest, written BEFORE the run -------------------------------------
printf '%s\0' "${ARGS[@]}" > "$OUT/.argv"
DATA="$DATA" OUT="$OUT" BIN="$BIN" L="$L" T="$T" V="$V" H="$H" STEP="$STEP" \
ISL="$ISL" POP="$POP" ELITE="$ELITE" BP="$BP" SEED="$SEED" NP="$NP" \
"$PY" - <<'PY'
import hashlib, json, os, subprocess, time
d, out = os.environ["DATA"], os.environ["OUT"]
prep = json.load(open(os.path.join(d, "prep_meta.json")))
argv = open(os.path.join(out, ".argv"), "rb").read().split(b"\0")[:-1]
argv = [a.decode() for a in argv]

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
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    # ---- everything the scorer needs to align, in one place -------------------
    "params": {
        "L": i(os.environ["L"]), "T": i(os.environ["T"]), "V": i(os.environ["V"]),
        "OFFSET": i(os.environ["H"]), "window_step": i(os.environ["STEP"]),
        "ISL": i(os.environ["ISL"]), "POP": i(os.environ["POP"]),
        "ELITE": i(os.environ["ELITE"]), "BP": i(os.environ["BP"]),
        "seed": i(os.environ["SEED"]), "mpi_ranks": i(os.environ["NP"]),
        "pooled_panel": False, "sampling": "PER",
    },
    # ---- alignment contract ---------------------------------------------------
    "alignment": {
        "test_episode_of_generation": "g + T + V",
        "episode_to_window": "windows.csv row with window_id == episode",
        "file_row_to_timestep": "data row i (0-based) of generation_<g>_global_best.csv is "
                                "episode timestep j = i+1 (the writer starts at j=1)",
        "timestep_to_target_row": "target seg_row = tgt_start_seg_row + i + 1",
        "timestep_to_issue_row": "forecast-issue seg_row = target seg_row - OFFSET",
        "verification": "expected_N2O[i] must equal index.csv n2o_norm at the target "
                        "(seg_id, seg_row) to 1e-4; score_v2.py refuses the generation otherwise",
        "denormalise": "raw = pred * stats.N2O.scale + stats.N2O.center",
    },
    # ---- data provenance ------------------------------------------------------
    "data": {
        "dir": d,
        "prep_meta_sha256": sha(os.path.join(d, "prep_meta.json")),
        "index_csv_sha256": sha(os.path.join(d, "index.csv")),
        "windows_csv_sha256": sha(os.path.join(d, "windows.csv")),
        "mask": prep["mask"],
        "span_start": prep["segments"][0]["t_start"],
        "span_end": prep["segments"][-1]["t_end"],
        "burn_cut_time": prep["burn_cut_time"],
        "first_scored_window_time": prep["first_scored_window_time"],
        "n_segments": prep["n_segments_used"],
        "n_windows": prep["n_windows"],
        "max_generation": prep["max_generation"],
        "n2o_center": prep["stats"]["N2O"]["center"],
        "n2o_scale": prep["stats"]["N2O"]["scale"],
        # ordered segment map: position in this list == the order the files were
        # passed to --training_filenames, which is what fixes episode numbering
        "segment_map": [{"seg_id": s["seg_id"], "file": s["file"],
                         "n_rows": s["n_rows"], "n_windows": s["n_windows"],
                         "t_start": s["t_start"], "t_end": s["t_end"]}
                        for s in prep["segments"]],
    },
    # ---- exactly what was executed -------------------------------------------
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
      f"({man['data']['n_segments']} segments, {man['data']['n_windows']} windows, "
      f"max_generation {man['data']['max_generation']})")
PY
rm -f "$OUT/.argv"

echo "=== launching: L=$L T=$T V=$V H=$H step=$STEP ISL=$ISL POP=$POP ELITE=$ELITE BP=$BP seed=$SEED"
time srun --mpi=pmi2 -n "$NP" "$BIN" "${ARGS[@]}"
echo "=== prediction files: $(ls "$OUT"/generation_*_global_best.csv 2>/dev/null | wc -l)"
