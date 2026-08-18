#!/bin/bash -l
# run_v3.sh -- launch ONE-NAS on the v2/v3 segmented data and write a
# machine-readable record of EVERYTHING the scorer needs, BEFORE the binary starts.
#
# Required env: DATA (a prep output dir), OUT (run dir).
# Optional:     BASELINES (default $DATA/baselines), SRC (default $HOME/ONENAS),
#               BIN (default $SRC/build/mpi/onenas_mpi), EXAMM_SEED, GEN, WATCHDOG=0.
# Everything else defaults to the pre-registered primary arm in PROTOCOL.md.
#
# WHAT CHANGED FROM run_v2.sh, AND WHY.  Five defects, each demonstrated:
#
# (1) STALE PREDICTION FILES WERE SILENTLY SCORED.  v2 did `mkdir -p "$OUT"` and
#     never cleared it.  The scorer's expected-column check is DATA-derived, so a
#     generation file left behind by ANY earlier run at the same (DATA,L,T,V,H)
#     validated perfectly.  A reviewer overwrote generations 0-4 with a collapsed
#     constant and got "aligned 10, refused 0, worst expected-column error 0" and
#     a false "ONE-NAS BEATS the persistence gate" (nMSE 1.5878 vs gate 1.9433).
#     A 12 h timeout plus a requeue produces this by accident.
#     FIX: `rm -rf "$OUT"` below, plus an mtime fence in score_v3.py -- because
#     the runner cannot protect a directory somebody else pre-populated.
#
# (3) THE LOGS WERE MUTED.  v2 line 74 set `--std_message_level ERROR`, and
#     log.hxx:120-127 orders FATAL < ERROR < WARNING < INFO, so ERROR suppressed
#     WARNING and INFO: "ALL elite genomes failed the exploding-prediction guard"
#     and every global-best update were invisible for the whole run.
#     FIX: INFO into $OUT/onenas.log, WARNING and above echoed to job stdout,
#     plus a sidecar collapse watchdog (watchdog_v3.py) that can cancel the job.
#
# (4) THE MANIFEST DID NOT CLOSE THE SCORE.  The baseline directory was an argv
#     of the scorer and not in the manifest at all, preds_*.csv were unhashed,
#     the 64 segment CSVs (the actual model inputs) were unhashed, and
#     total_generation was unrecorded.  Since the scorer intersects the row set
#     with every loaded baseline, adding/removing/swapping one baseline file
#     silently moved the primary row set and therefore the gate.  Also
#     `onenas_git_dirty = bool(git(...))` reported False when git FAILED, not
#     only when the tree was clean.
#     FIX: all of it is recorded and the failure/clean distinction is explicit.
#
# (5) THE SEARCH RNG WAS WALL-CLOCK SEEDED.  `--online_series_seed` seeds only
#     the PER episode sampler (online_series.cxx:118-121); mutation, crossover
#     and island selection came from the clock (onenas/examm.cxx:68,
#     onenas/onenas.cxx:76) and the genome RNG was clock-seeded AND truncated to
#     int16_t (rnn/rnn_genome.cxx:127).  The manifest's `seed` was therefore
#     misleading provenance.
#     FIX: `--examm_seed` (see common/random_seed.hxx), defaulted to $SEED here
#     and recorded.  The binary is checked for the flag: this argument parser
#     ignores unknown arguments, so an unpatched binary would silently drop it.
#
# (6) --write_elite_predictions is now in the production argv.  It is called
#     AFTER evaluate_elite_population(validation) and select_elite_population()
#     (onenas_island_speciation_strategy.cxx:822), so it dumps the
#     POST-selection elites and cannot influence selection.
set -euo pipefail
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

PY=/apps/anvil/external/apps/conda/2024.02/bin/python3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=${DATA:?set DATA to a prep output directory}
OUT=${OUT:?set OUT to the run directory}
BASELINES=${BASELINES:-$DATA/baselines}
SRC=${SRC:-$HOME/ONENAS}
BIN=${BIN:-$SRC/build/mpi/onenas_mpi}

# --- parameters (defaults = pre-registered primary arm) -----------------------
L=${L:-144}          # --time_series_length ; ALSO the window step (see below)
T=${T:-80}           # --num_training_sets  : burn-in windows AND per-generation batch
V=${V:-5}            # --num_validation_sets
H=${H:-72}           # --time_offset        : the forecast horizon, in 5-min bins
ISL=${ISL:-16}       # --number_islands
POP=${POP:-5}        # --generated_population_size
ELITE=${ELITE:-8}    # --elite_population_size
BP=${BP:-10}         # --bp_iterations
SEED=${SEED:-42}     # --online_series_seed  (episode sampler ONLY)
EXAMM_SEED=${EXAMM_SEED-$SEED}   # --examm_seed (search RNG); set to "" for clock
GEN=${GEN:-}         # --total_generation (empty => run to max_generation)
NP=${NP:-16}         # MPI ranks
WATCHDOG=${WATCHDOG:-1}
WD_INTERVAL=${WD_INTERVAL:-60}
WD_THRESHOLD=${WD_THRESHOLD:-0.15}
WD_CONSECUTIVE=${WD_CONSECUTIVE:-20}

# WINDOW STEP.  Pinned to L, never exposed as a knob.  In non-pooled mode the
# training pool is {episode i : i < current_index} and the validation window IS
# current_index, so any step < L makes the last training window overlap the
# validation window by L-step rows -- straight leakage, which ONE-NAS only warns
# about (process_arguments.cxx:333).
STEP=$L

# --- (1) the output directory belongs to THIS run and nothing else ------------
# rm -rf, not mkdir -p.  A requeued or restarted job must not inherit an earlier
# job's generation_*_global_best.csv: they validate against the data, not against
# this run, so the scorer cannot tell them apart on content alone.
if [ -e "$OUT" ]; then
  echo "=== clearing existing output directory $OUT ($(ls "$OUT" 2>/dev/null | wc -l) entries)"
fi
rm -rf "$OUT"
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
  # target sd.  It does NOT catch COLLAPSE (sd -> 0) and it INVERTS on the 47/610
  # generations whose validation-target sd is < 0.010.  Kept because it is
  # pre-registered; watchdog_v3.py covers the low side in flight and
  # score_v3.py's verdict covers it after the fact.
  --max_pred_sd_ratio 3.0
  --online_series_seed "$SEED"
  --normalize none --compare_with_naive --control_size_method none
  # (3) INFO to $OUT/onenas.log via the tee below; WARNING+ echoed to job stdout.
  --std_message_level INFO --file_message_level ERROR
  --output_directory "$OUT"
)
if [ -n "$GEN" ]; then ARGS+=(--total_generation "$GEN"); fi

# --- (6) post-selection elite dump -------------------------------------------
# ISL x ELITE = 16 x 8 = 128 candidates per generation, written at
# onenas_island_speciation_strategy.cxx:822, AFTER
# evaluate_elite_population(validation) and select_elite_population(), so it
# cannot influence what is selected.  ELITE_PREDS=0 exists only so the smoke
# test can prove that by differencing the two runs; production leaves it on.
ELITE_PREDS=${ELITE_PREDS:-1}
if [ "$ELITE_PREDS" = "1" ]; then ARGS+=(--write_elite_predictions); fi

# --- (5) the search RNG -------------------------------------------------------
# This argument parser silently ignores arguments it does not know, so an
# unpatched binary would drop --examm_seed and produce an unreproducible run
# while the manifest claimed otherwise.  Check before launching.
if [ -n "$EXAMM_SEED" ]; then
  # grep -a directly on the binary: `strings ... | grep -q` would be killed by
  # SIGPIPE and, under `set -o pipefail`, report a false negative.
  if ! grep -qa -- "--examm_seed" "$BIN"; then
    echo "FATAL: $BIN does not support --examm_seed (the flag would be silently ignored" >&2
    echo "       and the run would not be reproducible). Build the patched source or" >&2
    echo "       set EXAMM_SEED= explicitly to accept a clock-seeded run." >&2
    exit 1
  fi
  ARGS+=(--examm_seed "$EXAMM_SEED")
else
  echo "WARNING: EXAMM_SEED is empty -- mutation/crossover/island selection and every"
  echo "         genome RNG will be seeded from the wall clock. This run will NOT be"
  echo "         reproducible and the manifest will say so."
fi

# --- the manifest, written BEFORE the run -------------------------------------
printf '%s\0' "${ARGS[@]}" > "$OUT/.argv"
DATA="$DATA" OUT="$OUT" BIN="$BIN" SRC="$SRC" BASELINES="$BASELINES" \
L="$L" T="$T" V="$V" H="$H" STEP="$STEP" \
ISL="$ISL" POP="$POP" ELITE="$ELITE" BP="$BP" SEED="$SEED" NP="$NP" \
EXAMM_SEED="$EXAMM_SEED" GEN="$GEN" ELITE_PREDS="$ELITE_PREDS" \
"$PY" - <<'PY'
import glob, hashlib, json, os, subprocess, time
d, out = os.environ["DATA"], os.environ["OUT"]
basedir = os.environ["BASELINES"]
prep = json.load(open(os.path.join(d, "prep_meta.json")))
argv = [a.decode() for a in open(os.path.join(out, ".argv"), "rb").read().split(b"\0")[:-1]]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# (4) git failure and a clean tree are DIFFERENT facts.  v2's
# `onenas_git_dirty = bool(git(...))` reported False for both.
def git(*a):
    try:
        r = subprocess.run(["git", "-C", os.environ["SRC"]] + list(a),
                           capture_output=True, timeout=60)
    except Exception as e:
        return None, f"git could not be run: {e!r}"
    if r.returncode != 0:
        return None, (f"git exited {r.returncode}: "
                      f"{r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout.decode("utf-8", "replace").strip(), None

commit, commit_err = git("rev-parse", "HEAD")
porcelain, porcelain_err = git("status", "--porcelain")
if porcelain_err is not None:
    git_state, git_dirty = "unknown", None
else:
    git_dirty = bool(porcelain)
    git_state = "dirty" if git_dirty else "clean"

i = int
gen_env = os.environ.get("GEN", "")
total_generation = i(gen_env) if gen_env else i(prep["max_generation"])
examm_seed_env = os.environ.get("EXAMM_SEED", "")
examm_seed = i(examm_seed_env) if examm_seed_env != "" else None

# (4) the baseline SET decides the scored row set (the scorer intersects on every
# baseline it loads), so it must be pinned by name AND content.
baseline_files = {}
for bf in sorted(glob.glob(os.path.join(basedir, "preds_*.csv"))):
    baseline_files[os.path.basename(bf)[6:-4]] = sha(bf)
if not baseline_files:
    raise SystemExit(f"FATAL: no preds_*.csv in {basedir}; refusing to launch a run whose "
                     f"scored row set cannot be pinned")

man = {
    "schema": "onenas_wwtp_run_manifest/3",
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "written_at_epoch": time.time(),
    # ---- everything the scorer needs to align, in one place -------------------
    "params": {
        "L": i(os.environ["L"]), "T": i(os.environ["T"]), "V": i(os.environ["V"]),
        "OFFSET": i(os.environ["H"]), "window_step": i(os.environ["STEP"]),
        "ISL": i(os.environ["ISL"]), "POP": i(os.environ["POP"]),
        "ELITE": i(os.environ["ELITE"]), "BP": i(os.environ["BP"]),
        "seed": i(os.environ["SEED"]),
        # (5) the search RNG, distinct from the episode sampler above.  null means
        # the run took its mutation/crossover/island/genome seeds from the clock
        # and cannot be reproduced.
        "examm_seed": examm_seed,
        "total_generation": total_generation,
        "mpi_ranks": i(os.environ["NP"]),
        "pooled_panel": False, "sampling": "PER",
        "write_elite_predictions": os.environ.get("ELITE_PREDS", "1") == "1",
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
                        "(seg_id, seg_row) to 1e-4; score_v3.py refuses the generation otherwise",
        "freshness": "a generation file whose mtime predates written_at is REFUSED as stale",
        "denormalise": "raw = pred * stats.N2O.scale + stats.N2O.center",
    },
    # ---- data provenance ------------------------------------------------------
    "data": {
        "dir": d,
        "baseline_dir": basedir,
        "baseline_files": baseline_files,
        "prep_meta_sha256": sha(os.path.join(d, "prep_meta.json")),
        "index_csv_sha256": sha(os.path.join(d, "index.csv")),
        "windows_csv_sha256": sha(os.path.join(d, "windows.csv")),
        "filelist_sha256": sha(os.path.join(d, "filelist.txt")),
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
        # passed to --training_filenames, which is what fixes episode numbering.
        # (4) sha256 added: these CSVs are the model's actual inputs and were the
        # only unhashed part of the data path.
        "segment_map": [{"seg_id": s["seg_id"], "file": s["file"],
                         "sha256": sha(s["file"] if os.path.isabs(s["file"])
                                       else os.path.join(d, s["file"])),
                         "n_rows": s["n_rows"], "n_windows": s["n_windows"],
                         "t_start": s["t_start"], "t_end": s["t_end"]}
                        for s in prep["segments"]],
    },
    # ---- exactly what was executed -------------------------------------------
    "binary": os.environ["BIN"],
    "binary_sha256": sha(os.environ["BIN"]),
    "source_tree": os.environ["SRC"],
    "onenas_git_commit": commit,
    "onenas_git_commit_error": commit_err,
    "onenas_git_dirty": git_dirty,        # True / False / null
    "onenas_git_state": git_state,        # "dirty" / "clean" / "unknown"
    "onenas_git_status_error": porcelain_err,
    "onenas_git_porcelain": None if porcelain is None else porcelain[:8000],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "argv": argv,
}
with open(os.path.join(out, "run_manifest.json"), "w") as fh:
    json.dump(man, fh, indent=2)
print(f"wrote {out}/run_manifest.json  "
      f"({man['data']['n_segments']} segments, {man['data']['n_windows']} windows, "
      f"total_generation {total_generation}, {len(baseline_files)} baselines, "
      f"examm_seed {examm_seed}, git {git_state})")
PY
rm -f "$OUT/.argv"

# --- (3b) sidecar collapse watchdog ------------------------------------------
WD_PID=""
if [ "$WATCHDOG" = "1" ]; then
  "$PY" "$HERE/watchdog_v3.py" --run "$OUT" --interval "$WD_INTERVAL" \
      --threshold "$WD_THRESHOLD" --consecutive "$WD_CONSECUTIVE" \
      --job_id "${SLURM_JOB_ID:-}" &
  WD_PID=$!
  echo "=== watchdog_v3.py pid $WD_PID (sd ratio < $WD_THRESHOLD for $WD_CONSECUTIVE gens => scancel)"
  trap '[ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null || true' EXIT
fi

echo "=== launching: L=$L T=$T V=$V H=$H step=$STEP ISL=$ISL POP=$POP ELITE=$ELITE BP=$BP"
echo "===            online_series_seed=$SEED examm_seed=${EXAMM_SEED:-<clock>} NP=$NP"
echo "===            full INFO log -> $OUT/onenas.log ; WARNING and above echoed here"

# (3a) INFO to the log file, WARNING+ to job stdout so a 610-generation run has
# in-flight signal without a 36 MB slurm .out.
T0=$SECONDS
set +e
srun --mpi=pmi2 -n "$NP" "$BIN" "${ARGS[@]}" 2>&1 \
  | tee "$OUT/onenas.log" \
  | grep --line-buffered -E '\[(FATAL|ERROR|WARNING)|global best|elite genomes failed|exploding-prediction'
RC=${PIPESTATUS[0]}
set -e
echo "=== wall time: $((SECONDS - T0)) s"

if [ -n "$WD_PID" ]; then kill "$WD_PID" 2>/dev/null || true; fi
echo "=== srun exit $RC"
echo "=== prediction files: $(ls "$OUT"/generation_*_global_best.csv 2>/dev/null | wc -l)"
echo "=== elite files:      $(ls "$OUT"/generation_*_elites.csv 2>/dev/null | wc -l)"
echo "=== run dir size:     $(du -sh "$OUT" | cut -f1)"
exit "$RC"
