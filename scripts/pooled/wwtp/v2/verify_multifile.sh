#!/bin/bash -l
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:10:00
#SBATCH -J vmf
#SBATCH -o /anvil/scratch/x-jchang5/wwtp/v2/verify_multifile.out
#SBATCH -e /anvil/scratch/x-jchang5/wwtp/v2/verify_multifile.err
#
# TASK 2: is the "one file per contiguous valid segment" design actually safe?
#
# The entire v2 design rests on ONE claim: a training/validation/test window can
# never span the boundary between two --training_filenames.  Reading says yes
# (process_arguments.cxx slice_input_data loops per series and restarts at row 0),
# but the design is too load-bearing to accept on a reading.  This runs the real
# binary on a case where a boundary-spanning window would be unmistakable.
#
# SETUP.  Two files, 30 rows each, in disjoint value bands:
#     fileA row i : N2O = +0.500 + i/1000   (all values POSITIVE, 0.500..0.529)
#     fileB row i : N2O = -0.500 - i/1000   (all values NEGATIVE, -0.500..-0.529)
# so the SIGN of a value says which file it came from and the third decimal says
# which row.  L=8, H(--time_offset)=2.
#
# TWO FALSIFIABLE PREDICTIONS if slicing is per-file:
#   (1) COUNT.  Each file exports 30-2 = 28 input rows -> floor(28/8) = 3 windows,
#       6 total.  If the files were concatenated into one 60-row series it would
#       export 58 rows -> floor(58/8) = 7 windows.  6 != 7, so the count alone
#       discriminates.
#   (2) CONTENT.  Every window must be sign-pure.  A window that spans the
#       boundary would contain both positive and negative N2O values.  Windows are
#       dumped by --write_sliced_files, so this is checked directly, not inferred.
set -euo pipefail
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

D=/anvil/scratch/x-jchang5/wwtp/v2/mf
rm -rf "$D"; mkdir -p "$D/run"

/apps/anvil/external/apps/conda/2024.02/bin/python3 - "$D" <<'PY'
import sys
d = sys.argv[1]
for tag, sign in (("A", 1.0), ("B", -1.0)):
    with open(f"{d}/file{tag}.csv", "w") as fh:
        fh.write("N2O,NH4\n")
        for i in range(30):
            fh.write(f"{sign*(0.500+i/1000.0):.6f},{sign*(0.100+i/1000.0):.6f}\n")
print("wrote 2 x 30-row synthetic files")
PY

srun --mpi=pmi2 -n 2 "$HOME/ONENAS/build/mpi/onenas_mpi" \
  --training_filenames "$D/fileA.csv" "$D/fileB.csv" \
  --input_parameter_names N2O NH4 \
  --output_parameter_names N2O \
  --time_offset 2 \
  --time_series_length 8 \
  --window_step 8 \
  --num_training_sets 2 --num_validation_sets 1 \
  --get_train_data_by PER \
  --speciation_method onenas \
  --number_islands 2 --generated_population_size 2 --elite_population_size 2 \
  --bp_iterations 1 --num_mutations 1 \
  --possible_node_types simple \
  --total_generation 1 --rounds_per_generation 1 --repopulation_frequency 0 \
  --selection_metric mse \
  --normalize none --control_size_method none \
  --write_sliced_files \
  --std_message_level INFO --file_message_level ERROR \
  --output_directory "$D/run" 2>&1 | grep -Ei "number of sets|inputs shape|Before slicing|After slicing|episode|training filenames|resizing" | head -40

echo "=============== VERDICT ==============="
/apps/anvil/external/apps/conda/2024.02/bin/python3 - "$D" <<'PY'
import glob, os, re, sys
d = sys.argv[1]
fs = sorted(glob.glob(f"{d}/run/sliced_data/generation_*.csv"),
            key=lambda p: int(re.search(r"generation_(\d+)", p).group(1)))
print(f"windows emitted by ONE-NAS: {len(fs)}")
print("  per-file slicing predicts 6; concatenated slicing predicts 7")
mixed = 0
for p in fs:
    rows = [l.strip().split(",") for l in open(p).read().strip().split("\n")[1:] if l.strip()]
    v = [float(r[0]) for r in rows]
    signs = {"pos" if x > 0 else "neg" for x in v}
    src = "fileA" if "pos" in signs and len(signs) == 1 else ("fileB" if len(signs) == 1 else "MIXED!!")
    if src == "MIXED!!":
        mixed += 1
    print(f"  {os.path.basename(p):<22} n={len(v):>2} first={v[0]:+.3f} last={v[-1]:+.3f} -> {src}")
print()
print(f"windows containing rows from BOTH files: {mixed}")
ok = (len(fs) == 6 and mixed == 0)
print("RESULT:", "PASS - windows never span a file boundary" if ok
      else "FAIL - the v2 segment design is NOT safe, stop and redesign")
PY
