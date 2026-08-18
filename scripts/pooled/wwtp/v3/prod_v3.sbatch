#!/bin/bash -l
#SBATCH -A cis251123
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH -J prod_v3
#SBATCH -o /anvil/scratch/x-jchang5/wwtp/v3/prod_v3_%a_%j.out
#SBATCH -e /anvil/scratch/x-jchang5/wwtp/v3/prod_v3_%a_%j.err
#SBATCH --no-requeue
#
# PRODUCTION ARM, H=72 (6 h) -- the pre-registered primary configuration of
# PROTOCOL.md.  Submit as an array over seeds:  sbatch --array=41-45 prod_v3.sbatch
#
# DO NOT EDIT the parameters below without editing PROTOCOL.md first and saying
# so.  They are pre-registered; changing them after seeing a result is exactly
# the failure mode this campaign is recovering from.
#
# CHANGES FROM prod72.sbatch:
#
#   --no-requeue.  prod72.sbatch left `mkdir -p "$OUT"` to a requeued job, which
#   is how a stale-file result gets produced by accident: the second attempt
#   writes generations 0..k into a directory that still holds the first
#   attempt's 0..n, and score_v2.py validated all of them because its
#   expected-column check is data-derived, not run-derived.  run_v3.sh now does
#   `rm -rf "$OUT"` so a requeue would be safe, and --no-requeue makes a wall
#   clock overrun visible as a failed job instead of a silently restarted one.
#   Both, not either: the scorer's mtime fence is the third layer.
#
#   OUT points at a per-seed run directory that this job OWNS.  Nothing else may
#   write there.
#
#   EXAMM_SEED is set, so mutation / crossover / island selection and every
#   genome RNG derive from it (see common/random_seed.hxx).  BIN must be a build
#   that has the flag; run_v3.sh refuses to launch otherwise, because this
#   argument parser silently ignores unknown arguments.
#
#   The scorer takes only the run directory.  The baseline directory now comes
#   from the manifest, so it can no longer be changed from the command line
#   without the scorer noticing.  It is passed anyway, as an ASSERTION: if it
#   disagrees with the manifest the scorer refuses.
#
#   The job's exit status is the scorer's verdict: 0 PASS / STRONG PASS,
#   1 FAIL (scored, criteria not met), 2 REFUSED (cannot be scored).
set -euo pipefail
S=${SLURM_ARRAY_TASK_ID:-41}
export DATA=/anvil/scratch/x-jchang5/wwtp/v2/h72_L144
export BASELINES=$DATA/baselines
export OUT=/anvil/scratch/x-jchang5/wwtp/v3/prod_v3_s$S/run
export L=144 T=80 V=5 H=72 ISL=16 POP=5 ELITE=8 BP=10 NP=16
export SEED=$S            # --online_series_seed : the PER episode sampler
export EXAMM_SEED=$S      # --examm_seed        : mutation / crossover / islands / genomes
# SRC is a SCRATCH copy of ~/ONENAS carrying only the --examm_seed patch
# (git c4d494a, forked from ~/ONENAS f7349074).  Anvil purges scratch, so the
# patch itself is kept at ~/wwtp_scripts/v3/examm_seed.patch: if this tree is
# gone, re-apply it to a fresh checkout and rebuild rather than falling back to
# the unpatched ~/ONENAS/build, which would silently drop --examm_seed.
export SRC=/anvil/scratch/x-jchang5/wwtp/v3/ONENAS_seed
export BIN=$SRC/build/mpi/onenas_mpi
if [ ! -x "$BIN" ]; then
  echo "FATAL: $BIN is missing. Rebuild from ~/wwtp_scripts/v3/examm_seed.patch." >&2
  exit 1
fi
# GEN unset => run to max_generation (610 for this prep); recorded in the manifest
# as params.total_generation, and score_v3.py refuses a run that produced fewer.

bash "$HOME/wwtp_scripts/v3/run_v3.sh"

/apps/anvil/external/apps/conda/2024.02/bin/python3 \
  "$HOME/wwtp_scripts/v3/score_v3.py" "$OUT" "$BASELINES"
