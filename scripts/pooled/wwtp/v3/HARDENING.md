# Scorer/runner hardening — all six defects fixed, each proven by reproducing the attack

Code on Anvil at `~/wwtp_scripts/v3/`; patched build at
/anvil/scratch/x-jchang5/wwtp/v3/ONENAS_seed (git c4d494a, forked from f7349074).
Evidence at /anvil/scratch/x-jchang5/wwtp/v3/{attack_v3.py,wd_demo.py,pass_demo.py}.

## 1. Stale prediction files — attack reproduced, now refused

Generations 0-4 overwritten with a collapsed constant, mtimes backdated 3600 s
(what a 12 h timeout + requeue leaves behind). Expected columns untouched, so the
data-derived assertion still passes.

  score_v2.py: aligned 10, refused 0, nMSE 1.3109 vs gate 1.9433
               -> "ONE-NAS BEATS the persistence gate", exit 0
  score_v3.py: aligned 5, refused 5 (STALE), exit 2, verdict FAIL

Runner half verified live (job 20008623): OUT pre-populated with the 27-entry
attack dir, run_v3.sh cleared it, gen-0 md5 changed, rescore -> 0 stale.

## 2. Non-finite predictions — attack reproduced, now fails loudly

70 NaNs injected across 10 generations:

  score_v2.py: covered rows 1430 -> 1360, gate 1.9433 -> 1.9393, exit 0, silent
  score_v3.py: "refused for NON-FINITE predictions: 10 (70 values)", exit 2

Row set is now coverage-derived (`covered_gen >= 0`), never `isfinite(pred)`.

## 3. Collapse detection + logging

The old `--std_message_level ERROR` suppressed **8,311 INFO and 48 WARNING lines
in THREE generations** at ISL=16 — including 9 exploding-prediction-guard messages
and every global-best update. Now INFO to `$OUT/onenas.log`, WARNING+ to stdout.

`watchdog_v3.py` (sidecar, no C++ change): on synthetic input (3 healthy then 25
collapsed) it stayed quiet through the prefix and fired on the 20th consecutive
collapsed generation, exit 3. On a real healthy run it did not fire (sd ratio
0.4771). PROTOCOL §8 is now executable — sd ratio outside [0.3,3.0], >1% below the
physical floor, or nMSE >= gate all produce verdict FAIL and a non-zero exit. All
three exit codes exercised (0 STRONG_PASS, 1 FAIL, 2 REFUSED).

## 4. Manifest closes the score

Schema /3 adds baseline_dir, per-baseline SHA-256, all 64 segment-CSV SHA-256,
total_generation, examm_seed, and a TRI-STATE git check (clean/dirty/unknown —
v2's `bool(git(...))` reported False when git FAILED). Scorer reads the baseline
dir FROM the manifest; argv is an assertion only. Seven refusal paths demonstrated,
all exit 2: baseline removed / added / content-changed, argv != manifest, segment
CSV changed, v2 manifest offered, incomplete run.

## 5. Seeded RNG — works at NP=2, BROKEN AT NP=16 (production config)

`--examm_seed` plumbed via new common/random_seed.hxx/.cxx to examm.cxx:68,
onenas.cxx:76, rnn_genome.cxx:127 (also widened from int16_t, which allowed only
65,536 genome seeds). Default preserves the clock expression.

  same seed 12345, NP=2, repeat          10/10 byte-identical
  same seed, different SLURM job          10/10 identical, same md5
  seed 54321 vs 12345                     0/10 identical
  no --examm_seed (clock), twice          0/10 identical

**LIMITATION, UNFIXED:** at NP=8 only 2/10 matched; at NP=16 two runs of the same
config and seed gave 0/3. The master inserts genomes in ARRIVAL order
(mpi/onenas_mpi.cxx:345 MPI_Probe(MPI_ANY_SOURCE) -> :404 insert_genome), so
**production NP=16 runs are not bit-reproducible even with a fixed seed.** The seed
pins the RNG streams; it does not pin MPI insertion order. Fixing needs a
deterministic insertion barrier. prod_v3.sbatch still uses NP=16.

## 6. --write_elite_predictions is write-only, cost measured

Disambiguated at NP=2 (deterministic) with production ISL=16/ELITE=8: ON twice
3/3 identical, **ON vs OFF 3/3 identical** across global_best files, genomes/,
stats/ and fitness_log.csv. The NP=16 difference is the MPI ordering above, not the
flag. Cost: 342 KiB/generation -> ~213 MB/seed, 1.07 GB for 5 seeds (1.49 GB
allocated on Lustre).

## Not verified

No production run — longest was 10 generations at ISL=4, and ISL=16 runs were 3
generations. NP=16 reproducibility broken and unfixed. The watchdog's real
`scancel` path is untested against SLURM (dry-run only). **score_v3.py scores ONE
seed — the median-of-5 campaign aggregator does not exist yet.** The C++
`max_pred_sd_ratio` guard is still one-sided; the low side is covered only
externally by the watchdog. H=24 arm untested. Not run against the v3 prep (used
v2/h72_L144).
