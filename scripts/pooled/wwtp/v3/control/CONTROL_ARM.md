# Fixed-architecture control: the same binary with evolution switched off

Both reviews called this the most important omission. It is built, verified and
smoke-tested. Code: Anvil `~/wwtp_scripts/v3/control/`.

## Route: CLI, not reimplementation

PyTorch is absent from every Python on Anvil, so a torch control was never
available — but the CLI route is the preferred one anyway, because the control IS
the ONE-NAS training loop and no second implementation can drift from it.

Switches, each verified against source:
* `--num_mutations 0` — `EXAMM::mutate` breaks before any structural operator
  (examm.cxx:290). Topology untouched.
* `--mutation_rate 1.0 --intra_island_co_rate 0.0 --inter_island_co_rate 0.0` —
  the mutation branch is taken with probability 1
  (onenas_island_speciation_strategy.cxx:341), so crossover is unreachable.
* `--genome_bin <fixed.bin>` supplies the hand-picked topology (ONE-NAS's default
  seed is create_ff = 15 nodes / 14 edges, so a fixed control must supply its own).
* The launcher REFUSES ISL=1, because process_arguments.cxx:175 silently overrides
  intra_island_co_rate to 0.30 at one island, re-enabling crossover.

**Empirically frozen:** fitness_log.csv reports 19 nodes / 60 edges / 0 recurrent
edges, constant and identical across all 60 generations.

## The one unavoidable mismatch: elite_population_size 1 vs ONE-NAS's 8

`Population::insert_genome` (population.cxx:139) dedupes by structural hash, and
`RNN_Genome::equals` (rnn_genome.cxx:1517) compares nodes/edges/recurrent-edges
**and never weights**. With a single fixed architecture every genome is "equal", so
the elite population can never fill: at ELITE=8 the island is marked FILLED while
`elite_is_full()` stays false forever, and `generate_genome` recurses without
bound. Job 20008466 stalled at generation 7 and wrote a **7.4 GB log**. ELITE=1 is
the only legal value on the stock binary.

Cost of the mismatch, checked in source rather than assumed:
* **Compute: none.** All ISL*POP genomes are still generated and trained —
  measured 800 BP epochs/generation = 16x5x10, identical to the ONE-NAS arm.
* **PER new-episode priority: none.** min over 16 champions == min over 128 elites.
* **PER blended backfill: small and one-directional** — the 0.7/0.3 blend uses a
  champions-only mean, so replay is marginally flatter. Direction characterised,
  magnitude not measured.
* Bonus: `--write_elite_predictions` then emits exactly 16 groups = the ensemble.

Clean fix (not done, to avoid forking a repo two other agents are editing): a
one-line guard on the structural-hash block behind `--allow_duplicate_structures`.
`run_control.sh` refuses ELITE>1 unless `ALLOW_DUP_STRUCTURES=1`.

## Architecture

`lstm4` — one FC hidden layer of 4 LSTM cells, 14 inputs -> 1 output, **19 nodes /
60 edges / 119 weights**. ONE-NAS's evolved global bests reach 10-19 nodes / 21-58
edges, so this sits at the top of the node range and 3% above the edge range.
**No tuning touched post-burn-in data** — chosen from published size statistics and
node-type conventions before any control run existed. Eight alternates pre-built
for a sensitivity sweep (gru4, ugrnn4, mgu4, delta4, simple4, mixed4, lstm3, lstm2).

## Smoke test (60 of 610 generations — plumbing, not evidence)

| arm | nMSE | gate | pred/target sd | below floor | distinct |
|---|---|---|---|---|---|
| single global best | 1.2309 | 1.5118 | 0.858 | 18.48% | 8509/8580 |
| 16-champion **mean** | **1.0803** | 1.5118 | 0.684 | 0.17% | 8505/8580 |
| 16-champion rank-mean | 1.6086 | 1.5118 | 0.947 | 8.05% | 8044/8580 |

No constant-collapse: v1 gave sd ratio 0.018 with a near-constant output; this
gives 0.68-0.95 with 8044-8509 distinct values and zero non-finite.

## DECISION REQUIRED: the combiner

Rank-mean is a CROSS-SECTIONAL rule. Transposed to a univariate level forecast
scored by nMSE it is materially worse here — **1.61 vs 1.08** — and mean also
nearly eliminates below-floor predictions (0.17% vs 8.05%). Whichever rule the v3
ONE-NAS scorer uses, the control must use the same one, or the comparison is
between COMBINERS rather than architectures.

Note also: v2's score_v2.py scores the **single global_best**, not an ensemble. If
v3 keeps that, the control's single-network number (1.2309) is the matched one and
the ensemble is an extra arm.

## Cost: cheaper than the ONE-NAS arm

**2.767 s/generation, sd 0.009, drift -0.2%** across 60 generations — flat by
construction, since topology cannot grow. H=72 610 generations = 28 min wall,
**7.5 core-h/seed**; 5 seeds ~ 37 core-h. Plus H=24 3 seeds ~ 23 core-h.
**Arm total ~ 60 core-h** against the ONE-NAS arm's estimated 75-125. Identical
per-generation genome count and BP iterations; the wall-clock gap is per-genome
network size only.

## Not verified

No production run for either arm. **The v3 ONE-NAS ensemble rule was unreadable
when this was built** (score_v3.py did not yet exist) — the "16 champions by
rank-mean" spec came from the brief, not from code, and MUST be re-checked. Built
against v2's h72_L144; v3's prep_meta.json schema unverified. Long-run stability
over 610 generations untested (60 run). Below-floor fraction went 0.00% at 10
generations to 18.48% at 60 for the single network as the clock entered the
low-N2O span — the mean ensemble suppresses it to 0.17%, but its behaviour over
the full run is unknown. The ELITE mismatch's effect on results is characterised in
direction only. The eight alternate architectures are built but unrun.
