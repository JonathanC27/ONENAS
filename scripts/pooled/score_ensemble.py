#!/usr/bin/env python3
"""Score ENSEMBLES of elite genomes from a pooled ONE-NAS run.

score_stream.py scores ONE genome per generation: the global best, taken from
the global_best_predicted_<param>_s<k> columns of generation_<g>_global_best.csv.
A run started with --write_elite_predictions also dumps, every generation, the
test-window predictions of EVERY island's whole elite population to

    generation_<g>_elites.csv    island,elite_rank,stock,row,predicted

(tidy/long form, one row per island x elite x stock x timestep; elite_rank 0 is
that island's best; `row` is the 0-based data-row index inside the test window
and mirrors the row index of the wide global_best file, both starting at
timestep 1).  This script combines several of those elites into one signal and
scores it with score_stream's own metric and book code, so the numbers are
directly comparable to the single-genome ones.

Member selection (--ensemble):
  global_best       the single global-best genome; identical to score_stream
                    (the elites files are not even read, and --combine is
                    bypassed, so this mode reproduces score_stream EXACTLY)
  island_champions  (default) one member per island: the lowest elite_rank
                    present in that island that generation (rank 0 unless the
                    writer had to skip it)
  all_elites        every (island, elite_rank) pair present that generation
  topn              the --top-n best elites, ordered by (elite_rank, island):
                    all islands' rank-0 elites first in ascending island order,
                    then all rank-1 elites, and so on.  The elites file carries
                    NO fitness value, so elites of equal rank in different
                    islands cannot be compared on merit; ascending island index
                    is a deterministic, arbitrary tie-break, which makes `topn`
                    a round-robin over islands rather than a true global top-N.
  diverse           TARGET-FREE greedy max-diversity selection of --top-n
                    members from the elites with elite_rank <= --diverse-gate
                    (the within-island validation-MSE rank is the merit gate;
                    the file carries no MSE value).  Each candidate's signature
                    is its centred cross-sectional prediction ranks over the
                    whole test window, unit-normalised; similarity is the dot
                    product (pooled per-row rank correlation).  Selection seeds
                    with the rank-0 elite least similar to the field on average
                    and greedily adds the candidate whose MAXIMUM similarity to
                    the chosen set is smallest.  The realised return is never
                    consulted, so unlike an IC-greedy pick this cannot select
                    on label noise; it tests the ambiguity-decomposition
                    hypothesis (ensemble error = member error - disagreement)
                    directly.

  Membership is re-derived every generation, because the elite populations are
  re-selected every generation: (island, elite_rank) is a SLOT, not a fixed
  genome.  A slot is what the per-member diagnostics below follow over time.

Combination (--combine), applied CROSS-SECTIONALLY per timestep:
  rank_mean    (default) each member's cross-section is replaced by its average
               ranks (ties share the mean rank), centred to 0 by subtracting
               (n_stocks-1)/2, and the centred ranks are averaged over members.
               Scale-free: a member whose outputs saturate at +-1e6 gets exactly
               the same vote as one emitting 1e-4, and the result is invariant
               to ANY strictly monotone per-member rescaling.
  zscore_mean  per-member cross-sectional z-score ((x-mean)/sd, sample sd),
               then averaged.  Invariant to positive affine rescaling only.  A
               member with zero cross-sectional spread contributes a flat zero
               cross-section (it still counts in the denominator).
  mean         raw average.  Included only as a baseline: it lets one saturated
               member dictate the whole ordering, which is exactly the failure
               mode the other two rules exist to avoid.

  Members whose cross-section for a timestep is missing or non-finite are
  dropped for that timestep only, and the drop count is reported.

  NOTE rank_mean and zscore_mean produce mean-zero scores, so --book algo1's
  "all top-k > 0 and all bottom-k < 0" trigger fires every single day and its
  turnover is meaningless.  Use --book sleeves (rank-driven, scale invariant)
  when combining; the script warns when you do not.

Row mapping and stitching are score_stream's, imported unchanged: generation g
maps to test window tw = g + num_training_windows + validation_sets, file row i
maps to panel row tw*step + offset + i (offset verified empirically at startup,
normally +2), and only the newest `step` file rows of each generation are kept.
The elites file's `row` field is that same file row index i.

Diagnostics (the whole point of ensembling):
  1. Member dispersion -- per timestep, the MEAN PAIRWISE Spearman correlation
     between the selected members' cross-sections, reported as a distribution
     (mean, p10, p50, p90) over the scored span.  Computed exactly, in O(m*n)
     rather than O(m^2*n), from the identity
         sum_{i!=j} u_i.u_j = ||sum_i u_i||^2 - m
     for the centred unit-norm rank vectors u_i.  Dispersion at 1.0 means the
     members are the same signal and ensembling cannot possibly help.
  2. Per-member IC vs ensemble IC -- each member SLOT is stitched into its own
     daily stream and scored with the same rank IC@1; the mean/sd/min/max over
     slots is printed next to the ensemble's, together with the day-pooled mean
     member IC (mean over days of the mean over that day's members), which is
     the like-for-like "average member" the ensemble has to beat.
  3. Row-alignment cross-check -- the fraction of scored days on which SOME
     elite in the file reproduces the wide file's global_best_predicted column
     exactly (to 1e-4 relative).  The global best is normally re-selected from
     the current elites every generation, so a high fraction is independent
     evidence that the elites file's rows line up with the wide file's.

Outputs (to --out-dir, default <run-dir>), all prefixed so they never clobber
score_stream's files:
  ensemble_stitched_predictions.csv  date,stock,pred,real,naive
  ensemble_book_daily.csv            daily book returns/equity, model and naive
  ensemble_rolling_rank_ic.csv       trailing --rolling-window mean rank IC
  ensemble_diagnostics.csv           per-day n_members, dispersion, ensemble
                                     rank IC, mean member rank IC
plus score_stream's per-year + overall summary table on stdout, the ensemble
spec in the header, and --emit-json.

--self-test runs the whole thing on a synthetic fixture with no cluster data:
hand-checked combination rules, a brute-force check of the fast dispersion
identity, and an end-to-end run whose --ensemble global_best output is compared
field by field against score_stream.py's own JSON on the same fixture.

Python 3 stdlib only (3.6-compatible; the cluster login nodes have 3.6.8).
"""

import argparse
import csv
import glob
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import score_stream as ss
from score_stream import (  # noqa: E402  (deliberate: keep the numbers identical)
    CAPITAL,
    GROSS_NOTIONAL,
    NAN,
    book_stats,
    daily_ics,
    finite,
    fmt,
    horizon_rank_ics,
    load_panel,
    load_stock_series,
    mean,
    mean_se_t,
    pearson,
    ranks,
    rolling_betas,
    run_book,
    spearman,
    std,
    stitch,
    verify_mapping,
)

ELITES_HEADER = "island,elite_rank,stock,row,predicted"


# --------------------------------------------------------------- combination

def combine(cross, rule):
    """Combine member cross-sections (list of [n_stocks]) into one [n_stocks].

    See the module docstring for the three rules.  `cross` must be non-empty
    and every member must have the same length.
    """
    m = len(cross)
    n = len(cross[0])
    if rule == "mean":
        return [sum(c[k] for c in cross) / m for k in range(n)]
    if rule == "rank_mean":
        mid = (n - 1) / 2.0
        acc = [0.0] * n
        for c in cross:
            r = ranks(c)
            for k in range(n):
                acc[k] += r[k] - mid
        return [a / m for a in acc]
    if rule == "zscore_mean":
        acc = [0.0] * n
        for c in cross:
            mu = sum(c) / n
            sd = std(c)
            if not (sd == sd) or sd <= 0.0:
                continue  # degenerate member: contributes a flat zero vote
            for k in range(n):
                acc[k] += (c[k] - mu) / sd
        return [a / m for a in acc]
    raise ValueError("unknown combine rule %r" % (rule,))


def mean_pairwise_spearman(cross):
    """Mean Spearman correlation over all member pairs of one timestep.

    Exact, but O(m*n) instead of O(m^2*n): rank each member, centre it and
    scale it to unit norm, then for unit vectors u_i

        sum_{i != j} u_i . u_j = || sum_i u_i ||^2 - m

    and the mean over the m(m-1) ordered pairs is that divided by m(m-1).
    Members with no cross-sectional spread (all ranks tied) have an undefined
    correlation with anything and are skipped.  Returns NaN with fewer than
    two usable members.
    """
    units = []
    for c in cross:
        r = ranks(c)
        mu = sum(r) / len(r)
        d = [x - mu for x in r]
        nrm = math.sqrt(sum(x * x for x in d))
        if nrm <= 0.0:
            continue
        units.append([x / nrm for x in d])
    m = len(units)
    if m < 2:
        return NAN
    n = len(units[0])
    tot = 0.0
    for k in range(n):
        s = 0.0
        for u in units:
            s += u[k]
        tot += s * s
    return (tot - m) / (m * (m - 1))


def percentile(v, q):
    """Linearly interpolated percentile of a list (q in [0,1])."""
    if not v:
        return NAN
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


# ------------------------------------------------------------ elites loading

def load_generation_elites(path, n_stocks, lo, hi):
    """Read generation_<g>_elites.csv, keeping file rows lo <= row < hi.

    Returns {(island, elite_rank): {row: [n_stocks] predictions}}.  Rows that
    are not fully populated (a stock missing entirely) are dropped and counted;
    NaN/inf values written by an exploding genome are kept here and dropped
    later, per timestep, by the caller.
    """
    members = {}
    with open(path) as fh:
        header = fh.readline().lstrip("#").strip()
        if header != ELITES_HEADER:
            sys.exit(
                "%s: unexpected header %r, expected %r" % (path, header, ELITES_HEADER)
            )
        for line in fh:
            if not line.strip():
                continue
            p = line.split(",")
            if len(p) != 5:
                sys.exit("%s: malformed line with %d fields: %r" % (path, len(p), line))
            row = int(p[3])
            if row < lo or row >= hi:
                continue
            k = int(p[2])
            if k < 0 or k >= n_stocks:
                sys.exit(
                    "%s: stock index %d outside 0..%d; the run's panel does not "
                    "match --sidecar-dir" % (path, k, n_stocks - 1)
                )
            key = (int(p[0]), int(p[1]))
            rows = members.get(key)
            if rows is None:
                rows = members[key] = {}
            vec = rows.get(row)
            if vec is None:
                vec = rows[row] = [None] * n_stocks
            vec[k] = float(p[4])
    incomplete = 0
    for key in list(members):
        rows = members[key]
        for row in list(rows):
            if any(x is None for x in rows[row]):
                del rows[row]
                incomplete += 1
        if not rows:
            del members[key]
    if not members:
        sys.exit("%s: no usable elite predictions for file rows %d..%d"
                 % (path, lo, hi - 1))
    return members, incomplete


def select_diverse(members, n, gate):
    """Greedy max-diversity selection of n member slots, target-free.

    Candidates are the slots with elite_rank <= gate.  A candidate's signature
    is the concatenation, over the window rows every candidate has finite
    predictions for, of its centred cross-sectional prediction ranks,
    normalised to unit norm; similarity between two candidates is the dot
    product of their signatures.  The seed is the rank-0 elite with the lowest
    mean similarity to the whole candidate field; each further pick minimises
    the MAXIMUM similarity to the already-chosen set (deterministic ties by
    slot order).  Realised returns are never consulted.
    """
    cands = sorted(s for s in members if s[1] <= gate)
    common = None
    for s in cands:
        rs = set(r for r, v in members[s].items()
                 if all(math.isfinite(x) for x in v))
        common = rs if common is None else (common & rs)
    if not common:
        # every candidate row is broken somewhere; fall back to champions
        return select_members(sorted(members), "island_champions", n)
    rows = sorted(common)
    sig = {}
    for s in cands:
        vec = []
        for r in rows:
            rk = ranks(members[s][r])
            mu = sum(rk) / len(rk)
            vec.extend(x - mu for x in rk)
        nrm = math.sqrt(sum(x * x for x in vec))
        if nrm > 0.0:
            sig[s] = [x / nrm for x in vec]
    cands = [s for s in cands if s in sig]
    if len(cands) <= n:
        return cands
    gram = {}
    for i, a in enumerate(cands):
        va = sig[a]
        for b in cands[i + 1:]:
            vb = sig[b]
            d = 0.0
            for x, y in zip(va, vb):
                d += x * y
            gram[(a, b)] = gram[(b, a)] = d
    seeds = [s for s in cands if s[1] == 0] or cands
    seed = min(seeds, key=lambda s: (mean([gram[(s, o)] for o in cands
                                           if o != s]), s))
    chosen = [seed]
    rest = [s for s in cands if s != seed]
    while len(chosen) < n and rest:
        nxt = min(rest, key=lambda s: (max(gram[(s, c)] for c in chosen), s))
        chosen.append(nxt)
        rest.remove(nxt)
    return sorted(chosen)


def select_members(available, mode, top_n):
    """Pick member slots from the (island, elite_rank) pairs of one generation."""
    if mode == "island_champions":
        best = {}
        for isl, rk in available:
            if isl not in best or rk < best[isl]:
                best[isl] = rk
        return sorted((isl, best[isl]) for isl in best)
    if mode == "all_elites":
        return sorted(available)
    if mode == "topn":
        # (elite_rank, island): every island's rank-0 elite first, in ascending
        # island order, then every rank-1 elite, ...  The file carries no
        # fitness, so island index is the (arbitrary) deterministic tie-break.
        return sorted(available, key=lambda s: (s[1], s[0]))[:top_n]
    raise ValueError("unknown ensemble mode %r" % (mode,))


def find_elite_generations(run_dir):
    files = glob.glob(os.path.join(run_dir, "generation_*_elites.csv"))
    return set(
        int(re.search(r"generation_(\d+)_elites", os.path.basename(p)).group(1))
        for p in files
    )


def _vectors_match(a, b, rel=1e-4):
    for x, y in zip(a, b):
        if not (x == x and y == y):
            return False
        if abs(x - y) > rel * max(1.0, abs(x), abs(y)):
            return False
    return True


# ------------------------------------------------------------ ensemble build

def build_ensemble(run_dir, gens, stream, args, n_stocks, offset):
    """Replace the global-best signal of `stream` with the combined ensemble.

    Returns (ens_stream, info) where ens_stream has score_stream's tuple shape
    (panel_row, date, signal, real, naive) and info carries the diagnostics.

    The elites file's `row` field is the same file row index the wide file's
    rows are numbered by, so day di of the stitched stream (generation index
    gi, file row i) takes its members from row i of generation gens[gi]'s
    elites file.  That identification is asserted against score_stream's own
    panel row for every day, so a future change to stitch()'s ordering cannot
    silently misalign the members.
    """
    step, L = args.step, args.length
    lo, hi = L - 1 - step, L - 1
    # diverse selection estimates member similarity over the WHOLE test window
    # (more rows -> better correlation estimates; target-free, so no leakage),
    # while combination and booking still use only the newest `step` rows
    sel_lo = 0 if args.ensemble == "diverse" else lo
    have = find_elite_generations(run_dir)
    missing = [g for g in gens if g not in have]
    if missing:
        sys.exit(
            "%d of %d generations have no generation_<g>_elites.csv (first "
            "missing: %d); rerun onenas_mpi with --write_elite_predictions, or "
            "score with --ensemble global_best" % (len(missing), len(gens), missing[0])
        )

    signals = []
    dispersion = []
    n_members_day = []
    member_ic = {}          # slot -> {day index: rank IC@1}
    members_per_gen = []
    dropped = 0
    incomplete = 0
    gb_matched = set()
    for gi, g in enumerate(gens):
        path = os.path.join(run_dir, "generation_%d_elites.csv" % g)
        members, inc = load_generation_elites(path, n_stocks, sel_lo, hi)
        incomplete += inc
        if args.ensemble == "diverse":
            slots = select_diverse(members, args.top_n, args.diverse_gate)
        else:
            slots = select_members(sorted(members), args.ensemble, args.top_n)
        if not slots:
            sys.exit("%s: --ensemble %s selected no members" % (path, args.ensemble))
        members_per_gen.append(len(slots))
        for i in range(lo, hi):
            di = gi * step + (i - lo)
            day = stream[di]
            want = (g + args.num_training_windows + args.validation_sets) * step \
                + offset + i
            if day[0] != want:
                sys.exit(
                    "internal error: stitched day %d is panel row %d but "
                    "generation %d file row %d maps to panel row %d"
                    % (di, day[0], g, i, want)
                )
            cross, used = [], []
            for s in slots:
                v = members[s].get(i)
                if v is None or not all(math.isfinite(x) for x in v):
                    dropped += 1
                    continue
                cross.append(v)
                used.append(s)
            if not cross:
                sys.exit(
                    "generation %d file row %d (%s): every selected member's "
                    "prediction is missing or non-finite" % (g, i, day[1])
                )
            signals.append(combine(cross, args.combine))
            dispersion.append(mean_pairwise_spearman(cross))
            n_members_day.append(len(cross))
            for s, v in zip(used, cross):
                ic = spearman(v, day[3])
                member_ic.setdefault(s, {})[di] = ic if ic is not None else NAN
            # row-alignment cross-check against the wide file's global best
            for rows in members.values():
                v = rows.get(i)
                if v is not None and _vectors_match(v, day[2]):
                    gb_matched.add(di)
                    break

    if len(signals) != len(stream):
        sys.exit("internal error: %d ensemble days for %d stitched days"
                 % (len(signals), len(stream)))
    ens = [(d[0], d[1], signals[j], d[3], d[4]) for j, d in enumerate(stream)]
    info = {
        "dispersion": dispersion,
        "n_members_day": n_members_day,
        "members_per_gen": members_per_gen,
        "member_ic": member_ic,
        "dropped": dropped,
        "incomplete": incomplete,
        "gb_matched": gb_matched,
    }
    return ens, info


# ------------------------------------------------------------------ reporting

def member_diagnostics(info, keep, ens_rank_ic):
    """Summarise dispersion and the per-member vs ensemble IC comparison.

    `keep` is the list of stitched-stream indices that survived --score-from,
    so every diagnostic covers exactly the scored span.
    """
    keep_set = set(keep)
    disp = finite([info["dispersion"][i] for i in keep])
    per_slot = []
    days_per_slot = []
    slot_names = []
    by_slot = {}
    for slot in sorted(info["member_ic"]):
        ics = finite([v for i, v in info["member_ic"][slot].items() if i in keep_set])
        if ics:
            per_slot.append(mean(ics))
            days_per_slot.append(len(ics))
            slot_names.append("island%d_elite%d" % slot)
            by_slot[slot_names[-1]] = {"rank_ic_1": mean(ics), "n_days": len(ics)}
    # day-pooled: mean over days of the mean over that day's members
    day_means = []
    by_day = {}
    for slot, d in info["member_ic"].items():
        for i, v in d.items():
            if i in keep_set and v == v:
                by_day.setdefault(i, []).append(v)
    for i in sorted(by_day):
        day_means.append(mean(by_day[i]))
    nmem = [info["n_members_day"][i] for i in keep]
    return {
        "members_per_day": {
            "mean": mean(nmem), "min": min(nmem) if nmem else NAN,
            "max": max(nmem) if nmem else NAN,
        },
        "members_per_generation": {
            "mean": mean(info["members_per_gen"]),
            "min": min(info["members_per_gen"]),
            "max": max(info["members_per_gen"]),
        },
        "dispersion": {
            "n_days": len(disp), "mean": mean(disp),
            "p10": percentile(disp, 0.10), "p50": percentile(disp, 0.50),
            "p90": percentile(disp, 0.90),
        },
        "member_rank_ic_1": {
            "n_slots": len(per_slot), "mean": mean(per_slot), "sd": std(per_slot),
            "min": min(per_slot) if per_slot else NAN,
            "max": max(per_slot) if per_slot else NAN,
            "mean_days_per_slot": mean(days_per_slot),
        },
        "member_slots": slot_names,
        "member_rank_ic_1_by_slot": by_slot,
        "day_pooled_mean_member_rank_ic_1": mean(day_means),
        "ensemble_rank_ic_1": ens_rank_ic,
        "ensemble_minus_mean_member": ens_rank_ic - mean(day_means)
        if day_means and ens_rank_ic == ens_rank_ic else NAN,
        "dropped_member_days": info["dropped"],
        "incomplete_member_rows": info["incomplete"],
        "global_best_matched_days": len(info["gb_matched"] & keep_set),
    }


def print_diagnostics(diag, n_days):
    d = diag["dispersion"]
    m = diag["member_rank_ic_1"]
    print()
    print("# members/day: mean %.2f (min %d, max %d); per generation mean %.2f "
          "(min %d, max %d)"
          % (diag["members_per_day"]["mean"], diag["members_per_day"]["min"],
             diag["members_per_day"]["max"],
             diag["members_per_generation"]["mean"],
             diag["members_per_generation"]["min"],
             diag["members_per_generation"]["max"]))
    print("# member dispersion (mean pairwise Spearman of member cross-sections, "
          "per day, n=%d):" % d["n_days"])
    print("#     mean %s   p10 %s   p50 %s   p90 %s"
          % (fmt(d["mean"]), fmt(d["p10"]), fmt(d["p50"]), fmt(d["p90"])))
    print("# per-member rank IC@1 over %d member slots (each stitched over its "
          "own %.0f days):" % (m["n_slots"], m["mean_days_per_slot"]))
    print("#     mean %s   sd %s   min %s   max %s"
          % (fmt(m["mean"]), fmt(m["sd"], "{:.4f}"), fmt(m["min"]), fmt(m["max"])))
    print("# ensemble rank IC@1 %s  vs  day-pooled mean member %s  (ensemble - "
          "member = %s)"
          % (fmt(diag["ensemble_rank_ic_1"]),
             fmt(diag["day_pooled_mean_member_rank_ic_1"]),
             fmt(diag["ensemble_minus_mean_member"])))
    if diag["dropped_member_days"] or diag["incomplete_member_rows"]:
        print("# dropped %d non-finite member-timesteps, %d incomplete member rows"
              % (diag["dropped_member_days"], diag["incomplete_member_rows"]))
    print("# row-alignment cross-check: the wide file's global_best column is "
          "reproduced by some elite on %d/%d scored days"
          % (diag["global_best_matched_days"], n_days))


# ----------------------------------------------------------------- the score

def score(args):
    """Run the whole pipeline; print the report and return the JSON payload."""
    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    dates, tickers, prc, tc, ret_raw = load_panel(args.sidecar_dir)
    n_stocks = len(tickers)

    if args.realized == "sidecar" and ret_raw is None:
        sys.exit(
            "--realized sidecar: %s has no RET_raw_<TICKER> columns; use "
            "--realized expected instead"
            % os.path.join(args.sidecar_dir, "panel_dates.csv")
        )
    use_sidecar = ret_raw is not None if args.realized == "auto" else \
        args.realized == "sidecar"
    realized = ret_raw if use_sidecar else None
    realized_label = "sidecar RET_raw" if use_sidecar else "expected_%s" % args.param
    print("# realized: %s (param %s)" % (realized_label, args.param))
    if not use_sidecar and args.param != "RET":
        print("# warning: scoring against expected_%s, which is NOT a return "
              "series; the book P&L and multi-horizon cumulative returns are "
              "not meaningful" % args.param, file=sys.stderr)

    gens = ss.find_generations(args.run_dir)
    if args.skip_verify:
        offset = 2
        print("# row-mapping verification skipped; assuming offset +2")
    else:
        offset = verify_mapping(args.run_dir, args.sidecar_dir, tickers, gens, args)

    stream = stitch(args.run_dir, gens, offset, dates, n_stocks, args,
                    realized=realized)
    print("# stitched %d days from %d generations (%d*step = %d); %s .. %s "
          "(panel rows %d..%d, contiguous)"
          % (len(stream), len(gens), len(gens), len(gens) * args.step,
             stream[0][1], stream[-1][1], stream[0][0], stream[-1][0]))

    if args.ensemble == "global_best":
        # The single global-best genome: no combination, no elites file.  This
        # path is deliberately a verbatim passthrough so the output reproduces
        # score_stream.py exactly.
        ens, info = list(stream), None
        print("# ensemble: global_best (1 member, the global-best genome); "
              "--combine %s is not applied -- this reproduces score_stream.py"
              % args.combine)
    else:
        ens, info = build_ensemble(args.run_dir, gens, stream, args, n_stocks,
                                   offset)
        spec = args.ensemble
        if args.ensemble == "topn":
            spec += " (N=%d, ordered by (elite_rank, island))" % args.top_n
        if args.ensemble == "diverse":
            spec += (" (N=%d, elite_rank <= %d, greedy min-max-similarity, "
                     "target-free)" % (args.top_n, args.diverse_gate))
        print("# ensemble: %s, combine=%s, over %d generations"
              % (spec, args.combine, len(gens)))
        if args.book == "algo1" and args.combine in ("rank_mean", "zscore_mean"):
            print("# warning: --combine %s produces a mean-zero score, so "
                  "--book algo1's sign trigger fires every day and its turnover "
                  "is not meaningful; --book sleeves is the right book for a "
                  "combined signal" % args.combine, file=sys.stderr)

    # (a) tidy CSV of the ensemble stream
    tidy_path = os.path.join(out_dir, "ensemble_stitched_predictions.csv")
    with open(tidy_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "stock", "pred", "real", "naive"])
        for _, date, pred, real, naive in ens:
            for k, t in enumerate(tickers):
                w.writerow([date, t, repr(pred[k]), repr(real[k]), repr(naive[k])])

    keep = [i for i, d in enumerate(ens) if d[1] >= args.score_from]
    days = [ens[i] for i in keep]
    if not days:
        sys.exit("no stitched days on/after --score-from %s" % args.score_from)
    print("# scoring %d days from %s to %s (--score-from %s)"
          % (len(days), days[0][1], days[-1][1], args.score_from))

    PRED, NAIVE = 2, 4
    pears = {"model": daily_ics(days, PRED, pearson),
             "naive": daily_ics(days, NAIVE, pearson)}
    hics = {
        name: {h: horizon_rank_ics(days, idx, h)
               for h in range(1, args.max_horizon + 1)}
        for name, idx in (("model", PRED), ("naive", NAIVE))
    }
    spear = {name: hics[name][1] for name in hics}

    betas = None
    if args.beta_neutral:
        if realized is not None:
            src = {r: realized[r] for r in range(len(realized))}
            src_label = "sidecar RET_raw"
        else:
            src = src_label = None
            for col in ("RET", args.param):
                src = load_stock_series(args.sidecar_dir, tickers, col)
                if src is not None:
                    src_label = "per-stock sidecar column '%s'" % col
                    break
            if src is None:
                src = {row: real for row, _, _, real, _ in ens}
                src_label = "the stitched stream itself"
        betas = rolling_betas(days, src, args.beta_window)
        n_ok = sum(1 for b in betas if b is not None)
        print("# beta-neutral: %d-row causal betas vs the equal-weight panel "
              "mean from %s; %d/%d days neutralised"
              % (args.beta_window, src_label, n_ok, len(betas)))

    def book_of(idx):
        return run_book(days, idx, prc, tc, args.top_k, args.cost_bps,
                        book=args.book, hold_days=args.hold_days, betas=betas)

    books = {"model": book_of(PRED), "naive": book_of(NAIVE)}
    h = ", H=%d" % args.hold_days if args.book == "sleeves" else ""
    print("# book: %s%s, top-k %d, beta-neutral %s"
          % (args.book, h, args.top_k, "on" if args.beta_neutral else "off"))
    print("# book costs: " + ("TC/PRC from the panel" if args.cost_bps is None
                              else "flat %g bps one-way" % args.cost_bps)
          + ", charged on netted traded notional")

    # (b) daily book returns
    book_path = os.path.join(out_dir, "ensemble_book_daily.csv")
    with open(book_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date",
                    "model_ret", "model_equity", "model_rebalanced",
                    "model_cost", "model_traded",
                    "naive_ret", "naive_equity", "naive_rebalanced",
                    "naive_cost", "naive_traded"])
        eq = {"model": CAPITAL, "naive": CAPITAL}
        for i, day in enumerate(days):
            rec = [day[1]]
            for name in ("model", "naive"):
                b = books[name]
                eq[name] += CAPITAL * b["daily_ret"][i]
                rec += ["%.8f" % b["daily_ret"][i], "%.4f" % eq[name],
                        b["rebalanced"][i], "%.6f" % b["cost"][i],
                        "%.6f" % b["traded"][i]]
            w.writerow(rec)

    # (c) rolling mean rank IC
    roll_path = os.path.join(out_dir, "ensemble_rolling_rank_ic.csv")
    W = args.rolling_window
    with open(roll_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "model_rank_ic_%dd" % W, "naive_rank_ic_%dd" % W])
        for i in range(W - 1, len(days)):
            m = mean(finite(spear["model"][i - W + 1:i + 1]))
            n = mean(finite(spear["naive"][i - W + 1:i + 1]))
            w.writerow([days[i][1], "%.6f" % m, "%.6f" % n])

    # ------------------------------------------------------------- summary
    years = sorted(set(d[1][:4] for d in days))
    periods = years + ["overall"]
    idx_of = {p: [i for i, d in enumerate(days) if p == "overall" or d[1][:4] == p]
              for p in periods}

    def slice_stats(name, p):
        ix = idx_of[p]
        pear_m, pear_se, pear_t = mean_se_t(finite([pears[name][i] for i in ix]))
        rank_m, rank_se, rank_t = mean_se_t(finite([spear[name][i] for i in ix]))
        row = {
            "n_days": len(ix),
            "pearson_ic": pear_m, "pearson_se": pear_se, "pearson_t": pear_t,
            "rank_ic_1": rank_m, "rank_ic_se": rank_se, "rank_ic_t": rank_t,
        }
        for hz in (5, 10):
            if hz <= args.max_horizon:
                row["rank_ic_%d" % hz] = mean(
                    finite([hics[name][hz][i] for i in ix]))
        net, sharpe, mdd = book_stats([books[name]["daily_ret"][i] for i in ix])
        row.update(net_pct=net, sharpe=sharpe, mdd_pct=mdd)
        row["cost_pct"] = 100.0 * sum(books[name]["cost"][i] for i in ix) / CAPITAL
        row["turnover"] = mean([books[name]["traded"][i] for i in ix]) / GROSS_NOTIONAL
        if p == "overall":
            row["n_rebalances"] = books[name]["n_rebalances"]
            row["avg_holding_days"] = books[name]["avg_holding_days"]
        return row

    summary = {p: {name: slice_stats(name, p) for name in ("model", "naive")}
               for p in periods}

    print()
    print("period    who     days  pearsonIC  rankIC@1  rankIC@5  rankIC@10"
          "      net%   sharpe     MDD% turnover    cost%")
    for p in periods:
        for name in ("model", "naive"):
            r = summary[p][name]
            print("%-9s %-6s %5d  %s   %s   %s   %s  %s %s %s %s %s"
                  % (p, name, r["n_days"], fmt(r["pearson_ic"]),
                     fmt(r["rank_ic_1"]), fmt(r.get("rank_ic_5", NAN)),
                     fmt(r.get("rank_ic_10", NAN)), fmt(r["net_pct"], "{:+8.2f}"),
                     fmt(r["sharpe"], "{:+8.2f}"), fmt(r["mdd_pct"], "{:8.2f}"),
                     fmt(r["turnover"], "{:8.4f}"), fmt(r["cost_pct"], "{:8.2f}")))
    print()
    for name in ("model", "naive"):
        r = summary["overall"][name]
        print("# %s overall: pearson IC %s (SE %s, t %s); rank IC %s "
              "(SE %s, t %s); rebalances %d, avg holding %s days"
              % (name, fmt(r["pearson_ic"]), fmt(r["pearson_se"], "{:.4f}"),
                 fmt(r["pearson_t"], "{:+.2f}"), fmt(r["rank_ic_1"]),
                 fmt(r["rank_ic_se"], "{:.4f}"), fmt(r["rank_ic_t"], "{:+.2f}"),
                 r["n_rebalances"], fmt(r["avg_holding_days"], "{:.1f}")))
    hz = {name: {h: mean(finite(hics[name][h]))
                 for h in range(1, args.max_horizon + 1)}
          for name in ("model", "naive")}
    print("# rank IC by horizon (overall): " + "  ".join(
        "h=%d:" % h + fmt(hz["model"][h]) for h in range(1, args.max_horizon + 1)))
    print("#   naive                     : " + "  ".join(
        "h=%d:" % h + fmt(hz["naive"][h]) for h in range(1, args.max_horizon + 1)))

    diag = None
    if info is None:
        print()
        print("# member dispersion and per-member IC are not applicable to "
              "--ensemble global_best (one genome, no members to disagree)")
    if info is not None:
        diag = member_diagnostics(info, keep, summary["overall"]["model"]["rank_ic_1"])
        print_diagnostics(diag, len(days))
        diag_path = os.path.join(out_dir, "ensemble_diagnostics.csv")
        with open(diag_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "n_members", "dispersion", "ensemble_rank_ic",
                        "mean_member_rank_ic"])
            for j, i in enumerate(keep):
                ics = finite([d[i] for d in info["member_ic"].values() if i in d])
                w.writerow([days[j][1], info["n_members_day"][i],
                            "%.6f" % info["dispersion"][i],
                            "%.6f" % spear["model"][j],
                            "%.6f" % mean(ics)])
        print("# wrote %s" % diag_path)
    print("# wrote %s" % tidy_path)
    print("# wrote %s" % book_path)
    print("# wrote %s" % roll_path)

    payload = {
        "meta": {
            "run_dir": args.run_dir,
            "param": args.param,
            "realized": realized_label,
            "realized_mode": args.realized,
            "generations": len(gens),
            "row_offset": offset,
            "stitched_days": len(stream),
            "scored_days": len(days),
            "first_scored_date": days[0][1],
            "last_scored_date": days[-1][1],
            "score_from": args.score_from,
            "top_k": args.top_k,
            "cost_bps": args.cost_bps,
            "book": args.book,
            "hold_days": args.hold_days if args.book == "sleeves" else None,
            "beta_neutral": args.beta_neutral,
            "beta_window": args.beta_window if args.beta_neutral else None,
            "ensemble": args.ensemble,
            "combine": args.combine if args.ensemble != "global_best" else None,
            "top_n": args.top_n if args.ensemble in ("topn", "diverse") else None,
            "diverse_gate": args.diverse_gate if args.ensemble == "diverse"
            else None,
        },
        "summary": summary,
        "horizon_rank_ic": hz,
        "diagnostics": diag,
    }
    if args.emit_json:
        def clean(o):
            if isinstance(o, float):
                return None if o != o else o
            if isinstance(o, dict):
                return dict((k, clean(v)) for k, v in o.items())
            return o
        print(json.dumps(clean(payload), indent=1))
    return payload


# ---------------------------------------------------------------- self-test

def _lcg_panel(n_rows, n_stocks, seed):
    rnd = random.Random(seed)
    ret = [[rnd.gauss(0.0, 0.02) for _ in range(n_stocks)] for _ in range(n_rows)]
    prc = [[50.0 + 10.0 * k + r * 0.01 for k in range(n_stocks)]
           for r in range(n_rows)]
    return ret, prc


def build_fixture(root, n_stocks=6, n_gens=8, step=2, length=6, ntw=3, V=1,
                  seed=7):
    """A synthetic run + sidecar panel with both wide and elites files.

    Four members per island, two islands.  Member (0,0) and (0,1) are good
    (true return plus noise) at ordinary scale, member (1,0) is good but
    saturated at a scale of 1e6, and member (1,1) is pure noise at scale 1e6.
    A raw `mean` combination is therefore dominated by the two 1e6-scale
    members, while rank_mean gives all four an equal vote.
    """
    run = os.path.join(root, "run")
    side = os.path.join(root, "sidecar")
    os.makedirs(run)
    os.makedirs(side)
    tickers = ["S%02d" % k for k in range(n_stocks)]
    n_rows = (n_gens - 1 + ntw + V) * step + 2 + (length - 1) + 5
    ret, prc = _lcg_panel(n_rows, n_stocks, seed)

    with open(os.path.join(side, "panel_dates.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "date"] + ["PRC_" + t for t in tickers]
                   + ["TC_" + t for t in tickers] + ["RET_raw_" + t for t in tickers])
        for r in range(n_rows):
            date = "20%02d-%02d-%02d" % (20 + r // 300, 1 + (r // 28) % 12, 1 + r % 28)
            w.writerow([r, date] + ["%.6f" % v for v in prc[r]]
                       + ["%.6f" % (0.02 * (k + 1)) for k in range(n_stocks)]
                       + [repr(v) for v in ret[r]])
    for k, t in enumerate(tickers):
        with open(os.path.join(side, t + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["RET", "VOL"])
            for r in range(n_rows):
                w.writerow([repr(ret[r][k]), "1.0"])

    rnd = random.Random(seed + 1)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for g in range(n_gens):
        base = (g + ntw + V) * step + 2
        wide = os.path.join(run, "generation_%d_global_best.csv" % g)
        with open(wide, "w") as fh:
            hdr = []
            for k in range(n_stocks):
                hdr += ["expected_RET_s%d" % k, "naive_RET_s%d" % k,
                        "global_best_predicted_RET_s%d" % k]
            fh.write("#" + ",".join(hdr) + "\n")
            for i in range(length - 1):
                rec = []
                for k in range(n_stocks):
                    rec += [repr(ret[base + i][k]), repr(ret[base + i - 1][k]),
                            repr(0.5 * ret[base + i][k] + rnd.gauss(0, 0.01))]
                fh.write(",".join(rec) + "\n")
        with open(os.path.join(run, "generation_%d_elites.csv" % g), "w") as fh:
            fh.write(ELITES_HEADER + "\n")
            for isl, rk in slots:
                scale = 1.0 if isl == 0 else 1e6
                noise = 0.01 if (isl, rk) != (1, 1) else 0.0
                for k in range(n_stocks):
                    for i in range(length - 1):
                        if (isl, rk) == (1, 1):
                            v = rnd.gauss(0, 1.0) * scale
                        else:
                            v = (ret[base + i][k] + rnd.gauss(0, noise)) * scale
                        fh.write("%d,%d,%d,%d,%r\n" % (isl, rk, k, i, v))
    return run, side


def _args_for(run, side, **over):
    d = dict(run_dir=run, sidecar_dir=side, step=2, length=6,
             num_training_windows=3, validation_sets=1, param="RET",
             realized="auto", score_from="2000-01-01", top_k=2, book="sleeves",
             hold_days=2, beta_neutral=False, beta_window=60, cost_bps=None,
             rolling_window=5, max_horizon=3, out_dir=None, emit_json=False,
             skip_verify=False, ensemble="island_champions", top_n=2,
             diverse_gate=1, combine="rank_mean", self_test=False)
    d.update(over)
    return argparse.Namespace(**d)


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


def run_self_test():
    ok = [0]
    fail = []

    def check(name, cond, detail=""):
        if cond:
            ok[0] += 1
            print("  PASS  %s" % name)
        else:
            fail.append(name)
            print("  FAIL  %s  %s" % (name, detail))

    print("== combination rules on a hand-checked 3-member x 5-stock fixture")
    m0 = [1.0, 2.0, 3.0, 4.0, 5.0]                  # ranks 0 1 2 3 4
    m1 = [5000.0, 4000.0, 3000.0, 2000.0, 1000.0]   # ranks 4 3 2 1 0, scale 1e3
    m2 = [1e-6, 3e-6, 2e-6, 5e-6, 4e-6]             # ranks 0 2 1 4 3, scale 1e-6
    cross = [m0, m1, m2]
    want = [-2.0 / 3, 0.0, -1.0 / 3, 2.0 / 3, 1.0 / 3]
    got = combine(cross, "rank_mean")
    check("rank_mean equals the hand-computed centred-rank average",
          all(abs(a - b) < 1e-12 for a, b in zip(got, want)),
          "got %s want %s" % (got, want))

    want_mean = [(1 + 5000 + 1e-6) / 3, (2 + 4000 + 3e-6) / 3, (3 + 3000 + 2e-6) / 3,
                 (4 + 2000 + 5e-6) / 3, (5 + 1000 + 4e-6) / 3]
    got_mean = combine(cross, "mean")
    check("mean equals the hand-computed raw average",
          all(abs(a - b) < 1e-9 for a, b in zip(got_mean, want_mean)))
    check("mean is dictated by the largest-scale member (its ordering is m1's)",
          ranks(got_mean) == ranks(m1))

    # per-member strictly monotone rescaling: linear, cubic, exponential
    resc = [[x * 1e-3 for x in m0], [x ** 3 for x in m1],
            [math.exp(x * 1e6) for x in m2]]
    check("rank_mean is invariant to per-member monotone rescaling "
          "(x1e-3, x^3, exp)",
          all(abs(a - b) < 1e-12
              for a, b in zip(combine(resc, "rank_mean"), got)))
    # shrinking m1 alone hands control of `mean` from m1 to m0+m2 and flips its
    # ordering end to end, while rank_mean does not move at all
    shrunk = [m0, [x * 1e-9 for x in m1], m2]
    check("rank_mean is invariant to shrinking one member by 1e-9",
          all(abs(a - b) < 1e-12
              for a, b in zip(combine(shrunk, "rank_mean"), got)))
    check("mean is NOT invariant to it: its ordering flips from m1's to m0's",
          ranks(got_mean) == ranks(m1)
          and ranks(combine(shrunk, "mean")) == ranks(m0)
          and ranks(m0) != ranks(m1))

    # ranks of [1,1,2,2,3] are [.5,.5,2.5,2.5,4], centred by (5-1)/2 = 2
    tied = combine([[1.0, 1.0, 2.0, 2.0, 3.0]], "rank_mean")
    check("rank_mean gives tied values the average rank",
          all(abs(a - b) < 1e-12
              for a, b in zip(tied, [-1.5, -1.5, 0.5, 0.5, 2.0])),
          "got %s" % (tied,))

    z = combine([m0, [3.0 * x + 7.0 for x in m0]], "zscore_mean")
    z1 = combine([m0, m0], "zscore_mean")
    check("zscore_mean is invariant to positive affine per-member rescaling",
          all(abs(a - b) < 1e-12 for a, b in zip(z, z1)))
    check("zscore_mean of a degenerate member is a flat zero vote",
          all(abs(a - b / 2.0) < 1e-12
              for a, b in zip(combine([m0, [4.0] * 5], "zscore_mean"),
                              combine([m0], "zscore_mean"))))

    print("== dispersion identity vs brute force")
    rnd = random.Random(11)
    worst = 0.0
    for _ in range(20):
        mem = [[rnd.gauss(0, 1) for _ in range(9)] for _ in range(5)]
        pairs = [spearman(mem[i], mem[j])
                 for i in range(len(mem)) for j in range(i + 1, len(mem))]
        worst = max(worst, abs(mean(pairs) - mean_pairwise_spearman(mem)))
    check("O(m*n) mean pairwise Spearman matches the O(m^2) brute force",
          worst < 1e-12, "max abs diff %g" % worst)
    check("dispersion of identical members is 1.0",
          abs(mean_pairwise_spearman([m0, [x * 3 for x in m0]]) - 1.0) < 1e-12)
    check("dispersion of a single member is NaN",
          mean_pairwise_spearman([m0]) != mean_pairwise_spearman([m0]))

    print("== end-to-end on a synthetic run (wide + elites + sidecar panel)")
    root = tempfile.mkdtemp(prefix="score_ensemble_selftest_")
    try:
        run, side = build_fixture(root)
        gb_dir = os.path.join(root, "out_gb")
        mine = _quiet(score, _args_for(run, side, ensemble="global_best",
                                       out_dir=gb_dir))
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(here, "score_stream.py"),
               "--run-dir", run, "--sidecar-dir", side, "--step", "2",
               "--length", "6", "--num-training-windows", "3",
               "--validation-sets", "1", "--param", "RET",
               "--score-from", "2000-01-01", "--top-k", "2",
               "--book", "sleeves", "--hold-days", "2", "--rolling-window", "5",
               "--max-horizon", "3", "--out-dir", os.path.join(root, "out_ss"),
               "--emit-json"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if p.returncode != 0:
            check("score_stream.py runs on the fixture", False, p.stderr[-500:])
        else:
            blob = p.stdout[p.stdout.rindex("\n{\n") + 1:]
            ref = json.loads(blob)

            def clean(o):
                # round-trip through JSON so int horizon keys become the
                # strings score_stream's own JSON carries
                if isinstance(o, float):
                    return None if o != o else o
                if isinstance(o, dict):
                    return dict((k, clean(v)) for k, v in o.items())
                return o

            def rt(o):
                return json.loads(json.dumps(clean(o)))
            check("--ensemble global_best reproduces score_stream's summary "
                  "field for field",
                  rt(mine["summary"]) == ref["summary"],
                  "\n    mine=%s\n    ref =%s" % (rt(mine["summary"])["overall"],
                                                  ref["summary"]["overall"]))
            check("--ensemble global_best reproduces score_stream's horizon "
                  "rank ICs",
                  rt(mine["horizon_rank_ic"]) == ref["horizon_rank_ic"],
                  "\n    mine=%s\n    ref =%s" % (rt(mine["horizon_rank_ic"]),
                                                  ref["horizon_rank_ic"]))

        res = {}
        for rule in ("rank_mean", "zscore_mean", "mean"):
            res[rule] = _quiet(score, _args_for(
                run, side, ensemble="all_elites", combine=rule,
                out_dir=os.path.join(root, "out_" + rule)))
        ic = dict((r, res[r]["summary"]["overall"]["model"]["rank_ic_1"])
                  for r in res)
        print("  ensemble rank IC@1 on the fixture: "
              + "  ".join("%s %s" % (r, fmt(ic[r])) for r in
                          ("rank_mean", "zscore_mean", "mean")))
        check("rank_mean beats raw mean when one member saturates at 1e6",
              ic["rank_mean"] > ic["mean"] + 0.1,
              "rank_mean %.4f vs mean %.4f" % (ic["rank_mean"], ic["mean"]))
        d = res["rank_mean"]["diagnostics"]
        print("  dispersion mean %s; member rank IC@1 mean %s (min %s max %s); "
              "ensemble %s" % (fmt(d["dispersion"]["mean"]),
                               fmt(d["member_rank_ic_1"]["mean"]),
                               fmt(d["member_rank_ic_1"]["min"]),
                               fmt(d["member_rank_ic_1"]["max"]),
                               fmt(d["ensemble_rank_ic_1"])))
        check("member dispersion is strictly below 1.0",
              d["dispersion"]["mean"] < 1.0)
        check("the ensemble beats its average member on the fixture",
              d["ensemble_minus_mean_member"] > 0)
        check("all_elites picks up all 4 member slots",
              d["member_rank_ic_1"]["n_slots"] == 4)
        champs = _quiet(score, _args_for(run, side, ensemble="island_champions",
                                         out_dir=os.path.join(root, "out_ch")))
        check("island_champions picks one member per island",
              champs["diagnostics"]["members_per_day"]["mean"] == 2.0)
        check("island_champions picks elite_rank 0 of each island",
              champs["diagnostics"]["member_slots"]
              == ["island0_elite0", "island1_elite0"])
        t1 = _quiet(score, _args_for(run, side, ensemble="topn", top_n=3,
                                     out_dir=os.path.join(root, "out_t3")))
        check("topn N=3 takes rank 0 of both islands, then rank 1 of island 0",
              t1["diagnostics"]["members_per_day"]["mean"] == 3.0
              and t1["diagnostics"]["member_slots"]
              == ["island0_elite0", "island0_elite1", "island1_elite0"],
              "got %s" % (t1["diagnostics"]["member_slots"],))
        dv = _quiet(score, _args_for(run, side, ensemble="diverse", top_n=2,
                                     diverse_gate=1,
                                     out_dir=os.path.join(root, "out_dv")))
        # member (1,1) is pure noise: maximally dissimilar to everything, so a
        # gate that admits it (gate=1) must see greedy diversity pick it up
        check("diverse N=2 gate=1 runs end-to-end and picks exactly 2 members",
              dv["diagnostics"]["members_per_day"]["mean"] == 2.0,
              "got %s" % (dv["diagnostics"]["member_slots"],))
        check("diverse selection picks up the dissimilar (noise) member",
              "island1_elite1" in dv["diagnostics"]["member_slots"],
              "got %s" % (dv["diagnostics"]["member_slots"],))
        dv0 = _quiet(score, _args_for(run, side, ensemble="diverse", top_n=2,
                                      diverse_gate=0,
                                      out_dir=os.path.join(root, "out_dv0")))
        check("diverse gate=0 restricts candidates to the island champions",
              dv0["diagnostics"]["member_slots"]
              == ["island0_elite0", "island1_elite0"],
              "got %s" % (dv0["diagnostics"]["member_slots"],))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    print("%d passed, %d failed" % (ok[0], len(fail)))
    for f in fail:
        print("  failed: %s" % f)
    return 1 if fail else 0


# ---------------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir")
    ap.add_argument("--sidecar-dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--length", type=int)
    ap.add_argument("--num-training-windows", type=int)
    ap.add_argument("--validation-sets", type=int)
    ap.add_argument("--param", default="RET",
                    help="target the model was TRAINED on; names the "
                         "expected_/naive_/global_best_predicted_ column groups "
                         "of the wide prediction files (e.g. RET, RET_CS)")
    ap.add_argument("--realized", choices=("auto", "sidecar", "expected"),
                    default="auto",
                    help="series every metric is scored against; see "
                         "score_stream.py --realized")
    ap.add_argument("--score-from", default="2020-01-01",
                    help="first date (inclusive) to score; ISO yyyy-mm-dd")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--book", choices=("algo1", "sleeves"), default="algo1")
    ap.add_argument("--hold-days", type=int, default=10, metavar="H")
    ap.add_argument("--beta-neutral", action="store_true")
    ap.add_argument("--beta-window", type=int, default=60, metavar="N")
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--rolling-window", type=int, default=63)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--out-dir", default=None,
                    help="where to write the ensemble_*.csv outputs "
                         "(default: run dir)")
    ap.add_argument("--emit-json", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip empirical row-mapping verification (assume +2)")
    ap.add_argument("--ensemble", default="island_champions",
                    choices=("global_best", "island_champions", "all_elites",
                             "topn", "diverse"),
                    help="which elite genomes are the ensemble members "
                         "(default island_champions)")
    ap.add_argument("--diverse-gate", type=int, default=3, metavar="R",
                    help="--ensemble diverse: only elites with elite_rank <= R "
                         "are candidates (the within-island validation-MSE "
                         "rank is the merit gate; default 3, i.e. the top half "
                         "of an 8-elite island)")
    ap.add_argument("--top-n", type=int, default=8, metavar="N",
                    help="member count for --ensemble topn; members are taken "
                         "in (elite_rank, island) order, i.e. every island's "
                         "rank-0 elite in ascending island order first, then "
                         "every rank-1 elite, and so on (the elites file has no "
                         "fitness column, so island index is the deterministic "
                         "tie-break between equal ranks)")
    ap.add_argument("--combine", default="rank_mean",
                    choices=("rank_mean", "zscore_mean", "mean"),
                    help="cross-sectional combination rule (default rank_mean, "
                         "the only one invariant to per-member rescaling)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the unit tests and a synthetic end-to-end check, "
                         "then exit; needs no run directory")
    args = ap.parse_args(argv)
    if args.self_test:
        return args
    required = ("run_dir", "sidecar_dir", "step", "length",
                "num_training_windows", "validation_sets")
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        ap.error("the following arguments are required: "
                 + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.hold_days < 1:
        ap.error("--hold-days must be >= 1")
    if args.top_n < 1:
        ap.error("--top-n must be >= 1")
    return args


def main():
    args = parse_args()
    if args.self_test:
        sys.exit(run_self_test())
    score(args)


if __name__ == "__main__":
    main()
