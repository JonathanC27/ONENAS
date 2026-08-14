#!/usr/bin/env python3
"""Stitch pooled ONE-NAS per-generation predictions into a daily stream and score it.

Reads generation_<g>_global_best.csv files from a sliding-window pooled run,
maps each file row to a panel date via the sidecar panel_dates.csv, keeps only
the newest `step` rows of each generation (consecutive windows overlap), and
computes:

  1. Daily cross-sectional Pearson IC and Spearman rank IC (mean, SE, t-stat).
  2. Multi-horizon rank IC h=1..10 vs forward cumulative real returns.
  3. A daily long-short book, in one of two constructions (--book):
       algo1   (default) the ICAIF Algorithm 1 -- top-10 long / bottom-10
               short, rebalanced only when all top-10 preds > 0 and all
               bottom-10 < 0, else held;
       sleeves Jegadeesh-Titman overlapping portfolios -- one new sleeve every
               day, held exactly --hold-days days, 1/H of capital per side;
     both with per-trade costs of TC/PRC on *netted* traded notional.
  4. Naive-persistence baselines of 1 and 3.

Realized series (--realized, see below):
  The model may be TRAINED on a transformed target -- e.g. --param RET_CS, a
  per-date cross-sectional rank-normal transform of the next-day return.  The
  expected_<param>_s<k> columns of the prediction files then hold rank-normal
  scores, NOT returns, and using them as the realized series would make the
  long-short book's P&L and the multi-horizon cumulative returns meaningless.

  The v2 panels therefore carry the untransformed next-day simple returns in
  the sidecar as RET_raw_<TICKER> columns of panel_dates.csv, in the same
  sorted-ticker order as PRC_/TC_.  When they are present (--realized auto, the
  default) they are the realized series for EVERY metric -- Pearson IC, rank
  IC, multi-horizon rank IC and the book -- and the expected_<param> columns
  are used only to verify the file-row -> panel-row mapping.  Older panels
  (set*_clean) have no RET_raw_ columns; there the expected_<param> columns are
  the realized series, exactly as before.

  Note the predicted/naive SIGNALS always stay in <param> units: they only ever
  get sorted and rank-correlated, and --book algo1's rebalance trigger tests
  their sign, which a rank-normal transform preserves.

Scale invariance (--book sleeves):
  The Algorithm 1 trigger reads the LEVEL of the predictions, so it is not
  invariant to rescaling them.  Under a cross-sectional rank-normal target
  (--param RET_CS) the models emit rank-normal scores whose signs flip roughly
  half the names every day, the trigger fires nearly every day, and turnover
  explodes -- the trigger ends up measuring the signal's scale, not its
  content.  --book sleeves uses nothing but the cross-sectional ORDER of the
  predictions, so its net %, Sharpe, MDD, turnover and cost % are identical
  under any strictly monotone per-day rescaling of the signal.

Row mapping (VERIFIED EMPIRICALLY, see verify_mapping()):
  Generation g has clock window cw = g + num_training_windows and test window
  tw = cw + V.  The input window starts at panel row tw*step + 1 (NOT tw*step),
  and with --time_offset 1 the L-1 file rows are targets for panel rows
      tw*step + 2 + i          for file row i in [0, L-1)
  i.e. an offset of +2 from tw*step, one more than the naive derivation
  (window start tw*step, targets tw*step + 1 + i).  The offset is detected and
  re-verified at startup against the per-stock CSVs in the sidecar dir by
  matching the expected_<param>_s<k> columns to the stock's real column.
  Stock s<k> is the k-th ticker in sorted order of the sidecar CSV filenames,
  which also matches the PRC_/TC_ column order in panel_dates.csv (verified).

Each generation contributes its newest `step` file rows, indices
(L-1-step)..(L-2); coverage across generations is asserted contiguous.

Book accounting: $100 base capital, $100 long + $100 short notional
(equal-weight, $100/top_k per name).  A rebalance moves the book to the new
target weights and costs are charged on the NETTED change in each name's
notional, not on a full liquidate-and-rebuild: a name that stays on the same
side at the same target weight only pays for its drift, a name that flips
long->short pays the full round trip, a name that leaves the book pays to be
closed.  Cost fraction per unit traded notional is TC/PRC priced at the
previous panel row (= trade at prior close), or a flat one-way --cost-bps.
Positions accrue that same day's realized return.  Daily net return is
(P&L - costs) / initial $100 capital, so cumulative net % is additive.

--book sleeves: every day forms ONE new sleeve -- top_k long / bottom_k short
by that day's cross-sectional prediction rank, $100/H per side and hence
($100/H)/top_k per name -- and retires the sleeve formed H days earlier.  At
steady state H sleeves are live, so the aggregate book is again $100 long +
$100 short and 1/H of it rolls each day.  The traded notional and the cost are
computed on the change in the AGGREGATE position per name, so a name held by
several live sleeves is never double-traded; the same netted TC/PRC (or
--cost-bps) pricing at the previous panel row applies.  Every live sleeve
accrues that day's realized return and drifts with it, exactly as algo1's book
does.  There is no trigger and no threshold: only the rank ordering is read.

--beta-neutral: scales the SHORT leg by s = beta(long leg) / beta(short leg)
so the book's ex-ante exposure to the panel's equal-weight mean return is ~0.
Per-stock betas are OLS slopes against that equal-weight mean over the
--beta-window (default 60) panel rows STRICTLY BEFORE the formation day, so
the construction stays causal.  Under --book sleeves the scale is applied per
sleeve at its own formation day; under --book algo1 it is applied to the
aggregate target book at each rebalance.  s is clipped to [0.2, 5] and falls
back to 1.0 when either leg's summed beta is non-positive or the window holds
fewer than 20 usable rows.  The book is then no longer exactly $100/$100:
turnover stays normalised by the nominal $200 gross so it stays comparable.

Outputs (to --out-dir, default <run-dir>):
  stitched_predictions.csv   date,stock,pred,real,naive  (tidy, full stream);
                             `real` is whichever realized series was used, so
                             raw returns under --realized sidecar and the
                             expected_<param> values otherwise
  book_daily.csv             daily book returns/equity for model and naive
  rolling_rank_ic.csv        trailing 63-day mean daily rank IC
plus a per-year + overall summary table on stdout (--emit-json for JSON).

Python 3 stdlib only.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

NAN = float("nan")


# ---------------------------------------------------------------- statistics

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def mean(v):
    return sum(v) / len(v) if v else NAN


def std(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def mean_se_t(v):
    if not v:
        return NAN, NAN, NAN
    m = mean(v)
    s = std(v)
    if not (s == s) or s == 0:
        return m, NAN, NAN
    se = s / math.sqrt(len(v))
    return m, se, m / se


# ------------------------------------------------------------------- loading

def load_panel(sidecar_dir):
    """Return (dates, tickers, prc, tc, ret_raw).

    prc/tc/ret_raw are [row][stock] lists.  ret_raw holds the untransformed
    next-day simple returns from the RET_raw_<TICKER> columns, and is None on
    older panels that do not carry them.
    """
    path = os.path.join(sidecar_dir, "panel_dates.csv")
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        prc_cols = [(i, c[4:]) for i, c in enumerate(header) if c.startswith("PRC_")]
        tc_cols = [(i, c[3:]) for i, c in enumerate(header) if c.startswith("TC_")]
        raw_cols = [(i, c[8:]) for i, c in enumerate(header)
                    if c.startswith("RET_raw_")]
        tickers = [t for _, t in prc_cols]
        if [t for _, t in tc_cols] != tickers:
            sys.exit("panel_dates.csv: TC_ column order does not match PRC_ order")
        if raw_cols and [t for _, t in raw_cols] != tickers:
            sys.exit(
                "panel_dates.csv: RET_raw_ column order does not match PRC_ "
                "order; the realized-return -> stock index mapping would be "
                f"wrong (PRC_ {tickers}, RET_raw_ {[t for _, t in raw_cols]})"
            )
        date_i = header.index("date")
        dates, prc, tc = [], [], []
        ret_raw = [] if raw_cols else None
        for row in rdr:
            dates.append(row[date_i])
            prc.append([float(row[i]) for i, _ in prc_cols])
            tc.append([float(row[i]) for i, _ in tc_cols])
            if raw_cols:
                ret_raw.append([float(row[i]) for i, _ in raw_cols])
    # cross-check against sorted per-stock CSV filenames when present
    csvs = sorted(
        f[:-4]
        for f in os.listdir(sidecar_dir)
        if f.endswith(".csv") and f != "panel_dates.csv"
    )
    if csvs and csvs != tickers:
        sys.exit(
            "sorted per-stock CSV names do not match PRC_ column order in "
            "panel_dates.csv; stock index mapping would be ambiguous"
        )
    return dates, tickers, prc, tc, ret_raw


def load_stock_series(sidecar_dir, tickers, col):
    """{panel_row: [n_stocks]} read from column `col` of the per-stock CSVs.

    The per-stock CSVs are indexed by panel row in the same order as
    panel_dates.csv (this is what verify_mapping relies on).  Returns None when
    any file or the column is missing, so callers can fall back.
    """
    series = []
    for t in tickers:
        p = os.path.join(sidecar_dir, t + ".csv")
        if not os.path.exists(p):
            return None
        with open(p, newline="") as fh:
            rdr = csv.reader(fh)
            hdr = next(rdr)
            if col not in hdr:
                return None
            ci = hdr.index(col)
            series.append([float(r[ci]) for r in rdr])
    n = min(len(s) for s in series)
    return {r: [s[r] for s in series] for r in range(n)}


def load_generation(path, param, n_stocks):
    """Return (pred, real, naive) as [row][stock] float lists."""
    with open(path, newline="") as fh:
        header = fh.readline().lstrip("#").strip().split(",")
        rows = [list(map(float, r)) for r in csv.reader(fh) if r]
    pat = re.compile(
        r"^(expected|naive|global_best_predicted)_%s_s(\d+)$" % re.escape(param)
    )
    cols = {}
    for idx, name in enumerate(header):
        m = pat.match(name)
        if m:
            cols.setdefault(int(m.group(2)), {})[m.group(1)] = idx
    if sorted(cols) != list(range(n_stocks)):
        sys.exit(f"{path}: expected s0..s{n_stocks - 1} column groups for {param}")
    exp_i = [cols[s]["expected"] for s in range(n_stocks)]
    nai_i = [cols[s]["naive"] for s in range(n_stocks)]
    prd_i = [cols[s]["global_best_predicted"] for s in range(n_stocks)]
    pred = [[r[i] for i in prd_i] for r in rows]
    real = [[r[i] for i in exp_i] for r in rows]
    naive = [[r[i] for i in nai_i] for r in rows]
    return pred, real, naive


def find_generations(run_dir):
    files = glob.glob(os.path.join(run_dir, "generation_*_global_best.csv"))
    gens = sorted(
        int(re.search(r"generation_(\d+)_global_best", os.path.basename(p)).group(1))
        for p in files
    )
    if not gens:
        sys.exit(f"no generation_*_global_best.csv files in {run_dir}")
    if gens != list(range(gens[0], gens[-1] + 1)):
        sys.exit(f"generation files are not contiguous: {gens[0]}..{gens[-1]}")
    if gens[0] != 0:
        print(f"# warning: first generation is {gens[0]}, not 0", file=sys.stderr)
    return gens


# --------------------------------------------------------- mapping + stitching

def verify_mapping(run_dir, sidecar_dir, tickers, gens, args):
    """Empirically detect the panel-row offset of file row 0 from tw*step.

    Matches expected_<param>_s<k> columns of a few generation files against
    the <param> column of the k-th sorted per-stock CSV.  Returns the unique
    offset in 0..3 that matches every sampled (generation, stock) pair.

    This is a check on the ROW MAPPING only, so it always runs against the
    training target <param> (the column the prediction file's expected_ values
    were written from) regardless of which series is used to score.
    """
    param, step, ntw, V = args.param, args.step, args.num_training_windows, args.validation_sets
    series = {}
    for t in tickers:
        p = os.path.join(sidecar_dir, t + ".csv")
        if not os.path.exists(p):
            print(f"# warning: {p} missing; skipping empirical row-mapping "
                  f"verification, assuming offset 2", file=sys.stderr)
            return 2
        with open(p, newline="") as fh:
            rdr = csv.reader(fh)
            hdr = next(rdr)
            if param not in hdr:
                sys.exit(
                    f"{p}: no column named '{param}' (header: {','.join(hdr)}); "
                    f"--param must name a column of the per-stock CSVs so the "
                    f"expected_{param}_s<k> values can be matched against it"
                )
            ci = hdr.index(param)
            series[t] = [float(row[ci]) for row in rdr]
    sample = sorted({gens[0], gens[len(gens) // 2], gens[-1]})
    offsets = set()
    for g in sample:
        _, real, _ = load_generation(
            os.path.join(run_dir, f"generation_{g}_global_best.csv"), param, len(tickers)
        )
        base = (g + ntw + V) * step
        for k, t in enumerate(tickers):
            matched = [
                off
                for off in range(4)
                if all(
                    abs(real[i][k] - series[t][base + off + i]) < 1e-4
                    for i in range(len(real))
                )
            ]
            if len(matched) != 1:
                sys.exit(
                    f"row-mapping verification failed for generation {g} stock "
                    f"s{k} ({t}): expected_{param}_s{k} could not be matched "
                    f"against column '{param}' of "
                    f"{os.path.join(sidecar_dir, t + '.csv')} at a unique "
                    f"panel-row offset in 0..3 (offsets matched = {matched})"
                )
            offsets.add(matched[0])
    if len(offsets) != 1:
        sys.exit(f"row-mapping offset inconsistent across samples: {sorted(offsets)}")
    off = offsets.pop()
    print(
        f"# row mapping verified on generations {sample}, all {len(tickers)} "
        f"stocks: file row i -> panel row tw*step + {off} + i "
        f"(tw = g + num_training_windows + V)"
    )
    if off != 2:
        print(f"# note: offset {off} differs from the expected +2", file=sys.stderr)
    return off


def stitch(run_dir, gens, offset, dates, n_stocks, args, realized=None):
    """Return list of days: (panel_row, date, pred[], real[], naive[]).

    `realized`, when given, is the panel's [row][stock] raw-return array and
    replaces the prediction file's expected_<param> values in the `real` slot,
    so every downstream metric scores against raw returns.  When None the
    expected_<param> values are the realized series, as before.
    """
    step, L, ntw, V = args.step, args.length, args.num_training_windows, args.validation_sets
    stream = []
    prev_end = None
    for g in gens:
        path = os.path.join(run_dir, f"generation_{g}_global_best.csv")
        pred, real, naive = load_generation(path, args.param, n_stocks)
        if len(pred) != L - 1:
            sys.exit(f"{path}: {len(pred)} rows, expected L-1 = {L - 1}")
        base = (g + ntw + V) * step + offset
        # newest `step` rows only: file rows (L-1-step)..(L-2)
        for i in range(L - 1 - step, L - 1):
            row = base + i
            if row >= len(dates):
                sys.exit(f"{path}: panel row {row} beyond panel ({len(dates)} rows)")
            if prev_end is not None and row != prev_end + 1:
                sys.exit(
                    f"stitched coverage not contiguous at generation {g}: "
                    f"panel row {row} follows {prev_end}"
                )
            prev_end = row
            real_i = real[i] if realized is None else realized[row]
            stream.append((row, dates[row], pred[i], real_i, naive[i]))
    return stream


# ----------------------------------------------------------------- the book

CAPITAL = 100.0            # base capital; daily net returns are P&L / CAPITAL
GROSS_NOTIONAL = 200.0     # $100 long + $100 short when fully invested

BETA_SCALE_LO = 0.2        # clip on the beta-neutral short-leg scale
BETA_SCALE_HI = 5.0
BETA_MIN_OBS = 20          # usable rows needed in the window to trust a beta


def _cost_frac(prc, tc, price_row, cost_bps):
    """Cost fraction per unit traded notional, as a function of the stock."""
    if cost_bps is None:
        return lambda k: tc[price_row][k] / abs(prc[price_row][k])
    return lambda k: cost_bps / 1e4


def rolling_betas(days, ret_by_row, window):
    """Per-day, per-stock beta against the panel's equal-weight mean return.

    ret_by_row maps panel row -> [n_stocks] realized returns and must cover
    rows BEFORE the scored range too.  For a day at panel row r the OLS slope
    uses rows r-window .. r-1 only, so nothing from the formation day or later
    can leak in.  Returns a list parallel to `days`; an entry is None when the
    window has fewer than BETA_MIN_OBS usable rows or the market return has no
    variation, and the caller then leaves the book un-neutralised that day.
    """
    mkt = {}
    for r, v in ret_by_row.items():
        fin = [x for x in v if x == x]
        if fin:
            mkt[r] = sum(fin) / len(fin)
    out = []
    for day in days:
        r0 = day[0]
        rows = [r for r in range(r0 - window, r0) if r in mkt]
        if len(rows) < BETA_MIN_OBS:
            out.append(None)
            continue
        m = [mkt[r] for r in rows]
        mm = sum(m) / len(m)
        var = sum((y - mm) ** 2 for y in m)
        if var <= 0.0:
            out.append(None)
            continue
        md = [y - mm for y in m]
        betas = []
        for k in range(len(day[3])):
            xs = [ret_by_row[r][k] for r in rows]
            mx = sum(xs) / len(xs)
            betas.append(sum((x - mx) * d for x, d in zip(xs, md)) / var)
        out.append(betas)
    return out


def short_leg_scale(beta, top, bot):
    """Short-leg multiplier s making the two legs' betas cancel.

    Equal weight w per name gives a long-leg market exposure of w*sum(beta over
    top) and a short-leg exposure of -s*w*sum(beta over bot); setting the sum
    to zero gives s = sum_top / sum_bot.  Returns 1.0 (no neutralisation) when
    betas are unavailable or either leg's summed beta is non-positive, since a
    negative s would mean going long the short leg.
    """
    if beta is None:
        return 1.0
    bl = sum(beta[k] for k in top)
    bs = sum(beta[k] for k in bot)
    if not (bl == bl and bs == bs) or bl <= 0.0 or bs <= 0.0:
        return 1.0
    return min(BETA_SCALE_HI, max(BETA_SCALE_LO, bl / bs))


def run_book(days, signal_idx, prc, tc, top_k, cost_bps=None,
             book="algo1", hold_days=10, betas=None):
    """Long-short book over the stitched stream.

    days: list of (panel_row, date, pred, real, naive); signal_idx selects
    which tuple slot (2=model pred, 4=naive) drives the sort.

    book: "algo1" (default, the historical construction) or "sleeves"
    (Jegadeesh-Titman overlapping portfolios with a --hold-days holding
    period).  hold_days is ignored by algo1.

    cost_bps: flat one-way cost in basis points on traded notional; when
    None the per-name TC/|PRC| of the previous panel row is used instead.

    betas: optional list parallel to `days` of [n_stocks] betas against the
    panel's equal-weight mean return (see rolling_betas); when given the short
    leg is scaled so the book's market exposure is ~zero.

    The signature is positional-compatible with the pre-`--book` version --
    run_book(days, idx, prc, tc, top_k, cost_bps) is unchanged in both meaning
    and output -- because the baselines call it that way.

    Returns dict with daily series and stats.
    """
    if book == "algo1":
        return _run_algo1(days, signal_idx, prc, tc, top_k, cost_bps, betas)
    if book == "sleeves":
        return _run_sleeves(days, signal_idx, prc, tc, top_k, cost_bps,
                            hold_days, betas)
    raise ValueError(f"unknown book construction {book!r}")


def _run_algo1(days, signal_idx, prc, tc, top_k, cost_bps, betas):
    """ICAIF Algorithm 1 long-short book with netted rebalancing.

    Positions are signed notionals that drift with realized returns.  The
    rebalance trigger is unchanged (all top_k signals > 0 and all bottom_k
    < 0, otherwise hold), but a rebalance moves the book to the target
    weights and charges cost only on |target - current| per name, so a name
    that stays on the same side pays only for its drift while a name that
    flips sides pays the full round trip.

    NOTE the trigger reads the SIGN of the signal, so this book is not
    invariant to rescaling the predictions -- see --book sleeves.
    """
    positions = {}  # stock -> signed notional
    daily_ret, rebal_flags, cost_series, traded_series = [], [], [], []
    rebal_idx = []
    per_name = CAPITAL / top_k
    for di, day in enumerate(days):
        row, _, _, real, _ = day[0], day[1], day[2], day[3], day[4]
        sig = day[signal_idx]
        order = sorted(range(len(sig)), key=lambda k: sig[k], reverse=True)
        top, bot = order[:top_k], order[-top_k:]
        cost = 0.0
        traded = 0.0
        rebal = all(sig[k] > 0 for k in top) and all(sig[k] < 0 for k in bot)
        if rebal:
            price_row = max(row - 1, 0)  # trade at prior close
            frac = _cost_frac(prc, tc, price_row, cost_bps)
            s = short_leg_scale(betas[di] if betas is not None else None,
                                top, bot)
            target = {k: per_name for k in top}
            target.update({k: -per_name * s for k in bot})
            for k in set(positions) | set(target):
                delta = target.get(k, 0.0) - positions.get(k, 0.0)
                if delta == 0.0:
                    continue
                traded += abs(delta)
                cost += abs(delta) * frac(k)
            positions = dict(target)
            rebal_idx.append(di)
        pnl = 0.0
        for k in list(positions):
            r = real[k]
            pnl += positions[k] * r
            positions[k] *= 1.0 + r
        daily_ret.append((pnl - cost) / CAPITAL)
        rebal_flags.append(1 if rebal else 0)
        cost_series.append(cost)
        traded_series.append(traded)
    # holding periods: days between consecutive rebalances (+ tail segment)
    holds = [b - a for a, b in zip(rebal_idx, rebal_idx[1:])]
    if rebal_idx:
        holds.append(len(days) - rebal_idx[-1])
    return {
        "daily_ret": daily_ret,
        "rebalanced": rebal_flags,
        "cost": cost_series,
        "traded": traded_series,
        "n_rebalances": len(rebal_idx),
        "avg_holding_days": mean(holds) if holds else NAN,
    }


def _run_sleeves(days, signal_idx, prc, tc, top_k, cost_bps, hold_days, betas):
    """Jegadeesh-Titman overlapping sleeves: scale-invariant, 1/H turnover.

    Each day: aggregate the live sleeves, retire the one formed H days ago,
    form a new one from today's cross-sectional prediction RANK at
    ($100/H)/top_k per name per side, re-aggregate, and charge costs on the
    per-name change in the AGGREGATE notional (so a name several sleeves hold
    is netted, not traded twice).  Then every live sleeve accrues today's
    realized return.

    Nothing but the ordering of `sig` is read, so the daily return series --
    and hence net %, Sharpe, MDD, turnover and cost % -- is unchanged by any
    strictly monotone per-day transform of the predictions.
    """
    H = max(1, int(hold_days))
    per_name = (CAPITAL / H) / top_k
    sleeves = []  # list of [formation day index, {stock: signed notional}]
    daily_ret, rebal_flags, cost_series, traded_series = [], [], [], []
    for di, day in enumerate(days):
        row, _, _, real, _ = day[0], day[1], day[2], day[3], day[4]
        sig = day[signal_idx]
        order = sorted(range(len(sig)), key=lambda k: sig[k], reverse=True)
        top, bot = order[:top_k], order[-top_k:]

        before = {}
        for _, pos in sleeves:
            for k, v in pos.items():
                before[k] = before.get(k, 0.0) + v
        # a sleeve formed on day f is live for the returns of days f..f+H-1
        sleeves = [s for s in sleeves if s[0] > di - H]
        s_scale = short_leg_scale(betas[di] if betas is not None else None,
                                  top, bot)
        new = {k: per_name for k in top}
        new.update({k: -per_name * s_scale for k in bot})
        sleeves.append([di, new])
        after = {}
        for _, pos in sleeves:
            for k, v in pos.items():
                after[k] = after.get(k, 0.0) + v

        price_row = max(row - 1, 0)  # trade at prior close
        frac = _cost_frac(prc, tc, price_row, cost_bps)
        cost = 0.0
        traded = 0.0
        for k in set(before) | set(after):
            delta = after.get(k, 0.0) - before.get(k, 0.0)
            if delta == 0.0:
                continue
            traded += abs(delta)
            cost += abs(delta) * frac(k)

        pnl = 0.0
        for _, pos in sleeves:
            for k in list(pos):
                r = real[k]
                pnl += pos[k] * r
                pos[k] *= 1.0 + r
        daily_ret.append((pnl - cost) / CAPITAL)
        rebal_flags.append(1)  # a sleeve is formed every single day
        cost_series.append(cost)
        traded_series.append(traded)
    return {
        "daily_ret": daily_ret,
        "rebalanced": rebal_flags,
        "cost": cost_series,
        "traded": traded_series,
        "n_rebalances": len(days),
        # every sleeve is held exactly H days by construction; the min() only
        # matters for a sample shorter than the holding period
        "avg_holding_days": float(min(H, len(days))) if days else NAN,
    }


def book_stats(daily_ret):
    """net %, annualized Sharpe, max drawdown % from a daily net-return slice."""
    if not daily_ret:
        return NAN, NAN, NAN
    net = 100.0 * sum(daily_ret)
    m, s = mean(daily_ret), std(daily_ret)
    sharpe = m / s * math.sqrt(252) if s == s and s > 0 else NAN
    equity, peak, mdd = 100.0, 100.0, 0.0
    for r in daily_ret:
        equity += 100.0 * r
        peak = max(peak, equity)
        mdd = max(mdd, (peak - equity) / peak)
    return net, sharpe, 100.0 * mdd


# ------------------------------------------------------------------ metrics

def daily_ics(days, signal_idx, fn):
    out = []
    for day in days:
        ic = fn(day[signal_idx], day[3])
        out.append(ic if ic is not None else NAN)
    return out


def horizon_rank_ics(days, signal_idx, h):
    """Daily rank IC of signal vs forward cumulative h-day real return.

    Day t uses real returns of days t..t+h-1 (h=1 reduces to the plain daily
    rank IC).  Days without h future days are dropped (returned as NaN so the
    list stays date-aligned).
    """
    n = len(days)
    out = []
    for t in range(n):
        if t + h > n:
            out.append(NAN)
            continue
        sig = days[t][signal_idx]
        fwd = []
        for k in range(len(sig)):
            acc = 1.0
            for j in range(h):
                acc *= 1.0 + days[t + j][3][k]
            fwd.append(acc - 1.0)
        ic = spearman(sig, fwd)
        out.append(ic if ic is not None else NAN)
    return out


def finite(v):
    return [x for x in v if x == x]


# ------------------------------------------------------------------- output

def fmt(x, spec="{:+.4f}"):
    return spec.format(x) if x == x else "   nan"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--num-training-windows", type=int, required=True)
    ap.add_argument("--validation-sets", type=int, required=True)
    ap.add_argument("--param", default="RET",
                    help="target the model was TRAINED on; names the "
                         "expected_/naive_/global_best_predicted_ column "
                         "groups of the prediction files (e.g. RET, RET_CS)")
    ap.add_argument("--realized", choices=("auto", "sidecar", "expected"),
                    default="auto",
                    help="which series every metric is scored against: "
                         "'sidecar' = the panel's RET_raw_<TICKER> raw returns "
                         "(required when --param is a transform such as "
                         "RET_CS); 'expected' = the prediction file's "
                         "expected_<param> columns (pre-RET_raw behaviour); "
                         "'auto' (default) = sidecar when the panel carries "
                         "RET_raw_ columns, else expected")
    ap.add_argument("--score-from", default="2020-01-01",
                    help="first date (inclusive) to score; ISO yyyy-mm-dd")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--book", choices=("algo1", "sleeves"), default="algo1",
                    help="portfolio construction: 'algo1' (default) = the "
                         "ICAIF Algorithm 1 trigger book, kept as the default "
                         "so published numbers stay reproducible; 'sleeves' = "
                         "Jegadeesh-Titman overlapping portfolios, one new "
                         "sleeve per day held --hold-days days, driven purely "
                         "by cross-sectional rank and therefore invariant to "
                         "any monotone rescaling of the predictions")
    ap.add_argument("--hold-days", type=int, default=10, metavar="H",
                    help="sleeve holding period for --book sleeves; H sleeves "
                         "are live at steady state, each carrying 1/H of the "
                         "capital per side, so 1/H of the book rolls daily "
                         "(ignored by --book algo1)")
    ap.add_argument("--beta-neutral", action="store_true",
                    help="scale the short leg so the book's exposure to the "
                         "panel's equal-weight mean return is ~zero, using "
                         "per-stock betas estimated causally over the "
                         "--beta-window rows strictly before formation")
    ap.add_argument("--beta-window", type=int, default=60, metavar="N",
                    help="lookback in panel rows for the --beta-neutral betas "
                         "(default 60)")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="flat one-way transaction cost in basis points of "
                         "traded notional, overriding the panel's TC/PRC "
                         "columns (for cost-sensitivity runs)")
    ap.add_argument("--rolling-window", type=int, default=63)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--out-dir", default=None,
                    help="where to write CSV outputs (default: run dir)")
    ap.add_argument("--emit-json", action="store_true",
                    help="also print the summary as JSON on stdout")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip empirical row-mapping verification (assume +2)")
    args = ap.parse_args()

    if args.hold_days < 1:
        sys.exit("--hold-days must be >= 1")
    if args.beta_window < BETA_MIN_OBS:
        sys.exit(f"--beta-window must be >= {BETA_MIN_OBS}")

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    dates, tickers, prc, tc, ret_raw = load_panel(args.sidecar_dir)

    if args.realized == "sidecar" and ret_raw is None:
        sys.exit(
            f"--realized sidecar: {os.path.join(args.sidecar_dir, 'panel_dates.csv')} "
            f"has no RET_raw_<TICKER> columns (only the v2 panels carry them); "
            f"use --realized expected to score against the prediction file's "
            f"expected_{args.param} columns instead"
        )
    use_sidecar = ret_raw is not None if args.realized == "auto" else \
        args.realized == "sidecar"
    realized = ret_raw if use_sidecar else None
    realized_label = "sidecar RET_raw" if use_sidecar else f"expected_{args.param}"
    print(f"# realized: {realized_label} (param {args.param})")
    if not use_sidecar and args.param != "RET":
        print(f"# warning: scoring against expected_{args.param}, which is NOT "
              f"a return series; the book P&L and multi-horizon cumulative "
              f"returns are not meaningful", file=sys.stderr)

    gens = find_generations(args.run_dir)
    if args.skip_verify:
        offset = 2
        print("# row-mapping verification skipped; assuming offset +2")
    else:
        offset = verify_mapping(args.run_dir, args.sidecar_dir, tickers, gens, args)

    stream = stitch(args.run_dir, gens, offset, dates, len(tickers), args,
                    realized=realized)
    print(
        f"# stitched {len(stream)} days from {len(gens)} generations "
        f"({len(gens)}*step = {len(gens) * args.step}); "
        f"{stream[0][1]} .. {stream[-1][1]} (panel rows "
        f"{stream[0][0]}..{stream[-1][0]}, contiguous)"
    )

    # (a) tidy CSV of the full stitched stream
    tidy_path = os.path.join(out_dir, "stitched_predictions.csv")
    with open(tidy_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "stock", "pred", "real", "naive"])
        for _, date, pred, real, naive in stream:
            for k, t in enumerate(tickers):
                w.writerow([date, t, repr(pred[k]), repr(real[k]), repr(naive[k])])

    days = [d for d in stream if d[1] >= args.score_from]
    if not days:
        sys.exit(f"no stitched days on/after --score-from {args.score_from}")
    print(f"# scoring {len(days)} days from {days[0][1]} to {days[-1][1]} "
          f"(--score-from {args.score_from})")

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
                    src_label = f"per-stock sidecar column '{col}'"
                    break
            if src is None:
                src = {row: real for row, _, _, real, _ in stream}
                src_label = ("the stitched stream itself (no per-stock CSVs; "
                             "the first days have no pre-history and stay "
                             "un-neutralised)")
        betas = rolling_betas(days, src, args.beta_window)
        n_ok = sum(1 for b in betas if b is not None)
        print(f"# beta-neutral: {args.beta_window}-row causal betas vs the "
              f"equal-weight panel mean from {src_label}; "
              f"{n_ok}/{len(betas)} days neutralised")

    def book_of(idx):
        return run_book(days, idx, prc, tc, args.top_k, args.cost_bps,
                        book=args.book, hold_days=args.hold_days, betas=betas)

    books = {"model": book_of(PRED), "naive": book_of(NAIVE)}
    # The legacy algo1 book with no beta neutralisation must stay byte-identical
    # to pre-`--book` output, so this identifying line is printed only when the
    # construction is NOT that legacy default.  --emit-json's meta always
    # carries book/hold_days/beta_neutral regardless.
    if args.book != "algo1" or args.beta_neutral:
        h = f", H={args.hold_days}" if args.book == "sleeves" else ""
        print(f"# book: {args.book}{h}, top-k {args.top_k}, beta-neutral "
              f"{'on' if args.beta_neutral else 'off'}")
    print("# book costs: " + ("TC/PRC from the panel"
                              if args.cost_bps is None
                              else f"flat {args.cost_bps:g} bps one-way")
          + ", charged on netted traded notional")

    # (b) daily book returns
    book_path = os.path.join(out_dir, "book_daily.csv")
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
                rec += [f"{b['daily_ret'][i]:.8f}", f"{eq[name]:.4f}",
                        b["rebalanced"][i], f"{b['cost'][i]:.6f}",
                        f"{b['traded'][i]:.6f}"]
            w.writerow(rec)

    # (c) rolling mean rank IC
    roll_path = os.path.join(out_dir, "rolling_rank_ic.csv")
    W = args.rolling_window
    with open(roll_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", f"model_rank_ic_{W}d", f"naive_rank_ic_{W}d"])
        for i in range(W - 1, len(days)):
            m = mean(finite(spear["model"][i - W + 1:i + 1]))
            n = mean(finite(spear["naive"][i - W + 1:i + 1]))
            w.writerow([days[i][1], f"{m:.6f}", f"{n:.6f}"])

    # ------------------------------------------------------------- summary
    years = sorted({d[1][:4] for d in days})
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
        for h in (5, 10):
            if h <= args.max_horizon:
                row[f"rank_ic_{h}"] = mean(finite([hics[name][h][i] for i in ix]))
        net, sharpe, mdd = book_stats([books[name]["daily_ret"][i] for i in ix])
        row.update(net_pct=net, sharpe=sharpe, mdd_pct=mdd)
        # cost drag: total transaction cost as % of initial capital, and mean
        # daily traded notional as a fraction of the book's gross notional
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
            print(f"{p:<9} {name:<6} {r['n_days']:>5}  "
                  f"{fmt(r['pearson_ic'])}   {fmt(r['rank_ic_1'])}   "
                  f"{fmt(r.get('rank_ic_5', NAN))}   {fmt(r.get('rank_ic_10', NAN))}  "
                  f"{fmt(r['net_pct'], '{:+8.2f}')} {fmt(r['sharpe'], '{:+8.2f}')} "
                  f"{fmt(r['mdd_pct'], '{:8.2f}')} {fmt(r['turnover'], '{:8.4f}')} "
                  f"{fmt(r['cost_pct'], '{:8.2f}')}")
    print()
    for name in ("model", "naive"):
        r = summary["overall"][name]
        print(f"# {name} overall: pearson IC {fmt(r['pearson_ic'])} "
              f"(SE {fmt(r['pearson_se'], '{:.4f}')}, t {fmt(r['pearson_t'], '{:+.2f}')}); "
              f"rank IC {fmt(r['rank_ic_1'])} "
              f"(SE {fmt(r['rank_ic_se'], '{:.4f}')}, t {fmt(r['rank_ic_t'], '{:+.2f}')}); "
              f"rebalances {r['n_rebalances']}, "
              f"avg holding {fmt(r['avg_holding_days'], '{:.1f}')} days")
    hz = {name: {h: mean(finite(hics[name][h]))
                 for h in range(1, args.max_horizon + 1)}
          for name in ("model", "naive")}
    print("# rank IC by horizon (overall): " + "  ".join(
        f"h={h}:" + fmt(hz['model'][h]) for h in range(1, args.max_horizon + 1)))
    print("#   naive                     : " + "  ".join(
        f"h={h}:" + fmt(hz['naive'][h]) for h in range(1, args.max_horizon + 1)))
    print(f"# wrote {tidy_path}")
    print(f"# wrote {book_path}")
    print(f"# wrote {roll_path}")

    if args.emit_json:
        def clean(o):
            if isinstance(o, float):
                return None if o != o else o
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            return o
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
            },
            "summary": summary,
            "horizon_rank_ic": hz,
        }
        print(json.dumps(clean(payload), indent=1))


if __name__ == "__main__":
    main()
