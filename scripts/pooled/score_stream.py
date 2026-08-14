#!/usr/bin/env python3
"""Stitch pooled ONE-NAS per-generation predictions into a daily stream and score it.

Reads generation_<g>_global_best.csv files from a sliding-window pooled run,
maps each file row to a panel date via the sidecar panel_dates.csv, keeps only
the newest `step` rows of each generation (consecutive windows overlap), and
computes:

  1. Daily cross-sectional Pearson IC and Spearman rank IC (mean, SE, t-stat).
  2. Multi-horizon rank IC h=1..10 vs forward cumulative real returns.
  3. A daily long-short book per the ICAIF Algorithm 1 (top-10 long / bottom-10
     short, rebalanced only when all top-10 preds > 0 and all bottom-10 < 0),
     with per-trade costs of TC/PRC on traded notional.
  4. Naive-persistence baselines of 1 and 3.

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
(equal-weight, $100/top_k per name).  On a rebalance day the old book is fully
liquidated and the new one built (both legs charged: cost fraction TC/PRC per
name, on traded notional, priced at the previous panel row = trade at prior
close).  Positions accrue that same day's realized return.  Daily net return
is (P&L - costs) / initial $100 capital, so cumulative net % is additive.

Outputs (to --out-dir, default <run-dir>):
  stitched_predictions.csv   date,stock,pred,real,naive  (tidy, full stream)
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
    """Return (dates, tickers, prc, tc): prc/tc are [row][stock] lists."""
    path = os.path.join(sidecar_dir, "panel_dates.csv")
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        prc_cols = [(i, c[4:]) for i, c in enumerate(header) if c.startswith("PRC_")]
        tc_cols = [(i, c[3:]) for i, c in enumerate(header) if c.startswith("TC_")]
        tickers = [t for _, t in prc_cols]
        if [t for _, t in tc_cols] != tickers:
            sys.exit("panel_dates.csv: TC_ column order does not match PRC_ order")
        date_i = header.index("date")
        dates, prc, tc = [], [], []
        for row in rdr:
            dates.append(row[date_i])
            prc.append([float(row[i]) for i, _ in prc_cols])
            tc.append([float(row[i]) for i, _ in tc_cols])
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
    return dates, tickers, prc, tc


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
            ci = next(rdr).index(param)
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
                    f"s{k} ({t}): offsets matched = {matched}"
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


def stitch(run_dir, gens, offset, dates, n_stocks, args):
    """Return list of days: (panel_row, date, pred[], real[], naive[])."""
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
            stream.append((row, dates[row], pred[i], real[i], naive[i]))
    return stream


# ----------------------------------------------------------------- the book

def run_book(days, signal_idx, prc, tc, top_k):
    """ICAIF Algorithm 1 long-short book.

    days: list of (panel_row, date, pred, real, naive); signal_idx selects
    which tuple slot (2=model pred, 4=naive) drives the sort.
    Returns dict with daily series and stats.
    """
    positions = {}  # stock -> signed notional
    daily_ret, rebal_flags, cost_series = [], [], []
    rebal_idx = []
    per_name = 100.0 / top_k
    for di, day in enumerate(days):
        row, _, _, real, _ = day[0], day[1], day[2], day[3], day[4]
        sig = day[signal_idx]
        order = sorted(range(len(sig)), key=lambda k: sig[k], reverse=True)
        top, bot = order[:top_k], order[-top_k:]
        cost = 0.0
        rebal = all(sig[k] > 0 for k in top) and all(sig[k] < 0 for k in bot)
        if rebal:
            price_row = max(row - 1, 0)  # trade at prior close
            frac = lambda k: tc[price_row][k] / abs(prc[price_row][k])
            for k, notional in positions.items():  # liquidate everything
                cost += abs(notional) * frac(k)
            positions = {}
            for k in top:
                positions[k] = per_name
                cost += per_name * frac(k)
            for k in bot:
                positions[k] = -per_name
                cost += per_name * frac(k)
            rebal_idx.append(di)
        pnl = 0.0
        for k in list(positions):
            r = real[k]
            pnl += positions[k] * r
            positions[k] *= 1.0 + r
        daily_ret.append((pnl - cost) / 100.0)
        rebal_flags.append(1 if rebal else 0)
        cost_series.append(cost)
    # holding periods: days between consecutive rebalances (+ tail segment)
    holds = [b - a for a, b in zip(rebal_idx, rebal_idx[1:])]
    if rebal_idx:
        holds.append(len(days) - rebal_idx[-1])
    return {
        "daily_ret": daily_ret,
        "rebalanced": rebal_flags,
        "cost": cost_series,
        "n_rebalances": len(rebal_idx),
        "avg_holding_days": mean(holds) if holds else NAN,
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
    ap.add_argument("--param", default="RET")
    ap.add_argument("--score-from", default="2020-01-01",
                    help="first date (inclusive) to score; ISO yyyy-mm-dd")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rolling-window", type=int, default=63)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--out-dir", default=None,
                    help="where to write CSV outputs (default: run dir)")
    ap.add_argument("--emit-json", action="store_true",
                    help="also print the summary as JSON on stdout")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip empirical row-mapping verification (assume +2)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    dates, tickers, prc, tc = load_panel(args.sidecar_dir)
    gens = find_generations(args.run_dir)
    if args.skip_verify:
        offset = 2
        print("# row-mapping verification skipped; assuming offset +2")
    else:
        offset = verify_mapping(args.run_dir, args.sidecar_dir, tickers, gens, args)

    stream = stitch(args.run_dir, gens, offset, dates, len(tickers), args)
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

    books = {"model": run_book(days, PRED, prc, tc, args.top_k),
             "naive": run_book(days, NAIVE, prc, tc, args.top_k)}

    # (b) daily book returns
    book_path = os.path.join(out_dir, "book_daily.csv")
    with open(book_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date",
                    "model_ret", "model_equity", "model_rebalanced", "model_cost",
                    "naive_ret", "naive_equity", "naive_rebalanced", "naive_cost"])
        eq = {"model": 100.0, "naive": 100.0}
        for i, day in enumerate(days):
            rec = [day[1]]
            for name in ("model", "naive"):
                b = books[name]
                eq[name] += 100.0 * b["daily_ret"][i]
                rec += [f"{b['daily_ret'][i]:.8f}", f"{eq[name]:.4f}",
                        b["rebalanced"][i], f"{b['cost'][i]:.6f}"]
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
        if p == "overall":
            row["n_rebalances"] = books[name]["n_rebalances"]
            row["avg_holding_days"] = books[name]["avg_holding_days"]
        return row

    summary = {p: {name: slice_stats(name, p) for name in ("model", "naive")}
               for p in periods}

    print()
    print("period    who     days  pearsonIC  rankIC@1  rankIC@5  rankIC@10"
          "      net%   sharpe     MDD%")
    for p in periods:
        for name in ("model", "naive"):
            r = summary[p][name]
            print(f"{p:<9} {name:<6} {r['n_days']:>5}  "
                  f"{fmt(r['pearson_ic'])}   {fmt(r['rank_ic_1'])}   "
                  f"{fmt(r.get('rank_ic_5', NAN))}   {fmt(r.get('rank_ic_10', NAN))}  "
                  f"{fmt(r['net_pct'], '{:+8.2f}')} {fmt(r['sharpe'], '{:+8.2f}')} "
                  f"{fmt(r['mdd_pct'], '{:8.2f}')}")
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
                "generations": len(gens),
                "row_offset": offset,
                "stitched_days": len(stream),
                "scored_days": len(days),
                "first_scored_date": days[0][1],
                "last_scored_date": days[-1][1],
                "score_from": args.score_from,
            },
            "summary": summary,
            "horizon_rank_ic": hz,
        }
        print(json.dumps(clean(payload), indent=1))


if __name__ == "__main__":
    main()
