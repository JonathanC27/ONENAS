#!/usr/bin/env python3
"""Score a tidy `date,stock,pred` prediction file with score_stream's metrics.

The metric definitions are NOT re-derived here.  This module imports
``score_stream`` from the parent directory and calls its ``pearson``,
``spearman``, ``daily_ics``, ``horizon_rank_ics``, ``run_book``,
``book_stats`` and ``mean_se_t`` functions, so a baseline's numbers and a
ONE-NAS run's numbers come out of exactly the same code path.

The bridge is the ``days`` list that score_stream builds by stitching
generation files:

    days[i] = (panel_row, date, pred[stock], real[stock], naive[stock])

with ``pred`` at signal index 2 and ``naive`` (= the previous day's realised
return) at index 4.  A baseline just has to supply column 2; ``real`` and
``naive`` come straight off the panel.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_stream as ss  # noqa: E402  (metric definitions live here)

_REQUIRED = ("pearson", "spearman", "mean_se_t", "run_book", "book_stats",
             "daily_ics", "horizon_rank_ics", "finite", "mean", "fmt",
             "CAPITAL", "GROSS_NOTIONAL")
_missing = [n for n in _REQUIRED if not hasattr(ss, n)]
if _missing:
    raise SystemExit(
        "score_stream.py is missing %s; the baselines reuse its metric code "
        "verbatim and cannot run without it" % ", ".join(_missing)
    )

PRED, NAIVE = 2, 4
NAN = float("nan")


def build_days(panel, preds, rows):
    """days list for score_stream's metric functions.

    preds: dict panel_row -> ndarray[n_stocks] (or [n_stocks] list)
    rows:  panel rows to score, ascending; every row must be in preds and
           have row-1 >= 0 so the naive column is defined.
    """
    days = []
    for r in rows:
        if r not in preds:
            raise SystemExit(f"no prediction for panel row {r} ({panel.dates[r]})")
        p = np.asarray(preds[r], dtype=float)
        if p.shape != (panel.n_stocks,):
            raise SystemExit(
                f"prediction for row {r} has shape {p.shape}, expected "
                f"({panel.n_stocks},)"
            )
        if not np.isfinite(p).all():
            raise SystemExit(f"non-finite prediction for row {r} ({panel.dates[r]})")
        days.append((r, panel.dates[r], list(p),
                     list(panel.Y[r]), list(panel.Y[r - 1])))
    return days


def summarize(panel, days, top_k=10, cost_bps=None, max_horizon=10,
              with_naive=True):
    """Per-year + overall stats, identical in definition to score_stream's."""
    names = ("model", "naive") if with_naive else ("model",)
    idx_map = {"model": PRED, "naive": NAIVE}

    pears = {n: ss.daily_ics(days, idx_map[n], ss.pearson) for n in names}
    hics = {n: {h: ss.horizon_rank_ics(days, idx_map[n], h)
                for h in range(1, max_horizon + 1)} for n in names}
    spear = {n: hics[n][1] for n in names}
    books = {n: ss.run_book(days, idx_map[n], panel.prc, panel.tc, top_k,
                            cost_bps) for n in names}

    years = sorted({d[1][:4] for d in days})
    periods = years + ["overall"]
    idx_of = {p: [i for i, d in enumerate(days)
                  if p == "overall" or d[1][:4] == p] for p in periods}

    def slice_stats(name, p):
        ix = idx_of[p]
        pm, pse, pt = ss.mean_se_t(ss.finite([pears[name][i] for i in ix]))
        rm, rse, rt = ss.mean_se_t(ss.finite([spear[name][i] for i in ix]))
        row = {"n_days": len(ix),
               "pearson_ic": pm, "pearson_se": pse, "pearson_t": pt,
               "rank_ic_1": rm, "rank_ic_se": rse, "rank_ic_t": rt}
        for h in (5, 10):
            if h <= max_horizon:
                row[f"rank_ic_{h}"] = ss.mean(
                    ss.finite([hics[name][h][i] for i in ix]))
        net, sharpe, mdd = ss.book_stats(
            [books[name]["daily_ret"][i] for i in ix])
        row.update(net_pct=net, sharpe=sharpe, mdd_pct=mdd)
        row["cost_pct"] = 100.0 * sum(
            books[name]["cost"][i] for i in ix) / ss.CAPITAL
        row["turnover"] = ss.mean(
            [books[name]["traded"][i] for i in ix]) / ss.GROSS_NOTIONAL
        if p == "overall":
            row["n_rebalances"] = books[name]["n_rebalances"]
            row["avg_holding_days"] = books[name]["avg_holding_days"]
        return row

    summary = {p: {n: slice_stats(n, p) for n in names} for p in periods}
    horizon = {n: {h: ss.mean(ss.finite(hics[n][h]))
                   for h in range(1, max_horizon + 1)} for n in names}
    return {"summary": summary, "horizon_rank_ic": horizon,
            "periods": periods, "names": list(names),
            "books": books}


def print_table(res, title=None):
    if title:
        print(f"# {title}")
    print("period    who     days  pearsonIC  rankIC@1  rankIC@5  rankIC@10"
          "      net%   sharpe     MDD% turnover    cost%")
    for p in res["periods"]:
        for name in res["names"]:
            r = res["summary"][p][name]
            print(f"{p:<9} {name:<6} {r['n_days']:>5}  "
                  f"{ss.fmt(r['pearson_ic'])}   {ss.fmt(r['rank_ic_1'])}   "
                  f"{ss.fmt(r.get('rank_ic_5', NAN))}   "
                  f"{ss.fmt(r.get('rank_ic_10', NAN))}  "
                  f"{ss.fmt(r['net_pct'], '{:+8.2f}')} "
                  f"{ss.fmt(r['sharpe'], '{:+8.2f}')} "
                  f"{ss.fmt(r['mdd_pct'], '{:8.2f}')} "
                  f"{ss.fmt(r['turnover'], '{:8.4f}')} "
                  f"{ss.fmt(r['cost_pct'], '{:8.2f}')}")
    for name in res["names"]:
        r = res["summary"]["overall"][name]
        print(f"# {name} overall: pearson IC {ss.fmt(r['pearson_ic'])} "
              f"(SE {ss.fmt(r['pearson_se'], '{:.4f}')}, "
              f"t {ss.fmt(r['pearson_t'], '{:+.2f}')}); "
              f"rank IC {ss.fmt(r['rank_ic_1'])} "
              f"(SE {ss.fmt(r['rank_ic_se'], '{:.4f}')}, "
              f"t {ss.fmt(r['rank_ic_t'], '{:+.2f}')}); "
              f"rebalances {r['n_rebalances']}, avg holding "
              f"{ss.fmt(r['avg_holding_days'], '{:.1f}')} days")
    for name in res["names"]:
        print(f"# rank IC by horizon ({name}): " + "  ".join(
            f"h={h}:" + ss.fmt(v) for h, v in
            sorted(res["horizon_rank_ic"][name].items())))


def _clean(o):
    if isinstance(o, float):
        return None if o != o else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_outputs(out_dir, panel, days, res, meta):
    """predictions.csv, book_daily.csv, metrics.json into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    import csv as _csv

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["date", "stock", "pred"])
        for _, date, pred, _, _ in days:
            for k, t in enumerate(panel.tickers):
                w.writerow([date, t, repr(float(pred[k]))])

    books = res["books"]
    with open(os.path.join(out_dir, "book_daily.csv"), "w", newline="") as fh:
        w = _csv.writer(fh)
        cols = ["date"]
        for n in res["names"]:
            cols += [f"{n}_ret", f"{n}_equity", f"{n}_rebalanced",
                     f"{n}_cost", f"{n}_traded"]
        w.writerow(cols)
        eq = {n: ss.CAPITAL for n in res["names"]}
        for i, day in enumerate(days):
            rec = [day[1]]
            for n in res["names"]:
                b = books[n]
                eq[n] += ss.CAPITAL * b["daily_ret"][i]
                rec += [f"{b['daily_ret'][i]:.8f}", f"{eq[n]:.4f}",
                        b["rebalanced"][i], f"{b['cost'][i]:.6f}",
                        f"{b['traded'][i]:.6f}"]
            w.writerow(rec)

    payload = {"meta": meta,
               "summary": res["summary"],
               "horizon_rank_ic": res["horizon_rank_ic"]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(_clean(payload), fh, indent=1)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(_clean(meta), fh, indent=1)


def score_and_write(panel, preds, rows, out_dir, meta, top_k=10,
                    cost_bps=None, max_horizon=10, quiet=False):
    days = build_days(panel, preds, rows)
    res = summarize(panel, days, top_k=top_k, cost_bps=cost_bps,
                    max_horizon=max_horizon)
    if out_dir:
        write_outputs(out_dir, panel, days, res, meta)
    if not quiet:
        print_table(res, title=meta.get("model", "baseline"))
    return res


def rank_ic_mean(panel, preds, rows):
    """Mean daily rank IC over `rows` -- the hyperparameter tuning objective."""
    days = build_days(panel, preds, rows)
    return ss.mean(ss.finite(ss.daily_ics(days, PRED, ss.spearman)))
