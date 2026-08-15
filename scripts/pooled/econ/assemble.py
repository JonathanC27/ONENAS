#!/usr/bin/env python3
"""Assemble the economics comparison table from the re-booked runs.

Reads
  * results_econ/sleeves_econ.csv        (rebook.py output: every baseline,
                                          the ONE-NAS C7_E runs, EW buy&hold,
                                          all under sleeves H=10 + TC/PRC)
  * ../controls/results_controls/{sleeves_summary.csv,summary.csv}
                                          (READ-ONLY; the fixed-architecture
                                          controls, included for whatever
                                          core7 cells exist, marked partial)

Aggregation: per arm, seeds are averaged within a panel first, then the mean
and the standard error are taken ACROSS PANELS (n = number of panels with any
result), so every arm's +/- is a like-for-like panel-level SE.

Writes
  * results_econ/econ_table.csv       the paper table (mean +/- SE columns)
  * results_econ/econ_per_panel.csv   appendix: arm x panel means (over seeds)
  * stdout: formatted table + the paired headline comparisons
"""

import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_econ")
CONTROLS = os.path.join(os.path.dirname(HERE), "controls", "results_controls")

METRICS = ["rank_ic_1", "net_pct", "sharpe", "mdd_pct", "turnover", "cost_pct"]

ARM_ORDER = [
    "ew_buy_hold", "naive", "str1", "str5", "ridge", "ar", "lstm", "gru",
    "periodic_ridge_yearly", "periodic_ridge_quarterly",
    "periodic_ridge_monthly", "periodic_lstm_yearly",
    "periodic_lstm_quarterly", "periodic_lstm_monthly",
    "control1_random_single", "control1_random_ensemble",
    "control2_fixed_single", "control2_fixed_ensemble",
    "onenas_single", "onenas_ensemble",
]

LABELS = {
    "ew_buy_hold": "EW buy&hold (long-only ref)",
    "naive": "naive persistence",
    "str1": "STR1 (1-day reversal)",
    "str5": "STR5 (5-day reversal)",
    "ridge": "online ridge/RLS",
    "ar": "online AR (OGD/ONS)",
    "lstm": "online LSTM",
    "gru": "online GRU",
    "periodic_ridge_yearly": "periodic ridge, yearly",
    "periodic_ridge_quarterly": "periodic ridge, quarterly",
    "periodic_ridge_monthly": "periodic ridge, monthly",
    "periodic_lstm_yearly": "periodic LSTM, yearly",
    "periodic_lstm_quarterly": "periodic LSTM, quarterly",
    "periodic_lstm_monthly": "periodic LSTM, monthly",
    "control1_random_single": "control: random-arch RNN (single)",
    "control1_random_ensemble": "control: random-arch RNN ensemble",
    "control2_fixed_single": "control: fixed-arch RNN (single)",
    "control2_fixed_ensemble": "control: fixed-arch RNN ensemble",
    "onenas_single": "ONE-NAS evolved RNN (single)",
    "onenas_ensemble": "ONE-NAS evolved RNN ensemble",
}


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def mean_se(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd / math.sqrt(len(v))


def load_runs():
    """-> {arm: {panel: {seed: {metric: value}}}}"""
    runs = defaultdict(lambda: defaultdict(dict))
    path = os.path.join(RESULTS, "sleeves_econ.csv")
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            runs[rec["arm"]][rec["panel"]][int(rec["seed"])] = {
                m: fnum(rec.get(m)) for m in METRICS}

    # ------- fixed-architecture controls, read-only, core7 rows only -------
    ic = {}
    spath = os.path.join(CONTROLS, "summary.csv")
    if os.path.exists(spath):
        with open(spath, newline="") as fh:
            for rec in csv.DictReader(fh):
                ic[(rec["control"], rec["run"], rec["variant"])] = \
                    fnum(rec["rank_ic_1"])
    cpath = os.path.join(CONTROLS, "sleeves_summary.csv")
    if os.path.exists(cpath):
        with open(cpath, newline="") as fh:
            for rec in csv.DictReader(fh):
                run = rec["run"]
                if "_core7_seed" not in run:
                    continue
                panel, seed = run.rsplit("_seed", 1)
                cname = {"control1": "control1_random",
                         "control2": "control2_fixed"}[rec["control"]]
                arm = f"{cname}_{rec['variant']}"
                runs[arm][panel][int(seed)] = {
                    "rank_ic_1": ic.get((rec["control"], run, rec["variant"])),
                    "net_pct": fnum(rec["net_pct"]),
                    "sharpe": fnum(rec["sharpe"]),
                    "mdd_pct": fnum(rec["mdd_pct"]),
                    "turnover": fnum(rec["turnover"]),
                    "cost_pct": fnum(rec["cost_pct"])}
    return runs


def per_panel_means(runs, arm):
    """{panel: {metric: mean over seeds}}"""
    out = {}
    for panel, by_seed in runs[arm].items():
        out[panel] = {m: mean([sv[m] for sv in by_seed.values()])
                      for m in METRICS}
    return out


def main():
    runs = load_runs()
    all_panels = sorted({p for a in runs.values() for p in a})

    # ------------------------------------------------------------ main table
    table_rows = []
    print("\n=== ECONOMICS TABLE: core7 panels, 2020-2024, sleeves H=10 "
          "top-10, netted TC/PRC costs ===")
    print("(cells are mean +/- SE across panels; seeds averaged within "
          "panel first)\n")
    hdr = (f"{'arm':<36} {'rankIC@1':>18} {'net% (5y)':>16} {'Sharpe':>15} "
           f"{'MDD%':>14} {'turnover':>9} {'cost%':>7}  n")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARM_ORDER + sorted(set(runs) - set(ARM_ORDER)):
        if arm not in runs:
            continue
        pp = per_panel_means(runs, arm)
        panels = sorted(pp)
        n_runs = sum(len(runs[arm][p]) for p in panels)
        rec = {"arm": arm, "label": LABELS.get(arm, arm),
               "n_panels": len(panels), "n_runs": n_runs,
               "partial": "partial" if len(panels) < len(all_panels) else ""}
        cells = {}
        for m in METRICS:
            mn, se = mean_se([pp[p][m] for p in panels])
            rec[f"{m}_mean"] = mn
            rec[f"{m}_se"] = se
            cells[m] = (mn, se)
        table_rows.append(rec)

        def c(m, prec, pm="+"):
            mn, se = cells[m]
            if mn is None:
                return "--"
            s = f"%{pm}.{prec}f" % mn
            if se is not None:
                s += " ±%.*f" % (prec, se)
            return s
        print(f"{rec['label']:<36} {c('rank_ic_1', 4):>18} "
              f"{c('net_pct', 1):>16} {c('sharpe', 2):>15} "
              f"{c('mdd_pct', 1, ''):>14} "
              f"{(('%.3f' % cells['turnover'][0]) if cells['turnover'][0] is not None else '--'):>9} "
              f"{(('%.1f' % cells['cost_pct'][0]) if cells['cost_pct'][0] is not None else '--'):>7}  "
              f"{rec['n_runs']}{'*' if rec['partial'] else ''}")
    print("\n* = partial (fewer than %d panels or incomplete seeds)"
          % len(all_panels))

    with open(os.path.join(RESULTS, "econ_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0]))
        w.writeheader()
        w.writerows(table_rows)

    # ------------------------------------------------------ appendix CSV
    with open(os.path.join(RESULTS, "econ_per_panel.csv"), "w",
              newline="") as fh:
        cols = ["arm", "panel", "n_seeds"] + METRICS
        w = csv.writer(fh)
        w.writerow(cols)
        for arm in ARM_ORDER + sorted(set(runs) - set(ARM_ORDER)):
            if arm not in runs:
                continue
            pp = per_panel_means(runs, arm)
            for panel in sorted(pp):
                w.writerow([arm, panel, len(runs[arm][panel])] +
                           [pp[panel][m] for m in METRICS])

    # ------------------------------------------------- paired comparisons
    print("\n=== PAIRED per-panel differences (n = panels aligned; seeds "
          "averaged within panel) ===")
    print("CAVEAT: n=4 panels -- these t statistics have 3 degrees of "
          "freedom; treat as descriptive, not confirmatory.\n")
    pairs = [("onenas_ensemble", "periodic_lstm_yearly"),
             ("onenas_ensemble", "gru"),
             ("onenas_ensemble", "periodic_ridge_yearly"),
             ("onenas_ensemble", "str1"),
             ("onenas_ensemble", "control2_fixed_ensemble"),
             ("onenas_ensemble", "control1_random_ensemble"),
             ("onenas_ensemble", "ew_buy_hold"),
             ("periodic_lstm_yearly", "gru")]
    for a, b in pairs:
        if a not in runs or b not in runs:
            print(f"{a} vs {b}: missing arm, skipped")
            continue
        pa, pb = per_panel_means(runs, a), per_panel_means(runs, b)
        common = sorted(set(pa) & set(pb))
        for metric in ("net_pct", "sharpe", "rank_ic_1"):
            diffs = [pa[p][metric] - pb[p][metric] for p in common
                     if pa[p][metric] is not None and pb[p][metric] is not None]
            if not diffs:
                continue
            m, se = mean_se(diffs)
            t = (m / se) if (se not in (None, 0)) else float("nan")
            prec = 4 if metric == "rank_ic_1" else 2
            print(f"{a} - {b} [{metric}]: {m:+.{prec}f} "
                  f"(SE {se if se is not None else float('nan'):.{prec}f}, "
                  f"t {t:+.2f}, n={len(diffs)} panels; "
                  + " ".join(f"{p.split('_')[0]}:{d:+.{prec}f}"
                             for p, d in zip(common, diffs)) + ")")
    print("\nwrote %s and %s" % (os.path.join(RESULTS, "econ_table.csv"),
                                 os.path.join(RESULTS, "econ_per_panel.csv")))


if __name__ == "__main__":
    main()
