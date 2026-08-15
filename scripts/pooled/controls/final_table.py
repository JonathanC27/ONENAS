#!/usr/bin/env python3
"""Assemble the paper-facing controls table: v2 and core7 side by side.

Merges results_controls/summary.csv (rank IC, from scoring.py) with
results_controls/sleeves_summary.csv (net%/Sharpe under the sleeves H=10
book, from sleeves_book.py) into one grid: per control, per panel-set, per
seed, the single-champion and 8-member-ensemble rank IC@1 plus the
ensemble's sleeves net% and Sharpe, with per-(control,family,variant) means
and across-run sd.

    python3 final_table.py [--results-dir results_controls] [--md out.md]

Run `run_controls.py --table-only` and `sleeves_book.py --panels ...` first
if results changed on disk.
"""

import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if not os.path.exists(path):
        raise SystemExit("missing %s (run run_controls.py --table-only / "
                         "sleeves_book.py first)" % path)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def family_of(run):
    """set3_v2_seed44 -> ('v2', 'set3', 44)."""
    pname, seed = run.rsplit("_seed", 1)
    setname, fam = pname.split("_", 1)
    return fam, setname, int(seed)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir",
                    default=os.path.join(HERE, "results_controls"))
    ap.add_argument("--md", default=None,
                    help="also write a markdown table here")
    args = ap.parse_args()

    ics = load(os.path.join(args.results_dir, "summary.csv"))
    slv = load(os.path.join(args.results_dir, "sleeves_summary.csv"))
    S = {(r["control"], r["run"], r["variant"]): r for r in slv}

    # cell[(control, family, setname, seed)] = {variant: (ic, net, sharpe)}
    cell = defaultdict(dict)
    for r in ics:
        fam, setname, seed = family_of(r["run"])
        s = S.get((r["control"], r["run"], r["variant"]), {})
        cell[(r["control"], fam, setname, seed)][r["variant"]] = (
            float(r["rank_ic_1"]),
            float(s["net_pct"]) if s else float("nan"),
            float(s["sharpe"]) if s else float("nan"))

    fams = sorted({k[1] for k in cell})
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    hdr = ("| control | set | seed | " + " | ".join(
        "{0} single rIC | {0} ens rIC | {0} ens net% | {0} ens Sharpe"
        .format(f) for f in fams) + " |")
    emit(hdr)
    emit("|" + "---|" * (3 + 4 * len(fams)))

    controls = sorted({k[0] for k in cell})
    sets = sorted({k[2] for k in cell})
    seeds = sorted({k[3] for k in cell})
    agg = defaultdict(list)     # (control, fam, variant, metric) -> values
    for c in controls:
        for st in sets:
            for seed in seeds:
                row = ["", "", ""]
                row[0], row[1], row[2] = c, st, str(seed)
                cols = []
                any_data = False
                for f in fams:
                    v = cell.get((c, f, st, seed))
                    if not v or "single" not in v or "ensemble" not in v:
                        cols += ["--", "--", "--", "--"]
                        continue
                    any_data = True
                    sic = v["single"][0]
                    eic, enet, esh = v["ensemble"]
                    agg[(c, f, "single_ic")].append(sic)
                    agg[(c, f, "ens_ic")].append(eic)
                    agg[(c, f, "ens_net")].append(enet)
                    agg[(c, f, "ens_sharpe")].append(esh)
                    cols += ["%+.4f" % sic, "%+.4f" % eic,
                             "%+.1f" % enet, "%+.2f" % esh]
                if any_data:
                    emit("| " + " | ".join(row + cols) + " |")
    emit("")
    emit("| control | family | single rIC mean±sd | ens rIC mean±sd | "
         "ens net% mean | ens Sharpe mean | n runs |")
    emit("|---|---|---|---|---|---|---|")
    for c in controls:
        for f in fams:
            si = agg.get((c, f, "single_ic"))
            if not si:
                continue
            ei = agg[(c, f, "ens_ic")]
            emit("| %s | %s | %+.4f ± %.4f | %+.4f ± %.4f | %+.1f | %+.2f "
                 "| %d |"
                 % (c, f, mean(si), sd(si), mean(ei), sd(ei),
                    mean(agg[(c, f, "ens_net")]),
                    mean(agg[(c, f, "ens_sharpe")]), len(si)))
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\nwrote %s" % args.md)


if __name__ == "__main__":
    main()
