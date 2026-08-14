#!/usr/bin/env python3
"""Prepare a pooled stock panel for online ONE-NAS, without lookahead.

- Truncates all stocks in a set to the latest common start date (keeps the
  full 50-stock universe; every late starter in these sets begins by 2004-08).
- RET and sprtrn are left raw (already scale-comparable returns; raw targets
  keep prediction files directly usable for IC / long-short scoring).
- Other features are robust-normalized with statistics computed ONLY from
  rows dated <= --stats-end (burn-in period): x' = clip((x - median)/IQR, +/-8).
  ILLIQUIDITY, TURNOVER, BA_SPREAD are log1p'd first (heavy right tails).
- Writes numeric-only CSVs (no date/PRC/TRAN_COST) plus panel_meta.json with
  the stats, common start, and row<->date mapping endpoints for scoring.

Usage: prep_panel.py <set_dir> <out_dir> [--stats-end 2019-12-31]
"""
import csv, json, math, os, sys, glob

def main():
    set_dir, out_dir = sys.argv[1], sys.argv[2]
    stats_end = sys.argv[sys.argv.index("--stats-end") + 1] if "--stats-end" in sys.argv else "2019-12-31"
    os.makedirs(out_dir, exist_ok=True)
    RAW = ["RET", "sprtrn"]
    LOG = {"ILLIQUIDITY", "TURNOVER", "BA_SPREAD"}
    NORM = ["VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "TURNOVER"]
    files = sorted(glob.glob(os.path.join(set_dir, "*.csv")))
    data = {}
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        data[os.path.basename(f)] = rows
    common_start = max(rows[0]["date"] for rows in data.values())
    for name in data:
        data[name] = [r for r in data[name] if r["date"] >= common_start]
    lens = {len(v) for v in data.values()}
    assert len(lens) == 1, f"row counts differ after truncation: {lens}"
    dates = [r["date"] for r in next(iter(data.values()))]
    for name, rows in data.items():
        assert [r["date"] for r in rows] == dates, f"date misalignment: {name}"
    # burn-in-only robust stats, pooled across stocks
    stats = {}
    for col in NORM:
        vals = []
        for rows in data.values():
            for r in rows:
                if r["date"] > stats_end: break
                v = float(r[col])
                if col in LOG: v = math.log1p(max(v, 0.0))
                if not math.isnan(v): vals.append(v)
        vals.sort()
        q = lambda p: vals[min(int(p * len(vals)), len(vals) - 1)]
        med, iqr = q(0.5), max(q(0.75) - q(0.25), 1e-12)
        stats[col] = {"median": med, "iqr": iqr, "log1p": col in LOG}
    cols_out = ["RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER"]
    for name, rows in data.items():
        with open(os.path.join(out_dir, name), "w", newline="") as out:
            w = csv.writer(out)
            w.writerow(cols_out)
            for r in rows:
                rec = []
                for col in cols_out:
                    v = float(r[col])
                    if col in stats:
                        s = stats[col]
                        if s["log1p"]: v = math.log1p(max(v, 0.0))
                        v = max(-8.0, min(8.0, (v - s["median"]) / s["iqr"]))
                    rec.append(f"{v:.8g}")
                w.writerow(rec)
    meta = {"set_dir": set_dir, "stats_end": stats_end, "common_start": common_start,
            "n_stocks": len(data), "n_rows": len(dates), "first_date": dates[0],
            "last_date": dates[-1], "dates_index_anchor": {"0": dates[0], str(len(dates)-1): dates[-1]},
            "stats": stats, "raw_columns": RAW, "columns": cols_out}
    with open(os.path.join(out_dir, "panel_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    # sidecar with dates + PRC + TRAN_COST for the scoring/trading layer (never fed to the model)
    with open(os.path.join(out_dir, "panel_dates.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "date"] + [f"PRC_{n[:-4]}" for n in sorted(data)] + [f"TC_{n[:-4]}" for n in sorted(data)])
        names = sorted(data)
        for i, d in enumerate(dates):
            w.writerow([i, d] + [data[n][i]["PRC"] for n in names] + [data[n][i]["TRAN_COST"] for n in names])
    print(f"{len(data)} stocks, {len(dates)} rows, {dates[0]}..{dates[-1]}, stats<= {stats_end}")

if __name__ == "__main__":
    main()
