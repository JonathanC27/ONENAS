#!/usr/bin/env python3
"""Score pooled ONE-NAS per-generation prediction files with cross-sectional rank IC.

Reads generation_N_global_best.csv files (pooled format: expected_<p>_s<i>,
naive_<p>_s<i>, global_best_predicted_<p>_s<i> column groups), computes the
Spearman rank correlation across stocks between predicted and realized values
at each timestep, and reports per-generation mean IC for both the model and
the naive persistence baseline. No dependencies beyond the standard library.

Usage: score_ic.py <output_dir> [--param RET]
"""
import csv, glob, math, os, re, sys

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

def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / len(ra); mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return None
    return num / (da * db)

def score_file(path, param):
    with open(path) as fh:
        header = fh.readline().lstrip("#").strip().split(",")
        rows = [list(map(float, r)) for r in csv.reader(fh) if r]
    pat = re.compile(rf"^(expected|naive|global_best_predicted)_{re.escape(param)}_s(\d+)$")
    cols = {}
    for idx, name in enumerate(header):
        m = pat.match(name)
        if m:
            cols.setdefault(int(m.group(2)), {})[m.group(1)] = idx
    stocks = sorted(cols)
    if len(stocks) < 3:
        sys.exit(f"{path}: only {len(stocks)} stocks found — not a pooled prediction file?")
    ics, ics_naive = [], []
    for row in rows:
        exp = [row[cols[s]["expected"]] for s in stocks]
        pred = [row[cols[s]["global_best_predicted"]] for s in stocks]
        nai = [row[cols[s]["naive"]] for s in stocks]
        ic = spearman(pred, exp)
        icn = spearman(nai, exp)
        if ic is not None: ics.append(ic)
        if icn is not None: ics_naive.append(icn)
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return mean(ics), mean(ics_naive), len(rows)

def main():
    outdir = sys.argv[1]
    param = sys.argv[sys.argv.index("--param") + 1] if "--param" in sys.argv else "RET"
    files = glob.glob(os.path.join(outdir, "generation_*_global_best.csv"))
    files.sort(key=lambda p: int(re.search(r"generation_(\d+)_", p).group(1)))
    if not files:
        sys.exit(f"no generation_*_global_best.csv in {outdir}")
    print("generation,mean_ic,naive_ic,n_timesteps")
    all_ic, all_naive = [], []
    for p in files:
        g = int(re.search(r"generation_(\d+)_", p).group(1))
        ic, icn, n = score_file(p, param)
        all_ic.append(ic); all_naive.append(icn)
        print(f"{g},{ic:.6f},{icn:.6f},{n}")
    m = sum(all_ic) / len(all_ic); mn = sum(all_naive) / len(all_naive)
    se = (sum((x - m) ** 2 for x in all_ic) / max(len(all_ic) - 1, 1)) ** 0.5 / len(all_ic) ** 0.5
    print(f"# overall: model IC {m:+.6f} (SE {se:.6f}, {len(all_ic)} gens), naive IC {mn:+.6f}", file=sys.stderr)

if __name__ == "__main__":
    main()
