#!/usr/bin/env python3
"""Prepare a pooled stock panel for online ONE-NAS, without lookahead.

Panel construction
------------------
- Truncates all stocks in a set to the latest common start date (keeps the
  full 50-stock universe; every late starter in these sets begins by 2004-08).
- Every column that reaches the model is put on a comparable scale.  Pooled
  "robust z" normalization uses statistics computed ONLY from rows dated
  <= --stats-end (burn-in):  x' = clip((x - median) / IQR, +/-8).
- Writes numeric-only per-stock CSVs (no date/PRC/TRAN_COST) plus
  panel_dates.csv (row, date, PRC_*, TC_*, RET_raw_*) and panel_meta.json.

Column recipes
--------------
RET,     robust z of the raw simple return.  (Previously passed through raw,
sprtrn   which left the single most informative input 50-400x quieter than the
         normalized features.)  The RAW returns are still recoverable for
         scoring: panel_dates.csv carries a RET_raw_<TICKER> column per stock
         alongside PRC_/TC_, so the scorer can rebuild realized returns and the
         trading book without inverting any normalization.
ILLIQ-   abs(raw) -> trailing 21-day mean -> log -> robust z.  The upstream
UIDITY   CSVs store a SIGNED Amihud ratio RET/(VOL*PRC), so 47.5% of raw values
         are negative and the old `log1p(max(v, 0))` collapsed all of them onto
         a single constant: 48.1% of the emitted column was one value and the
         column was effectively a leaked indicator of sign(RET).  abs() recovers
         |RET|/(VOL*PRC) exactly; the 21-day trailing mean is what makes it the
         Amihud illiquidity measure rather than one day of |return| noise
         (expanding mean over the first 20 rows; trailing-only, no lookahead).
BA_      robust z, no log.  Its sd is 0.0057, i.e. entirely inside the linear
SPREAD   regime of log1p, so the old log was a no-op that only obscured units.
TURNOVER log1p -> robust z (heavy right tail, min 0).
VOL_     robust z.
CHANGE

Target columns (both always emitted, so a run can pick one via
--output_parameter_names without regenerating data)
---------------------------------------------------
RET_CS    cross-sectional rank-normal score of RET, computed per DATE across
          the N stocks of the panel:
              rank_t(i) = average rank of stock i's RET on date t (ties share
                          the mean rank), 1..N
              RET_CS(i,t) = Phi^-1((rank_t(i) - 0.5) / N)
          Phi^-1 is Acklam's rational approximation plus one Halley step
          against math.erfc (pure Python, no scipy).
RET_CS_Z  cross-sectional z-score of RET per date: (r - mean_t) / sd_t.
Both use only date-t returns, so they are contemporaneous and leak-free.

Ablation flags
--------------
--drop-sprtrn      omit sprtrn (identical across stocks on a date => zero
                   cross-sectional variance).
--add-exret        add EXRET = RET - sprtrn (robust z).
--add-rev5         add REV5 = trailing 5-day cumulative return,
                   prod(1 + r) - 1 over the last 5 days including today
                   (expanding over the first 4 rows), robust z.
--cs-rank-features replace the pooled robust z of the FEATURE columns with a
                   per-date cross-sectional rank mapped to [-1, 1]
                   (Gu-Kelly-Xiu): 2*(rank - 1)/(N - 1) - 1.
                   Applies to VOL_CHANGE, BA_SPREAD, ILLIQUIDITY, TURNOVER,
                   EXRET, REV5.  RET_CS/RET_CS_Z are targets and are untouched;
                   RET is excluded because it is itself a target column (the
                   scorer's default output parameter) and its cross-sectional
                   rank would simply duplicate RET_CS, and sprtrn is excluded
                   because it is constant across stocks on a date, so its
                   cross-sectional rank is degenerate (all ties -> 0).

Default (no flags) = the fixed baseline:
    RET, VOL_CHANGE, BA_SPREAD, ILLIQUIDITY, sprtrn, TURNOVER, RET_CS, RET_CS_Z

Usage:
    prep_panel.py <set_dir> <out_dir> [--stats-end 2019-12-31] [flags]
    prep_panel.py --selftest
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

STATS_END_DEFAULT = "2019-12-31"
CLIP = 8.0
AMIHUD_WINDOW = 21
AMIHUD_FLOOR = 1e-15  # min positive 21d mean in these sets is 7.1e-12
REV_WINDOW = 5
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------- inverse normal CDF

_PPF_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_PPF_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_PPF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
_PPF_PLOW = 0.02425


def norm_ppf(p):
    """Phi^-1(p): Acklam's rational approximation + one Halley refinement.

    Pure stdlib (math.erfc for the refinement); accurate to ~1e-15 relative
    over the range this script uses.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("norm_ppf: p must be in (0, 1), got %r" % (p,))
    a, b, c, d = _PPF_A, _PPF_B, _PPF_C, _PPF_D
    if p < _PPF_PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p > 1.0 - _PPF_PLOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    else:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    if abs(x) < 8.0:  # guard exp(x^2/2) overflow in the far tails
        e = 0.5 * math.erfc(-x / SQRT2) - p
        u = e * SQRT2PI * math.exp(x * x / 2.0)
        x -= u / (1.0 + x * u / 2.0)
    return x


# ------------------------------------------------------------------- numerics

def quantile(sorted_vals, p):
    """Linear-interpolated quantile of an already-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("quantile of empty sample")
    if n == 1:
        return sorted_vals[0]
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def robust_stats(vals):
    """(median, IQR) with a positive-definite IQR."""
    s = sorted(vals)
    return quantile(s, 0.5), max(quantile(s, 0.75) - quantile(s, 0.25), 1e-12)


def robust_z(v, med, iqr):
    return max(-CLIP, min(CLIP, (v - med) / iqr))


def trailing_mean(xs, w):
    """Trailing mean over w observations including today; expanding at the
    start (row i < w-1 averages the i+1 rows available).  No lookahead."""
    out = []
    run = 0.0
    for i, v in enumerate(xs):
        run += v
        if i >= w:
            run -= xs[i - w]
        out.append(run / min(i + 1, w))
    return out


def trailing_cumret(rets, w):
    """prod(1 + r) - 1 over the trailing w returns including today; expanding
    at the start.  No lookahead."""
    out = []
    for i in range(len(rets)):
        p = 1.0
        for j in range(max(0, i - w + 1), i + 1):
            p *= 1.0 + rets[j]
        out.append(p - 1.0)
    return out


def average_ranks(vals):
    """1-based ranks, ties sharing the mean of the ranks they span."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def cs_rank_normal(vals):
    """Cross-sectional rank-normal (van der Waerden) scores, Phi^-1((rk-.5)/N)."""
    n = len(vals)
    return [norm_ppf((r - 0.5) / n) for r in average_ranks(vals)]


def cs_rank_pm1(vals):
    """Gu-Kelly-Xiu cross-sectional rank mapped to [-1, 1]."""
    n = len(vals)
    if n < 2:
        return [0.0] * n
    return [2.0 * (r - 1.0) / (n - 1.0) - 1.0 for r in average_ranks(vals)]


def cs_zscore(vals):
    """Cross-sectional z-score, population sd; all-equal date -> zeros."""
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    if var <= 0.0:
        return [0.0] * n
    sd = math.sqrt(var)
    return [max(-CLIP, min(CLIP, (v - m) / sd)) for v in vals]


# ------------------------------------------------------------------- selftest

def selftest():
    """Unit-check the rank-normal transform and the trailing helpers."""
    ok = True

    # 1. rank-normal on a synthetic date with 50 distinct values
    n = 50
    raw = [math.sin(i * 1.7) * (1.0 + 0.01 * i) for i in range(n)]
    assert len(set(raw)) == n, "synthetic sample must be distinct"
    z = cs_rank_normal(raw)
    pairs = sorted(zip(raw, z))
    monotone = all(pairs[i][1] < pairs[i + 1][1] for i in range(n - 1))
    mean = sum(z) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in z) / n)
    zs = sorted(z)
    symm = max(abs(zs[i] + zs[n - 1 - i]) for i in range(n))
    print("rank-normal unit check (N=%d distinct values)" % n)
    print("  monotone in input : %s" % monotone)
    print("  mean              : %+.3e" % mean)
    print("  sd                : %.6f" % sd)
    print("  max |z_i + z_N+1-i| (symmetry): %.3e" % symm)
    print("  min/max           : %+.6f / %+.6f" % (zs[0], zs[-1]))
    ok &= monotone
    ok &= abs(mean) < 1e-12
    ok &= 0.90 < sd < 1.10
    ok &= symm < 1e-12

    # ties share a rank => share a score
    tied = cs_rank_normal([1.0, 2.0, 2.0, 3.0])
    ok &= abs(tied[1] - tied[2]) < 1e-15
    ok &= abs(sum(tied)) < 1e-12
    print("  ties share a score: %s" % (abs(tied[1] - tied[2]) < 1e-15))

    # 2. norm_ppf against known quantiles (both Acklam branches + refinement)
    known = [(0.5, 0.0), (0.975, 1.9599639845400545), (0.025, -1.9599639845400545),
             (0.99, 2.3263478740408408), (0.01, -2.3263478740408408),
             (0.001, -3.090232306167813), (0.8413447460685429, 1.0)]
    err = max(abs(norm_ppf(p) - x) for p, x in known)
    print("norm_ppf max abs error vs known quantiles: %.3e" % err)
    ok &= err < 1e-9
    # round-trip through erfc over a wide grid
    rt = max(abs(0.5 * math.erfc(-norm_ppf(p / 1000.0) / SQRT2) - p / 1000.0)
             for p in range(1, 1000))
    print("norm_ppf max |Phi(Phi^-1(p)) - p| over p=0.001..0.999: %.3e" % rt)
    ok &= rt < 1e-12

    # 3. cross-sectional z-score
    zz = cs_zscore([1.0, 2.0, 3.0, 4.0])
    m = sum(zz) / 4
    print("cs_zscore mean %.3e sd %.6f (degenerate date -> %s)"
          % (m, math.sqrt(sum((v - m) ** 2 for v in zz) / 4), cs_zscore([2.0] * 5)[:2]))
    ok &= abs(m) < 1e-12

    # 4. Gu-Kelly-Xiu [-1, 1] rank
    r1 = cs_rank_pm1([10.0, 20.0, 30.0])
    print("cs_rank_pm1([10,20,30]) = %s" % [round(v, 6) for v in r1])
    ok &= r1 == [-1.0, 0.0, 1.0]

    # 5. trailing helpers: expanding start, trailing-only window
    tm = trailing_mean([1.0, 2.0, 3.0, 4.0], 2)
    ok &= tm == [1.0, 1.5, 2.5, 3.5]
    tc = trailing_cumret([0.1, 0.1, 0.1], 2)
    print("trailing_mean(w=2) %s ; trailing_cumret(w=2) %s"
          % (tm, [round(v, 6) for v in tc]))
    ok &= abs(tc[0] - 0.1) < 1e-15 and abs(tc[2] - 0.21) < 1e-12
    # trailing_mean must not see the future: perturbing row k leaves rows < k
    base = trailing_mean([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    pert = list([1.0, 2.0, 3.0, 4.0, 5.0])
    pert[3] = 999.0
    ok &= trailing_mean(pert, 3)[:3] == base[:3]
    print("trailing_mean is causal (future perturbation leaves the past): True")

    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description="Prepare a pooled stock panel.")
    # set_dir/out_dir are required; --selftest is short-circuited in main()
    # before parse_args so it can run without them.
    p.add_argument("set_dir", help="directory of per-ticker CSVs")
    p.add_argument("out_dir", help="output directory")
    p.add_argument("--stats-end", default=STATS_END_DEFAULT,
                   help="last date (inclusive) usable for normalization stats")
    p.add_argument("--drop-sprtrn", action="store_true",
                   help="omit sprtrn (zero cross-sectional variance)")
    p.add_argument("--add-exret", action="store_true",
                   help="add EXRET = RET - sprtrn")
    p.add_argument("--add-rev5", action="store_true",
                   help="add REV5 = trailing 5-day cumulative return")
    p.add_argument("--cs-rank-features", action="store_true",
                   help="cross-sectional [-1,1] ranks for features instead of "
                        "pooled robust z (targets unaffected)")
    p.add_argument("--selftest", action="store_true",
                   help="run unit checks on the transforms and exit")
    return p.parse_args(argv)


def load_set(set_dir, out_dir):
    """Read per-ticker CSVs, truncate to the latest common start, align dates."""
    files = sorted(glob.glob(os.path.join(set_dir, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    if not files:
        sys.exit("no per-ticker CSVs in %s" % set_dir)
    data = {}
    for f in files:
        with open(f) as fh:
            data[os.path.basename(f)] = list(csv.DictReader(fh))
    common_start = max(rows[0]["date"] for rows in data.values())
    for name in data:
        data[name] = [r for r in data[name] if r["date"] >= common_start]
    lens = {len(v) for v in data.values()}
    assert len(lens) == 1, "row counts differ after truncation: %s" % (lens,)
    names = sorted(data)
    dates = [r["date"] for r in data[names[0]]]
    assert dates == sorted(dates), "dates are not ascending"
    for name in names:
        assert [r["date"] for r in data[name]] == dates, "date misalignment: %s" % name
    return names, dates, data, common_start


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = parse_args(argv)
    set_dir, out_dir, stats_end = args.set_dir, args.out_dir, args.stats_end
    os.makedirs(out_dir, exist_ok=True)

    names, dates, data, common_start = load_set(set_dir, out_dir)
    n_stocks, n_rows = len(names), len(dates)

    # ---- no-lookahead guard: burn-in is a strict date-ordered prefix ---------
    n_burn = sum(1 for d in dates if d <= stats_end)
    assert n_burn > 0, "no rows on or before --stats-end %s" % stats_end
    assert all(d <= stats_end for d in dates[:n_burn]), "burn-in prefix leaks"
    assert all(d > stats_end for d in dates[n_burn:]), "burn-in is not a prefix"

    # ---- per-stock series ---------------------------------------------------
    # series[col][stock_name] = list over rows; all transforms below that touch
    # the time axis are trailing-only.
    raw_ret, series = {}, {}
    for col in ("RET", "sprtrn", "VOL_CHANGE", "BA_SPREAD", "TURNOVER",
                "ILLIQUIDITY", "EXRET", "REV5"):
        series[col] = {}
    for name in names:
        rows = data[name]
        rets = [float(r["RET"]) for r in rows]
        raw_ret[name] = rets
        series["RET"][name] = list(rets)
        series["sprtrn"][name] = [float(r["sprtrn"]) for r in rows]
        series["VOL_CHANGE"][name] = [float(r["VOL_CHANGE"]) for r in rows]
        series["BA_SPREAD"][name] = [float(r["BA_SPREAD"]) for r in rows]
        series["TURNOVER"][name] = [math.log1p(max(float(r["TURNOVER"]), 0.0))
                                    for r in rows]
        # Amihud: |signed raw| == |RET|/(VOL*PRC) exactly -> trailing mean -> log
        amihud = trailing_mean([abs(float(r["ILLIQUIDITY"])) for r in rows],
                               AMIHUD_WINDOW)
        series["ILLIQUIDITY"][name] = [math.log(max(v, AMIHUD_FLOOR)) for v in amihud]
        series["EXRET"][name] = [rets[i] - series["sprtrn"][name][i]
                                 for i in range(n_rows)]
        series["REV5"][name] = trailing_cumret(rets, REV_WINDOW)
        for col in series:
            vals = series[col][name]
            assert len(vals) == n_rows, "%s/%s length mismatch" % (name, col)
            assert all(v == v and not math.isinf(v) for v in vals), \
                "non-finite value in %s/%s" % (name, col)

    # ---- column plan --------------------------------------------------------
    feature_cols = ["VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "TURNOVER"]
    if args.add_exret:
        feature_cols.append("EXRET")
    if args.add_rev5:
        feature_cols.append("REV5")
    # RET and sprtrn are always pooled-robust-z (see --cs-rank-features docs)
    pooled_cols = ["RET"] + ([] if args.drop_sprtrn else ["sprtrn"])
    if args.cs_rank_features:
        cs_cols = list(feature_cols)
    else:
        cs_cols = []
        pooled_cols += feature_cols

    cols_out = ["RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY"]
    if not args.drop_sprtrn:
        cols_out.append("sprtrn")
    cols_out.append("TURNOVER")
    if args.add_exret:
        cols_out.append("EXRET")
    if args.add_rev5:
        cols_out.append("REV5")
    cols_out += ["RET_CS", "RET_CS_Z"]

    # ---- pooled robust stats, burn-in rows only -----------------------------
    stats, used_max_date = {}, None
    for col in pooled_cols:
        vals = []
        for name in names:
            s = series[col][name]
            for i in range(n_burn):
                vals.append(s[i])
        assert len(vals) == n_stocks * n_burn
        med, iqr = robust_stats(vals)
        stats[col] = {"median": med, "iqr": iqr, "n": len(vals),
                      "source": "pooled rows <= %s" % stats_end}
        used_max_date = dates[n_burn - 1]
    assert used_max_date is None or used_max_date <= stats_end, \
        "normalization statistics used a row dated after %s" % stats_end

    # ---- transform ----------------------------------------------------------
    out = {name: [[0.0] * len(cols_out) for _ in range(n_rows)] for name in names}
    col_idx = {c: i for i, c in enumerate(cols_out)}
    for col in cols_out:
        if col in ("RET_CS", "RET_CS_Z"):
            continue
        j = col_idx[col]
        if col in stats:
            med, iqr = stats[col]["median"], stats[col]["iqr"]
            for name in names:
                s = series[col][name]
                o = out[name]
                for i in range(n_rows):
                    o[i][j] = robust_z(s[i], med, iqr)
    # cross-sectional passes: strictly same-date, one row at a time
    j_cs, j_csz = col_idx["RET_CS"], col_idx["RET_CS_Z"]
    cs_feature_idx = [col_idx[c] for c in cs_cols]
    for i in range(n_rows):
        d = dates[i]
        # every contributor to this row is dated exactly d (no other date is read)
        assert all(data[name][i]["date"] == d for name in names), \
            "cross-sectional row %d mixes dates" % i
        rv = [raw_ret[name][i] for name in names]
        for k, z in enumerate(cs_rank_normal(rv)):
            out[names[k]][i][j_cs] = z
        for k, z in enumerate(cs_zscore(rv)):
            out[names[k]][i][j_csz] = z
        for col, j in zip(cs_cols, cs_feature_idx):
            fv = [series[col][name][i] for name in names]
            for k, z in enumerate(cs_rank_pm1(fv)):
                out[names[k]][i][j] = z

    # ---- write --------------------------------------------------------------
    for name in names:
        with open(os.path.join(out_dir, name), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols_out)
            for rec in out[name]:
                w.writerow(["%.8g" % v for v in rec])

    # sidecar for the scoring/trading layer (never fed to the model).  RET_raw_*
    # keeps the un-normalized returns available now that RET itself is scaled.
    with open(os.path.join(out_dir, "panel_dates.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "date"]
                   + ["PRC_%s" % n[:-4] for n in names]
                   + ["TC_%s" % n[:-4] for n in names]
                   + ["RET_raw_%s" % n[:-4] for n in names])
        for i, d in enumerate(dates):
            w.writerow([i, d]
                       + [data[n][i]["PRC"] for n in names]
                       + [data[n][i]["TRAN_COST"] for n in names]
                       + [data[n][i]["RET"] for n in names])

    meta = {
        "set_dir": os.path.abspath(set_dir),
        "stats_end": stats_end,
        "common_start": common_start,
        "n_stocks": n_stocks,
        "n_rows": n_rows,
        "n_burn_in_rows": n_burn,
        "first_date": dates[0],
        "last_date": dates[-1],
        "last_burn_in_date": dates[n_burn - 1],
        "dates_index_anchor": {"0": dates[0], str(n_rows - 1): dates[-1]},
        "columns": cols_out,
        "target_columns": ["RET_CS", "RET_CS_Z", "RET"],
        "feature_columns": [c for c in cols_out
                            if c not in ("RET_CS", "RET_CS_Z", "RET")],
        "pooled_robust_z_columns": sorted(stats),
        "cs_rank_pm1_columns": cs_cols,
        "clip": CLIP,
        "stats": stats,
        "flags": {"drop_sprtrn": args.drop_sprtrn, "add_exret": args.add_exret,
                  "add_rev5": args.add_rev5,
                  "cs_rank_features": args.cs_rank_features},
        "raw_columns": [],
        "recipes": {
            "RET": "robust z of raw simple return (burn-in median/IQR, clip +/-8)",
            "sprtrn": "robust z of raw index return",
            "VOL_CHANGE": "robust z",
            "BA_SPREAD": "robust z, NO log (sd 0.0057 sits in log1p's linear regime)",
            "ILLIQUIDITY": ("abs(raw signed Amihud) -> trailing %d-day mean "
                            "(expanding first %d rows) -> log (floor %g) -> robust z"
                            % (AMIHUD_WINDOW, AMIHUD_WINDOW - 1, AMIHUD_FLOOR)),
            "TURNOVER": "log1p -> robust z",
            "EXRET": "RET - sprtrn -> robust z",
            "REV5": ("prod(1+r)-1 over trailing %d days incl. today "
                     "(expanding first %d rows) -> robust z" % (REV_WINDOW,
                                                                REV_WINDOW - 1)),
            "RET_CS": ("per-date cross-sectional rank-normal of raw RET: "
                       "Phi^-1((avg_rank - 0.5)/N), N = n_stocks; "
                       "contemporaneous, leak-free"),
            "RET_CS_Z": ("per-date cross-sectional z-score of raw RET: "
                         "(r - mean_t)/sd_t (population sd), clip +/-8"),
        },
        "raw_returns": {
            "where": "panel_dates.csv",
            "columns": ["RET_raw_%s" % n[:-4] for n in names],
            "note": ("RET is now normalized in the per-stock CSVs, so the raw "
                     "simple returns needed for realized-return / trading-book "
                     "scoring live in panel_dates.csv as RET_raw_<TICKER>, in "
                     "the same stock order as PRC_/TC_ (sorted ticker order). "
                     "RET can also be de-normalized as "
                     "raw = z * stats.RET.iqr + stats.RET.median, but only "
                     "where z is unclipped."),
        },
        "no_lookahead": ("all pooled normalization statistics come from rows "
                         "dated <= %s (a strict prefix of the panel, %d of %d "
                         "rows); every cross-sectional transform reads only "
                         "same-date values; every time-axis transform "
                         "(Amihud mean, REV5) is trailing-only"
                         % (stats_end, n_burn, n_rows)),
    }
    with open(os.path.join(out_dir, "panel_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("%d stocks, %d rows, %s..%s, stats<= %s (%d burn-in rows)"
          % (n_stocks, n_rows, dates[0], dates[-1], stats_end, n_burn))
    print("columns: %s" % ",".join(cols_out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
