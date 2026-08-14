#!/usr/bin/env python3
"""Run a named subset of the baseline suite over a panel and tabulate it.

Results land in  <results-dir>/<baseline>/<panel>/  as

    predictions.csv   date,stock,pred   (tidy, the scored span only)
    meta.json         model name + frozen hyperparameters + timings
    metrics.json      the full per-year/overall metric block
    book_daily.csv    daily long-short book returns
    run.log           everything the baseline printed, tuning table included

and the aggregate table is written to <results-dir>/summary_<panel>.csv.

    python3 run_all.py --panel .../set1_clean --baselines cheap
    python3 run_all.py --panel .../set1_clean --baselines all --results-dir results
    python3 run_all.py --panel .../set1_v2 --baselines paper

Group names: trivial, cheap, neural, periodic, paper, all.  Or pass a comma
separated list of individual baseline names.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> spec.  "arch_from" inherits a tuned architecture (periodic RNN reuses
# the online RNN's fixed architecture, per the spec's "same fixed LSTM");
# "config_from" freezes another baseline's whole tuned config, which is how the
# cadence sweep keeps everything except the cadence constant.
def _b(script, args, arch_from=None, config_from=None):
    return {"script": script, "args": args, "arch_from": arch_from,
            "config_from": config_from}


BASELINES = {
    "naive": _b("trivial.py", ["--model", "naive"]),
    "str1":  _b("trivial.py", ["--model", "str1"]),
    "str5":  _b("trivial.py", ["--model", "str5"]),
    "ridge": _b("online_ridge.py", []),
    "ar":    _b("online_ar.py", []),
    "lstm":  _b("online_rnn.py", ["--cell", "lstm"]),
    "gru":   _b("online_rnn.py", ["--cell", "gru"]),
}
# cadence is the experimental variable, so it is tuned once (yearly for ridge,
# quarterly for the LSTM) and the other cadences reuse that frozen config
for _m, _tuned in (("ridge", "yearly"), ("lstm", "quarterly")):
    for _cad in ("yearly", "quarterly", "monthly"):
        BASELINES[f"periodic_{_m}_{_cad}"] = _b(
            "periodic_retrain.py",
            ["--model", _m, "--cadence", _cad],
            arch_from="lstm" if _m == "lstm" else None,
            config_from=(None if _cad == _tuned else f"periodic_{_m}_{_tuned}"))

GROUPS = {
    "trivial": ["naive", "str1", "str5"],
    "cheap": ["naive", "str1", "str5", "ridge", "ar"],
    "neural": ["lstm", "gru"],
    "periodic": ["periodic_ridge_yearly", "periodic_ridge_quarterly",
                 "periodic_ridge_monthly", "periodic_lstm_quarterly",
                 "periodic_lstm_yearly", "periodic_lstm_monthly"],
}
GROUPS["paper"] = (GROUPS["cheap"] + GROUPS["neural"]
                   + ["periodic_ridge_yearly", "periodic_ridge_quarterly",
                      "periodic_ridge_monthly", "periodic_lstm_quarterly"])
GROUPS["all"] = (GROUPS["cheap"] + GROUPS["neural"] + GROUPS["periodic"])

COLUMNS = ["pearson_ic", "pearson_t", "rank_ic_1", "rank_ic_5", "rank_ic_10",
           "net_pct", "sharpe", "mdd_pct", "turnover", "cost_pct"]


def resolve(names):
    out = []
    for tok in names.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in GROUPS:
            out.extend(GROUPS[tok])
        elif tok in BASELINES:
            out.append(tok)
        else:
            raise SystemExit(
                f"unknown baseline/group {tok!r}; known baselines: "
                f"{', '.join(sorted(BASELINES))}; groups: {', '.join(sorted(GROUPS))}")
    # pull in dependencies (a cadence variant needs the run it inherits from)
    queue, out2 = list(out), []
    while queue:
        n = queue.pop(0)
        spec = BASELINES[n]
        for dep in (spec["arch_from"], spec["config_from"]):
            if dep and dep not in out2 and dep not in queue:
                out2.append(dep)
        out2.append(n)
    seen, uniq = set(), []
    for n in out2:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    # dependencies must run first
    order = {n: i for i, n in enumerate(uniq)}

    def depth(n, guard=()):
        if n in guard:
            raise SystemExit(f"cyclic baseline dependency at {n}")
        spec = BASELINES[n]
        deps = [d for d in (spec["arch_from"], spec["config_from"]) if d]
        return 0 if not deps else 1 + max(depth(d, guard + (n,)) for d in deps)

    return sorted(uniq, key=lambda n: (depth(n), order[n]))


def fmt(x, spec="{:+.4f}"):
    if x is None:
        return "   nan"
    return spec.format(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--baselines", default="paper")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    ap.add_argument("--no-tune", action="store_true",
                    help="use each baseline's frozen default instead of tuning")
    ap.add_argument("--extra", default="",
                    help="extra argv appended to every baseline, e.g. "
                         "\"--cost-bps 5\"")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-run a baseline that already has metrics.json")
    ap.add_argument("--table-only", action="store_true",
                    help="only rebuild the summary table from existing results")
    args = ap.parse_args()

    panel_name = os.path.basename(os.path.abspath(args.panel).rstrip("/"))
    names = resolve(args.baselines)
    print(f"# panel {panel_name}: {len(names)} baselines -> {', '.join(names)}")

    def meta_path(n):
        return os.path.join(args.results_dir, n, panel_name, "meta.json")

    timings = {}
    for name in names:
        spec = BASELINES[name]
        out_dir = os.path.join(args.results_dir, name, panel_name)
        done = os.path.exists(os.path.join(out_dir, "metrics.json"))
        if args.table_only or (args.skip_existing and done):
            print(f"# {name}: reusing {out_dir}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, spec["script"]),
               "--panel", args.panel, "--out-dir", out_dir,
               "--seed", str(args.seed),
               "--score-from", args.score_from, "--score-to", args.score_to]
        cmd += spec["args"]
        if spec["arch_from"] and os.path.exists(meta_path(spec["arch_from"])):
            cmd += ["--arch-from", meta_path(spec["arch_from"])]
        if spec["config_from"]:
            src = meta_path(spec["config_from"])
            if not os.path.exists(src):
                print(f"#   {name}: SKIPPED, needs {spec['config_from']} first")
                continue
            with open(src) as fh:
                hp = json.load(fh).get("hyperparameters", {})
            cmd += ["--config", json.dumps(hp)]
        if args.no_tune:
            cmd.append("--no-tune")
        if args.extra:
            cmd += args.extra.split()
        t0 = time.time()
        print(f"# running {name}: {' '.join(cmd[1:])}", flush=True)
        log_path = os.path.join(out_dir, "run.log")
        with open(log_path, "w") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               cwd=HERE)
        dt = time.time() - t0
        timings[name] = dt
        status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
        print(f"#   {name}: {status} in {dt:.1f}s (log {log_path})")
        if p.returncode != 0:
            with open(log_path) as fh:
                tail = fh.read().strip().splitlines()[-15:]
            print("\n".join("    " + t for t in tail))

    # ------------------------------------------------------------- aggregate
    rows = []
    for name in names:
        mpath = os.path.join(args.results_dir, name, panel_name, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as fh:
            m = json.load(fh)
        ov = m["summary"]["overall"]["model"]
        meta = m.get("meta", {})
        rec = {"baseline": name,
               "n_days": ov.get("n_days"),
               "hyperparameters": json.dumps(meta.get("hyperparameters", {}),
                                             sort_keys=True),
               "fit_seconds": meta.get("fit_seconds"),
               "total_seconds": meta.get("total_seconds"),
               "wall_seconds": round(timings.get(name, float("nan")), 1)
                               if name in timings else None}
        for c in COLUMNS:
            rec[c] = ov.get(c)
        rows.append(rec)
        if not any(r["baseline"] == "(naive reference)" for r in rows):
            nv = m["summary"]["overall"]["naive"]
            ref = {"baseline": "(naive reference)", "n_days": nv.get("n_days"),
                   "hyperparameters": "", "fit_seconds": None,
                   "total_seconds": None, "wall_seconds": None}
            ref.update({c: nv.get(c) for c in COLUMNS})
            rows.insert(0, ref)

    if not rows:
        raise SystemExit("no results to tabulate")

    out_csv = os.path.join(args.results_dir, f"summary_{panel_name}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"# {panel_name}: {args.score_from}..{args.score_to}, "
          f"long-short top/bottom-10, TC/PRC costs")
    hdr = (f"{'baseline':<26}{'days':>5} {'pearsIC':>9}{'t':>7} {'rIC@1':>9}"
           f"{'rIC@5':>9}{'rIC@10':>9} {'net%':>9}{'sharpe':>8}{'MDD%':>8}"
           f"{'turnov':>8}{'cost%':>8}{'secs':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['baseline']:<26}{r['n_days'] or 0:>5} "
              f"{fmt(r['pearson_ic'])} {fmt(r['pearson_t'], '{:+.2f}'):>6} "
              f"{fmt(r['rank_ic_1']):>9}{fmt(r['rank_ic_5']):>9}"
              f"{fmt(r['rank_ic_10']):>9} "
              f"{fmt(r['net_pct'], '{:+.2f}'):>9}{fmt(r['sharpe'], '{:+.2f}'):>8}"
              f"{fmt(r['mdd_pct'], '{:.2f}'):>8}{fmt(r['turnover'], '{:.4f}'):>8}"
              f"{fmt(r['cost_pct'], '{:.2f}'):>8}"
              f"{fmt(r['wall_seconds'], '{:.0f}'):>8}")
    print()
    print(f"# wrote {out_csv}")


if __name__ == "__main__":
    main()
