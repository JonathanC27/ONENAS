#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard for pooled ONE-NAS runs.

Reads a runs.json registry: [{name, run_dir, sidecar_dir, ntw, total_gen,
step, length, vsets, note, status?}] and emits one HTML file. Scored metrics
come from score_stream.py (invoked for runs with >= 3 generations).

Usage: gen_dashboard.py <runs.json> <out.html>
"""
import csv, glob, json, math, os, re, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

def gen_files(run_dir):
    fs = glob.glob(os.path.join(run_dir, "generation_*_global_best.csv"))
    fs.sort(key=lambda p: int(re.search(r"generation_(\d+)_", p).group(1)))
    return fs

def run_state(r):
    fs = gen_files(r["run_dir"])
    n = len(fs)
    if n == 0:
        return ("queued" if not os.path.isdir(r["run_dir"]) or n == 0 else "queued"), n, None, None
    done = n >= r["total_gen"]
    fresh = time.time() - os.path.getmtime(fs[-1]) < 900
    status = "done" if done else ("running" if fresh else "stalled")
    # pace from last up-to-20 generation mtimes
    recent = fs[-20:]
    pace = eta = None
    if len(recent) >= 3:
        dt = os.path.getmtime(recent[-1]) - os.path.getmtime(recent[0])
        if dt > 0:
            pace = dt / (len(recent) - 1)  # sec per generation
            if not done:
                eta = pace * (r["total_gen"] - n)
    return status, n, pace, eta

def fitness_curve(run_dir):
    path = os.path.join(run_dir, "fitness_log.csv")
    pts = []
    if os.path.exists(path):
        with open(path) as fh:
            rd = csv.reader(fh)
            hdr = next(rd, None)
            for row in rd:
                try: pts.append((float(row[0]), float(row[4])))
                except (ValueError, IndexError): pass
    if len(pts) > 400:
        k = len(pts) // 400 + 1
        pts = pts[::k]
    return pts

def score(r, n):
    if n < 3: return None, []
    cmd = [sys.executable, os.path.join(HERE, "score_stream.py"),
           "--run-dir", r["run_dir"], "--sidecar-dir", r["sidecar_dir"],
           "--step", str(r["step"]), "--length", str(r["length"]),
           "--num-training-windows", str(r["ntw"]), "--validation-sets", str(r["vsets"]),
           "--score-from", "1900-01-01", "--emit-json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
        summ = json.loads(out[out.index("{"):]) if "{" in out else None
    except Exception:
        summ = None
    roll = []
    rp = os.path.join(r["run_dir"], "rolling_rank_ic.csv")
    if os.path.exists(rp):
        with open(rp) as fh:
            for row in csv.DictReader(fh):
                try: roll.append((row["date"], float(row["model_rank_ic_63d"]), float(row["naive_rank_ic_63d"])))
                except ValueError: pass
    return summ, roll

def svg_line(series, w=560, h=170, ylabel="", zero=True, date_x=False):
    """series: [(label, colorvar, [(x,y),...])]. Returns svg string."""
    allpts = [p for _, _, pts in series for p in pts]
    if len(allpts) < 2: return "<div class='empty'>not enough data yet</div>"
    xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if zero: y0, y1 = min(y0, 0), max(y1, 0)
    pad = (y1 - y0) * 0.08 or 1e-9; y0 -= pad; y1 += pad
    L, R, T, B = 46, 8, 8, 20
    def X(v): return L + (v - x0) / (x1 - x0 or 1) * (w - L - R)
    def Y(v): return T + (y1 - v) / (y1 - y0 or 1) * (h - T - B)
    out = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='{ylabel}'>"]
    for i in range(4):
        gy = y0 + (y1 - y0) * i / 3
        out.append(f"<line x1='{L}' y1='{Y(gy):.1f}' x2='{w-R}' y2='{Y(gy):.1f}' class='grid'/>")
        out.append(f"<text x='{L-5}' y='{Y(gy)+3:.1f}' class='tick' text-anchor='end'>{gy:.4g}</text>")
    if zero and y0 < 0 < y1:
        out.append(f"<line x1='{L}' y1='{Y(0):.1f}' x2='{w-R}' y2='{Y(0):.1f}' class='zero'/>")
    for lbl, cv, pts in series:
        d = " ".join(f"{X(px):.1f},{Y(py):.1f}" for px, py in pts)
        out.append(f"<polyline points='{d}' fill='none' style='stroke:var({cv})' stroke-width='2' stroke-linejoin='round'/>")
        lx, ly = pts[-1]
        out.append(f"<text x='{min(X(lx)+4, w-2):.1f}' y='{Y(ly)+3:.1f}' class='slabel' style='fill:var({cv})'>{lbl}</text>")
        step = max(1, len(pts)//60)
        for px, py in pts[::step]:
            xv = datetime.fromtimestamp(px).strftime("%Y-%m-%d") if date_x else f"{px:.0f}"
            out.append(f"<circle cx='{X(px):.1f}' cy='{Y(py):.1f}' r='7' class='hit'><title>{lbl} {xv}: {py:.5g}</title></circle>")
    for frac in (0, 0.5, 1):
        xv = x0 + (x1 - x0) * frac
        xt = datetime.fromtimestamp(xv).strftime("%Y-%m") if date_x else f"{xv:.0f}"
        anch = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        out.append(f"<text x='{X(xv):.1f}' y='{h-6}' class='tick' text-anchor='{anch}'>{xt}</text>")
    out.append("</svg>")
    return "".join(out)

def fmt_dur(s):
    if s is None: return "—"
    if s < 3600: return f"{s/60:.0f} min"
    return f"{s/3600:.1f} h"

def main():
    reg_path, out_path = sys.argv[1], sys.argv[2]
    runs = json.load(open(reg_path))
    cards = []
    for r in runs:
        status, n, pace, eta = run_state(r)
        if r.get("status") == "queued" and n == 0: status = "queued"
        summ, roll = (None, []) if status == "queued" else score(r, n)
        fit = [] if status == "queued" else fitness_curve(r["run_dir"])
        pct = 100.0 * n / r["total_gen"]
        pill = {"running": ("running", "good"), "done": ("done", "good"),
                "stalled": ("stalled", "critical"), "queued": ("queued", "muted")}[status]
        tiles = f"""
        <div class='tiles'>
          <div class='tile'><div class='tv'>{n} / {r['total_gen']}</div><div class='tl'>generations ({pct:.0f}%)</div></div>
          <div class='tile'><div class='tv'>{fmt_dur(pace) if pace else '—'}</div><div class='tl'>per generation</div></div>
          <div class='tile'><div class='tv'>{fmt_dur(eta)}</div><div class='tl'>ETA</div></div>"""
        if summ and "overall" in summ:
            o = summ["overall"]["model"]
            tiles += f"""
          <div class='tile'><div class='tv'>{o['rank_ic_1']:+.4f}</div><div class='tl'>rank IC (interim)</div></div>
          <div class='tile'><div class='tv'>{o['net_pct']:+.1f}%</div><div class='tl'>book net (interim)</div></div>"""
        tiles += "</div>"
        fitsvg = svg_line([("best val MSE", "--s1", fit)], ylabel="best validation MSE", zero=False) if fit else "<div class='empty'>no fitness log yet</div>"
        if roll:
            rpts_m = [(datetime.strptime(d, "%Y-%m-%d").timestamp(), v) for d, v, _ in roll]
            rpts_n = [(datetime.strptime(d, "%Y-%m-%d").timestamp(), v) for d, _, v in roll]
            rollsvg = svg_line([("model", "--s1", rpts_m), ("naive", "--s2", rpts_n)], ylabel="rolling 63d rank IC", date_x=True)
            legend = "<div class='legend'><span><i style='background:var(--s1)'></i>model</span><span><i style='background:var(--s2)'></i>naive</span></div>"
        else:
            rollsvg, legend = "<div class='empty'>rolling IC needs ≥63 scored days</div>", ""
        prog = f"<div class='bar'><div class='fill' style='width:{pct:.1f}%'></div></div>"
        cards.append(f"""
      <section class='card'>
        <div class='chead'><h2>{r['name']}</h2><span class='pill {pill[1]}'>{pill[0]}</span></div>
        <p class='note'>{r.get('note','')}</p>
        {prog}{tiles}
        <div class='charts'>
          <figure><figcaption>Best validation MSE by evaluated genomes</figcaption>{fitsvg}</figure>
          <figure><figcaption>Rolling 63-day rank IC {legend}</figcaption>{rollsvg}</figure>
        </div>
      </section>""")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<title>ONE-NAS Run Monitor</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --card:#ffffff; --line:#e4e3df;
  --s1:#2a78d6; --s2:#eb6834; --good:#0ca30c; --critical:#d03b3b; --muted:#8a8880; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --card:#222221; --line:#3a3a38; --s1:#3987e5; --s2:#d95926; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --card:#222221; --line:#3a3a38; --s1:#3987e5; --s2:#d95926; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,'Helvetica Neue',sans-serif; margin:0; padding:2rem 1rem 4rem; }}
.page {{ max-width:64rem; margin:0 auto; }}
h1 {{ font-size:1.4rem; margin:0 0 .2rem; }}
.sub {{ color:var(--ink2); font-size:.85rem; margin:0 0 1.6rem; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1rem 1.2rem; margin-bottom:1.2rem; }}
.chead {{ display:flex; align-items:baseline; gap:.8rem; }} .chead h2 {{ font-size:1.05rem; margin:0; }}
.pill {{ font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; padding:.15rem .55rem; border-radius:999px; color:#fff; }}
.pill.good {{ background:var(--good); }} .pill.critical {{ background:var(--critical); }} .pill.muted {{ background:var(--muted); }}
.note {{ color:var(--ink2); font-size:.82rem; margin:.3rem 0 .7rem; }}
.bar {{ height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin-bottom:.9rem; }}
.fill {{ height:100%; background:var(--s1); }}
.tiles {{ display:flex; gap:1.4rem; flex-wrap:wrap; margin-bottom:1rem; }}
.tv {{ font-size:1.25rem; font-weight:650; font-variant-numeric:tabular-nums; }}
.tl {{ font-size:.72rem; color:var(--ink2); text-transform:uppercase; letter-spacing:.05em; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1.2rem; }}
figure {{ margin:0; }} figcaption {{ font-size:.78rem; color:var(--ink2); margin-bottom:.3rem; display:flex; gap:.8rem; align-items:center; }}
.chart {{ width:100%; height:auto; }}
.grid {{ stroke:var(--line); stroke-width:1; }} .zero {{ stroke:var(--ink2); stroke-width:1; stroke-dasharray:3 3; }}
.tick {{ font-size:9px; fill:var(--ink2); }} .slabel {{ font-size:10px; font-weight:600; }}
.hit {{ fill:transparent; }} .hit:hover {{ fill:var(--ink2); fill-opacity:.35; }}
.legend span {{ display:inline-flex; align-items:center; gap:.3rem; font-size:.75rem; color:var(--ink2); }}
.legend i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
.empty {{ color:var(--ink2); font-size:.85rem; padding:2rem 0; text-align:center; border:1px dashed var(--line); border-radius:6px; }}
</style>
<div class='page'>
  <h1>ONE-NAS Run Monitor</h1>
  <p class='sub'>Pooled online neuroevolution runs · snapshot generated {stamp} · refreshes when Claude checks runs (ask to refresh anytime). Interim IC/net include the unscored warm-up period.</p>
  {''.join(cards)}
</div>"""
    with open(out_path, "w") as fh: fh.write(html)
    print(f"wrote {out_path} ({len(cards)} runs)")

if __name__ == "__main__":
    main()
