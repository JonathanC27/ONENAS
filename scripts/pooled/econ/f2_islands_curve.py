#!/usr/bin/env python3
"""F2: performance vs island count figure for the paper.

Data: results_econ/islands_sweep_final.txt (2020-2024 scored span,
registered sleeves book; 10 seeds per width except 16 islands at n=6).
Left panel: pooled net % (mean +- SE). Right panel: Sharpe -- mean with a
+-1 seed-SD band and the worst seed marked, the reliability story.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

ISLANDS = [8, 16, 20, 40]
NET     = [32.8, 44.3, 42.4, 46.5]
NET_SE  = [3.0, 2.4, 1.8, 1.9]
SHARPE  = [0.71, 0.93, 0.86, 0.91]
SH_SD   = [0.171, 0.138, 0.108, 0.078]
WORST   = [0.41, 0.72, 0.68, 0.78]

BLUE = "#2a78d6"
INK2 = "#5a5c61"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.75))

ax1.errorbar(ISLANDS, NET, yerr=NET_SE, color=BLUE, lw=1.2, marker="o",
             ms=3, capsize=3)
ax1.set_xlabel("Islands")
ax1.set_ylabel("Net return 2020–24 (%)")
ax1.set_xticks(ISLANDS)
ax1.grid(axis="y", color="#ececea", lw=0.6)

ax2.fill_between(ISLANDS, [m - s for m, s in zip(SHARPE, SH_SD)],
                 [m + s for m, s in zip(SHARPE, SH_SD)],
                 color=BLUE, alpha=0.15, lw=0, label="±1 seed SD")
ax2.plot(ISLANDS, SHARPE, color=BLUE, lw=1.2, marker="o", ms=3,
         label="Mean Sharpe")
ax2.plot(ISLANDS, WORST, color=INK2, lw=0.9, ls="--", marker="s", ms=2.5,
         label="Worst seed")
ax2.set_xlabel("Islands")
ax2.set_ylabel("Sharpe, 2020–24")
ax2.set_xticks(ISLANDS)
ax2.grid(axis="y", color="#ececea", lw=0.6)
ax2.legend(frameon=False, fontsize=5.5, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "../../../paper/islands_scaling.pdf"))
fig.savefig("/Users/jonathanchang/Documents/ONENAS-paper/islands_scaling.pdf")
print("wrote islands_scaling.pdf")
