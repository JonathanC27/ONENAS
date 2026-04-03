#!/usr/bin/env python3
"""Generate paper-style visualizations for the catastrophic forgetting experiment."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Read fitness_log.csv ──
generations = []
best_val_mse = []
island_best = {i: [] for i in range(6)}

with open('fitness_log.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row_idx, row in enumerate(reader):
        generations.append(row_idx)
        best_val_mse.append(float(row[4]))  # Best Val. MSE
        for i in range(6):
            col = 8 + i * 2  # Island_i_best_fitness
            val = float(row[col])
            island_best[i].append(val)

generations = np.array(generations)
best_val_mse = np.array(best_val_mse)

BOUNDARY = 734

# ── Read forgetting_results.csv ──
fg_gen = []
fg_phase = []
fg_mse_a = []
fg_mae_a = []
fg_mse_b = []
fg_mae_b = []
fg_val_mse = []

with open('forgetting_results.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        fg_gen.append(int(row[0]))
        fg_phase.append(row[1])
        fg_mse_a.append(float(row[2]))
        fg_mae_a.append(float(row[3]))
        fg_mse_b.append(float(row[4]))
        fg_mae_b.append(float(row[5]))
        fg_val_mse.append(float(row[6]))

fg_gen = np.array(fg_gen)
fg_mse_a = np.array(fg_mse_a)
fg_mse_b = np.array(fg_mse_b)
fg_val_mse = np.array(fg_val_mse)

# ── Compute cumulative average MSE (paper style) ──
cumulative_mse = np.cumsum(best_val_mse) / np.arange(1, len(best_val_mse) + 1)

# ── Compute moving average MSE (window=50) ──
window = 50
moving_avg = np.convolve(best_val_mse, np.ones(window)/window, mode='valid')
moving_avg_gens = generations[window-1:]

# ── Figure setup ──
fig = plt.figure(figsize=(16, 20))
fig.suptitle('Catastrophic Forgetting Experiment - Trial 0\n(Paper-Style Presentation)',
             fontsize=16, fontweight='bold', y=0.98)

gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3,
              left=0.08, right=0.95, top=0.93, bottom=0.04)

# ════════════════════════════════════════════════════════
# Panel 1: Cumulative Average Validation MSE (like Figs 12-14)
# ════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(generations, cumulative_mse, color='#2166ac', linewidth=2, label='ONE-NAS Cumulative Avg MSE')
ax1.axvline(x=BOUNDARY, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Distribution Shift (Gen 734)')
ax1.set_xlabel('Generation', fontsize=11)
ax1.set_ylabel('Cumulative Average MSE', fontsize=11)
ax1.set_title('Cumulative Average Validation MSE Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim(0, len(generations))
ax1.grid(True, alpha=0.3)

# Add annotation for the trend
pre_final = cumulative_mse[BOUNDARY]
post_final = cumulative_mse[-1]
ax1.annotate(f'At boundary: {pre_final:.5f}', xy=(BOUNDARY, pre_final),
            xytext=(BOUNDARY+100, pre_final+0.0005),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')
ax1.annotate(f'Final: {post_final:.5f}', xy=(len(generations)-1, post_final),
            xytext=(len(generations)-300, post_final+0.0005),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

# ════════════════════════════════════════════════════════
# Panel 2: Moving Average MSE (smoothed view of per-gen MSE)
# ════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(generations, best_val_mse, color='#bdd7e7', linewidth=0.5, alpha=0.5, label='Per-generation MSE (raw)')
ax2.plot(moving_avg_gens, moving_avg, color='#2166ac', linewidth=2, label=f'Moving Average (window={window})')
ax2.axvline(x=BOUNDARY, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Distribution Shift')
ax2.set_xlabel('Generation', fontsize=11)
ax2.set_ylabel('Validation MSE', fontsize=11)
ax2.set_title(f'Validation MSE with {window}-Generation Moving Average', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim(0, len(generations))
ax2.grid(True, alpha=0.3)

# ════════════════════════════════════════════════════════
# Panel 3: Binned Average MSE (like Fig 10 style)
# ════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2, 0])

bin_size = 125
bin_labels = []
bin_means = []
bin_colors = []

for start in range(0, len(best_val_mse), bin_size):
    end = min(start + bin_size, len(best_val_mse))
    bin_labels.append(f'{start}-{end}')
    bin_means.append(np.mean(best_val_mse[start:end]))
    # Color: blue for pre-boundary, red-ish for bins that span/follow boundary
    if end <= BOUNDARY:
        bin_colors.append('#2166ac')
    elif start >= BOUNDARY:
        bin_colors.append('#b2182b')
    else:
        bin_colors.append('#f4a582')  # mixed bin

bars = ax3.bar(range(len(bin_labels)), bin_means, color=bin_colors, edgecolor='white', linewidth=0.5)
ax3.set_xticks(range(len(bin_labels)))
ax3.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Mean Validation MSE', fontsize=11)
ax3.set_title('Binned Average MSE by Generation Range', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2166ac', label='Pure A (pre-shift)'),
                   Patch(facecolor='#f4a582', label='Transition bin'),
                   Patch(facecolor='#b2182b', label='Mixed A+B (post-shift)')]
ax3.legend(handles=legend_elements, fontsize=9, loc='upper right')

# ════════════════════════════════════════════════════════
# Panel 4: Box plot of per-island best fitness, pre vs post (like Fig 9)
# ════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[2, 1])

# Collect per-island final MSE for pre-boundary and post-boundary
pre_island_means = []
post_island_means = []
for i in range(6):
    arr = np.array(island_best[i])
    pre_island_means.append(np.mean(arr[:BOUNDARY]))
    post_island_means.append(np.mean(arr[BOUNDARY:]))

positions_pre = [1, 2, 3, 4, 5, 6]
positions_post = [8, 9, 10, 11, 12, 13]

ax4.bar(positions_pre, pre_island_means, color='#2166ac', alpha=0.8, label='Pre-shift (Pure A)')
ax4.bar(positions_post, post_island_means, color='#b2182b', alpha=0.8, label='Post-shift (Mixed A+B)')
ax4.set_xticks([3.5, 10.5])
ax4.set_xticklabels(['Pre-Shift', 'Post-Shift'], fontsize=11)
ax4.set_ylabel('Mean Island Best Fitness (MSE)', fontsize=11)
ax4.set_title('Per-Island Average Fitness', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Add island labels
for i, p in enumerate(positions_pre):
    ax4.text(p, pre_island_means[i] + 0.0001, f'I{i}', ha='center', fontsize=7, color='gray')
for i, p in enumerate(positions_post):
    ax4.text(p, post_island_means[i] + 0.0001, f'I{i}', ha='center', fontsize=7, color='gray')

# ════════════════════════════════════════════════════════
# Panel 5: Eval MSE on Distribution A (fixed holdout - THE forgetting metric)
# ════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[3, 0])

baseline_a = fg_mse_a[0]  # Pre-shift baseline
ax5.plot(fg_gen, fg_mse_a, color='#2166ac', linewidth=2, marker='o', markersize=2, label='MSE on Eval A')
ax5.axhline(y=baseline_a, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Pre-shift baseline ({baseline_a:.6f})')
ax5.axvline(x=BOUNDARY, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax5.fill_between(fg_gen, baseline_a, fg_mse_a,
                 where=fg_mse_a > baseline_a,
                 color='#fee0d2', alpha=0.6, label='Forgetting episodes')
ax5.fill_between(fg_gen, baseline_a, fg_mse_a,
                 where=fg_mse_a < baseline_a,
                 color='#deebf7', alpha=0.6, label='Improvement over baseline')
ax5.set_xlabel('Generation', fontsize=11)
ax5.set_ylabel('MSE on Distribution A (Summer)', fontsize=11)
ax5.set_title('Catastrophic Forgetting: Eval on Distribution A', fontsize=13, fontweight='bold')
ax5.legend(fontsize=8, loc='upper right')
ax5.grid(True, alpha=0.3)

# ════════════════════════════════════════════════════════
# Panel 6: Eval MSE on Distribution B (fixed holdout - adaptation metric)
# ════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[3, 1])

baseline_b = fg_mse_b[0]
ax6.plot(fg_gen, fg_mse_b, color='#b2182b', linewidth=2, marker='o', markersize=2, label='MSE on Eval B')
ax6.axhline(y=baseline_b, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Pre-shift baseline ({baseline_b:.4f})')
ax6.axvline(x=BOUNDARY, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax6.fill_between(fg_gen, baseline_b, fg_mse_b,
                 where=fg_mse_b < baseline_b,
                 color='#deebf7', alpha=0.6, label='Improvement over baseline')
ax6.set_xlabel('Generation', fontsize=11)
ax6.set_ylabel('MSE on Distribution B (Winter)', fontsize=11)
ax6.set_title('Adaptation: Eval on Distribution B', fontsize=13, fontweight='bold')
ax6.legend(fontsize=8, loc='upper right')
ax6.grid(True, alpha=0.3)

plt.savefig('forgetting_paper_style.png', dpi=150, bbox_inches='tight')
print("Saved forgetting_paper_style.png")

# ════════════════════════════════════════════════════════
# SECOND FIGURE: Summary statistics table-style plot
# ════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Trial 0 Summary: Pre-Shift vs Post-Shift Performance',
              fontsize=14, fontweight='bold', y=1.02)

# Left: Dual-axis eval plot (A and B together, smoothed)
ax_left = axes2[0]
ax_left.plot(fg_gen, fg_mse_a * 1000, color='#2166ac', linewidth=2, label='MSE on A (×1000)')
ax_right = ax_left.twinx()
ax_right.plot(fg_gen, fg_mse_b * 1000, color='#b2182b', linewidth=2, label='MSE on B (×1000)')
ax_left.axvline(x=BOUNDARY, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax_left.set_xlabel('Generation', fontsize=11)
ax_left.set_ylabel('MSE on A (×1000)', fontsize=11, color='#2166ac')
ax_right.set_ylabel('MSE on B (×1000)', fontsize=11, color='#b2182b')
ax_left.set_title('Distribution A vs B Performance', fontsize=12, fontweight='bold')
ax_left.tick_params(axis='y', labelcolor='#2166ac')
ax_right.tick_params(axis='y', labelcolor='#b2182b')

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
ax_left.grid(True, alpha=0.3)

# Right: Summary bar chart
ax_bar = axes2[1]
categories = ['MSE on A\n(Summer)', 'MSE on B\n(Winter)', 'Best Val\nMSE']
pre_values = [fg_mse_a[0], fg_mse_b[0], fg_val_mse[0]]
post_values = [fg_mse_a[-1], fg_mse_b[-1], fg_val_mse[-1]]

x = np.arange(len(categories))
width = 0.35
bars1 = ax_bar.bar(x - width/2, pre_values, width, color='#2166ac', alpha=0.8, label='Pre-shift (Gen 733)')
bars2 = ax_bar.bar(x + width/2, post_values, width, color='#b2182b', alpha=0.8, label='Post-shift (Gen 1249)')

ax_bar.set_ylabel('MSE', fontsize=11)
ax_bar.set_title('Pre-Shift vs Post-Shift Metrics', fontsize=12, fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(categories, fontsize=10)
ax_bar.legend(fontsize=10)
ax_bar.grid(True, alpha=0.3, axis='y')

# Add percentage change labels
for i, (pre, post) in enumerate(zip(pre_values, post_values)):
    pct = (post - pre) / pre * 100
    color = '#2ca02c' if pct < 0 else '#d62728'
    symbol = '' if pct < 0 else '+'
    ax_bar.text(x[i] + width/2, post + 0.0002, f'{symbol}{pct:.1f}%',
               ha='center', fontsize=9, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('forgetting_summary.png', dpi=150, bbox_inches='tight')
print("Saved forgetting_summary.png")
