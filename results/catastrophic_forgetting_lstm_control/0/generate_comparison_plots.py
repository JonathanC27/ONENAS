#!/usr/bin/env python3
"""Generate comparison plots: LSTM SlidingWindow (control) vs ONE-NAS Uniform (replay)."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Read ONE-NAS Uniform Replay results ──
onenas_dir = '/Users/jonathanchang/ONENAS/results/catastrophic_forgetting/0'

onenas_fg_gen = []
onenas_fg_mse_a = []
onenas_fg_mae_a = []
onenas_fg_mse_b = []
onenas_fg_mae_b = []
onenas_fg_val_mse = []

with open(f'{onenas_dir}/forgetting_results.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        onenas_fg_gen.append(int(row[0]))
        onenas_fg_mse_a.append(float(row[2]))
        onenas_fg_mae_a.append(float(row[3]))
        onenas_fg_mse_b.append(float(row[4]))
        onenas_fg_mae_b.append(float(row[5]))
        onenas_fg_val_mse.append(float(row[6]))

onenas_fg_gen = np.array(onenas_fg_gen)
onenas_fg_mse_a = np.array(onenas_fg_mse_a)
onenas_fg_mae_a = np.array(onenas_fg_mae_a)
onenas_fg_mse_b = np.array(onenas_fg_mse_b)
onenas_fg_mae_b = np.array(onenas_fg_mae_b)

# ── Read LSTM SlidingWindow Control results ──
lstm_fg_gen = []
lstm_fg_mse_a = []
lstm_fg_mae_a = []
lstm_fg_mse_b = []
lstm_fg_mae_b = []
lstm_fg_val_mse = []

with open('forgetting_results.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        lstm_fg_gen.append(int(row[0]))
        lstm_fg_mse_a.append(float(row[2]))
        lstm_fg_mae_a.append(float(row[3]))
        lstm_fg_mse_b.append(float(row[4]))
        lstm_fg_mae_b.append(float(row[5]))
        lstm_fg_val_mse.append(float(row[6]))

lstm_fg_gen = np.array(lstm_fg_gen)
lstm_fg_mse_a = np.array(lstm_fg_mse_a)
lstm_fg_mae_a = np.array(lstm_fg_mae_a)
lstm_fg_mse_b = np.array(lstm_fg_mse_b)
lstm_fg_mae_b = np.array(lstm_fg_mae_b)

BOUNDARY = 734

# ── Colors ──
ONENAS_COLOR = '#2166ac'  # Blue
LSTM_COLOR = '#d62728'    # Red
ONENAS_LIGHT = '#92c5de'
LSTM_LIGHT = '#f4a582'

# ════════════════════════════════════════════════════════════
# FIGURE 1: Main comparison (paper-style, 2x2 grid)
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Catastrophic Forgetting: ONE-NAS (Uniform Replay) vs LSTM (SlidingWindow, No Replay)',
             fontsize=15, fontweight='bold', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
              left=0.08, right=0.95, top=0.92, bottom=0.06)

# ── Panel 1: MSE on Distribution A (forgetting metric) ──
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(onenas_fg_gen, onenas_fg_mse_a, color=ONENAS_COLOR, linewidth=2,
         marker='o', markersize=2, label='ONE-NAS (Uniform Replay)', zorder=3)
ax1.plot(lstm_fg_gen, lstm_fg_mse_a, color=LSTM_COLOR, linewidth=2,
         marker='s', markersize=2, label='LSTM (SlidingWindow, No Replay)', zorder=3)

baseline_a_onenas = onenas_fg_mse_a[0]
baseline_a_lstm = lstm_fg_mse_a[0]
ax1.axhline(y=baseline_a_onenas, color=ONENAS_LIGHT, linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'ONE-NAS pre-shift baseline ({baseline_a_onenas:.6f})')
ax1.axhline(y=baseline_a_lstm, color=LSTM_LIGHT, linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'LSTM pre-shift baseline ({baseline_a_lstm:.6f})')
ax1.axvline(x=BOUNDARY, color='gray', linestyle='--', linewidth=1.5, alpha=0.6,
            label='Distribution Shift (Gen 734)')

ax1.set_xlabel('Generation', fontsize=12)
ax1.set_ylabel('MSE on Distribution A (Summer)', fontsize=12)
ax1.set_title('Forgetting: Held-Out Summer Evaluation', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# ── Panel 2: MSE on Distribution B (adaptation metric) ──
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(onenas_fg_gen, onenas_fg_mse_b, color=ONENAS_COLOR, linewidth=2,
         marker='o', markersize=2, label='ONE-NAS (Uniform Replay)')
ax2.plot(lstm_fg_gen, lstm_fg_mse_b, color=LSTM_COLOR, linewidth=2,
         marker='s', markersize=2, label='LSTM (SlidingWindow)')

baseline_b_onenas = onenas_fg_mse_b[0]
baseline_b_lstm = lstm_fg_mse_b[0]
ax2.axhline(y=baseline_b_onenas, color=ONENAS_LIGHT, linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(y=baseline_b_lstm, color=LSTM_LIGHT, linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axvline(x=BOUNDARY, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

ax2.set_xlabel('Generation', fontsize=12)
ax2.set_ylabel('MSE on Distribution B (Winter)', fontsize=12)
ax2.set_title('Adaptation: Held-Out Winter Evaluation', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

# ── Panel 3: Summary bar chart (pre-shift vs final) ──
ax3 = fig.add_subplot(gs[1, 1])

categories = ['MSE on A\n(Summer)', 'MSE on B\n(Winter)']
x = np.arange(len(categories))
width = 0.18

# Pre-shift values
onenas_pre = [onenas_fg_mse_a[0], onenas_fg_mse_b[0]]
onenas_post = [onenas_fg_mse_a[-1], onenas_fg_mse_b[-1]]
lstm_pre = [lstm_fg_mse_a[0], lstm_fg_mse_b[0]]
lstm_post = [lstm_fg_mse_a[-1], lstm_fg_mse_b[-1]]

# Peak forgetting for LSTM on A
lstm_peak_a = np.max(lstm_fg_mse_a)
onenas_peak_a = np.max(onenas_fg_mse_a)

bars1 = ax3.bar(x - 1.5*width, onenas_pre, width, color=ONENAS_COLOR, alpha=0.5, label='ONE-NAS Pre-shift')
bars2 = ax3.bar(x - 0.5*width, onenas_post, width, color=ONENAS_COLOR, alpha=1.0, label='ONE-NAS Final')
bars3 = ax3.bar(x + 0.5*width, lstm_pre, width, color=LSTM_COLOR, alpha=0.5, label='LSTM Pre-shift')
bars4 = ax3.bar(x + 1.5*width, lstm_post, width, color=LSTM_COLOR, alpha=1.0, label='LSTM Final')

ax3.set_ylabel('MSE', fontsize=12)
ax3.set_title('Pre-Shift vs Final Performance', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=11)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')

# Add percentage change labels for final bars
for i, (pre_o, post_o) in enumerate(zip(onenas_pre, onenas_post)):
    pct = (post_o - pre_o) / pre_o * 100
    color = '#2ca02c' if pct < 0 else '#d62728'
    symbol = '' if pct < 0 else '+'
    ax3.text(x[i] - 0.5*width, post_o + 0.0002, f'{symbol}{pct:.1f}%',
             ha='center', fontsize=8, fontweight='bold', color=color)

for i, (pre_l, post_l) in enumerate(zip(lstm_pre, lstm_post)):
    pct = (post_l - pre_l) / pre_l * 100
    color = '#2ca02c' if pct < 0 else '#d62728'
    symbol = '' if pct < 0 else '+'
    ax3.text(x[i] + 1.5*width, post_l + 0.0002, f'{symbol}{pct:.1f}%',
             ha='center', fontsize=8, fontweight='bold', color=color)

plt.savefig('comparison_forgetting.png', dpi=150, bbox_inches='tight')
print("Saved comparison_forgetting.png")

# ════════════════════════════════════════════════════════════
# FIGURE 2: Forgetting ratio over time
# ════════════════════════════════════════════════════════════
fig2, (ax_ratio, ax_mae) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Catastrophic Forgetting: Detailed Comparison',
              fontsize=14, fontweight='bold', y=1.02)

# Left: Forgetting ratio (MSE_A_current / MSE_A_baseline)
onenas_ratio = onenas_fg_mse_a / onenas_fg_mse_a[0]
lstm_ratio = lstm_fg_mse_a / lstm_fg_mse_a[0]

ax_ratio.plot(onenas_fg_gen, onenas_ratio, color=ONENAS_COLOR, linewidth=2,
              marker='o', markersize=2, label='ONE-NAS (Uniform Replay)')
ax_ratio.plot(lstm_fg_gen, lstm_ratio, color=LSTM_COLOR, linewidth=2,
              marker='s', markersize=2, label='LSTM (SlidingWindow)')
ax_ratio.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='No forgetting (ratio = 1)')
ax_ratio.axvline(x=BOUNDARY, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

ax_ratio.set_xlabel('Generation', fontsize=12)
ax_ratio.set_ylabel('MSE Ratio (current / pre-shift baseline)', fontsize=12)
ax_ratio.set_title('Forgetting Ratio on Distribution A', fontsize=13, fontweight='bold')
ax_ratio.legend(fontsize=9)
ax_ratio.grid(True, alpha=0.3)

# Add peak annotations
lstm_peak_idx = np.argmax(lstm_ratio)
onenas_peak_idx = np.argmax(onenas_ratio)
ax_ratio.annotate(f'LSTM peak: {lstm_ratio[lstm_peak_idx]:.1f}x\n(gen {lstm_fg_gen[lstm_peak_idx]})',
                  xy=(lstm_fg_gen[lstm_peak_idx], lstm_ratio[lstm_peak_idx]),
                  xytext=(lstm_fg_gen[lstm_peak_idx]+50, lstm_ratio[lstm_peak_idx]+0.5),
                  arrowprops=dict(arrowstyle='->', color=LSTM_COLOR), fontsize=9, color=LSTM_COLOR)
ax_ratio.annotate(f'ONE-NAS peak: {onenas_ratio[onenas_peak_idx]:.1f}x\n(gen {onenas_fg_gen[onenas_peak_idx]})',
                  xy=(onenas_fg_gen[onenas_peak_idx], onenas_ratio[onenas_peak_idx]),
                  xytext=(onenas_fg_gen[onenas_peak_idx]+50, onenas_ratio[onenas_peak_idx]+0.5),
                  arrowprops=dict(arrowstyle='->', color=ONENAS_COLOR), fontsize=9, color=ONENAS_COLOR)

# Right: MAE comparison (both A and B)
ax_mae.plot(onenas_fg_gen, onenas_fg_mae_a, color=ONENAS_COLOR, linewidth=2,
            marker='o', markersize=2, label='ONE-NAS MAE on A')
ax_mae.plot(lstm_fg_gen, lstm_fg_mae_a, color=LSTM_COLOR, linewidth=2,
            marker='s', markersize=2, label='LSTM MAE on A')
ax_mae.plot(onenas_fg_gen, onenas_fg_mae_b, color=ONENAS_COLOR, linewidth=2, linestyle='--',
            marker='o', markersize=2, alpha=0.6, label='ONE-NAS MAE on B')
ax_mae.plot(lstm_fg_gen, lstm_fg_mae_b, color=LSTM_COLOR, linewidth=2, linestyle='--',
            marker='s', markersize=2, alpha=0.6, label='LSTM MAE on B')
ax_mae.axvline(x=BOUNDARY, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

ax_mae.set_xlabel('Generation', fontsize=12)
ax_mae.set_ylabel('MAE', fontsize=12)
ax_mae.set_title('MAE on Both Distributions', fontsize=13, fontweight='bold')
ax_mae.legend(fontsize=8, loc='upper right')
ax_mae.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_detailed.png', dpi=150, bbox_inches='tight')
print("Saved comparison_detailed.png")

# ════════════════════════════════════════════════════════════
# Print summary statistics
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\n{'Metric':<35} {'ONE-NAS (Replay)':<20} {'LSTM (No Replay)':<20}")
print("-"*75)
print(f"{'MSE on A (pre-shift)':<35} {onenas_fg_mse_a[0]:<20.6f} {lstm_fg_mse_a[0]:<20.6f}")
print(f"{'MSE on A (peak forgetting)':<35} {np.max(onenas_fg_mse_a):<20.6f} {np.max(lstm_fg_mse_a):<20.6f}")
print(f"{'MSE on A (final)':<35} {onenas_fg_mse_a[-1]:<20.6f} {lstm_fg_mse_a[-1]:<20.6f}")
print(f"{'Peak forgetting ratio (A)':<35} {np.max(onenas_fg_mse_a)/onenas_fg_mse_a[0]:<20.1f} {np.max(lstm_fg_mse_a)/lstm_fg_mse_a[0]:<20.1f}")
print(f"{'Final forgetting ratio (A)':<35} {onenas_fg_mse_a[-1]/onenas_fg_mse_a[0]:<20.1f} {lstm_fg_mse_a[-1]/lstm_fg_mse_a[0]:<20.1f}")
print(f"{'MSE on B (pre-shift)':<35} {onenas_fg_mse_b[0]:<20.6f} {lstm_fg_mse_b[0]:<20.6f}")
print(f"{'MSE on B (final)':<35} {onenas_fg_mse_b[-1]:<20.6f} {lstm_fg_mse_b[-1]:<20.6f}")
print(f"{'B improvement %':<35} {(onenas_fg_mse_b[-1]-onenas_fg_mse_b[0])/onenas_fg_mse_b[0]*100:<19.1f}% {(lstm_fg_mse_b[-1]-lstm_fg_mse_b[0])/lstm_fg_mse_b[0]*100:<19.1f}%")
