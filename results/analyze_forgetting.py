#!/usr/bin/env python3
"""
Analyze and plot catastrophic forgetting experiment results.

Compares 6 conditions (2 architectures x 3 replay strategies):
  1. ONE-NAS + Uniform Replay
  2. ONE-NAS + Stratified Replay
  3. ONE-NAS + SlidingWindow (no replay) [control]
  4. LSTM + Uniform Replay
  5. LSTM + Stratified Replay
  6. LSTM + SlidingWindow (no replay) [control]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

results_dir = os.path.dirname(os.path.abspath(__file__))

# Load all experiments
experiments = {
    "ONE-NAS + Uniform": "catastrophic_forgetting",
    "ONE-NAS + Stratified": "catastrophic_forgetting_stratified",
    "ONE-NAS + SlidingWindow": "catastrophic_forgetting_control",
    "LSTM + Uniform": "catastrophic_forgetting_lstm_baseline",
    "LSTM + Stratified": "catastrophic_forgetting_lstm_stratified",
    "LSTM + SlidingWindow": "catastrophic_forgetting_lstm_control",
}

data = {}
for label, folder in experiments.items():
    path = os.path.join(results_dir, folder, "0", "forgetting_results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        data[label] = df
        print(f"{label}: {len(df)} rows, gens {df['generation'].min()}-{df['generation'].max()}")
    else:
        print(f"{label}: NOT FOUND at {path}")

# Phase boundary
phase_boundary = 722  # num_episodes_a(822) - num_training_sets(100)

# Colors: solid for replay methods, dashed for controls
colors = {
    "ONE-NAS + Uniform": "#2196F3",        # blue
    "ONE-NAS + Stratified": "#9C27B0",     # purple
    "ONE-NAS + SlidingWindow": "#F44336",  # red
    "LSTM + Uniform": "#4CAF50",           # green
    "LSTM + Stratified": "#FF9800",        # orange
    "LSTM + SlidingWindow": "#795548",     # brown
}
linestyles = {
    "ONE-NAS + Uniform": "-",
    "ONE-NAS + Stratified": "-",
    "ONE-NAS + SlidingWindow": "--",
    "LSTM + Uniform": "-",
    "LSTM + Stratified": "-",
    "LSTM + SlidingWindow": "--",
}

# ============================================================
# Figure 1: 4-panel overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Catastrophic Forgetting Experiment: ONE-NAS vs LSTM", fontsize=14, fontweight="bold")

# Plot 1: MSE on A (summer) - the forgetting signal
ax = axes[0, 0]
for label, df in data.items():
    ax.plot(df["generation"], df["mse_on_a"], color=colors[label],
            linestyle=linestyles[label], linewidth=1.5, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7, label="Phase boundary")
ax.set_xlabel("Generation")
ax.set_ylabel("MSE")
ax.set_title("MSE on Summer (A) -- Forgetting Metric")
ax.legend(fontsize=7)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Plot 2: MSE on B (winter) - adaptation metric
ax = axes[0, 1]
for label, df in data.items():
    ax.plot(df["generation"], df["mse_on_b"], color=colors[label],
            linestyle=linestyles[label], linewidth=1.5, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7, label="Phase boundary")
ax.set_xlabel("Generation")
ax.set_ylabel("MSE")
ax.set_title("MSE on Winter (B) -- Adaptation Metric")
ax.legend(fontsize=7)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Plot 3: MAE on A
ax = axes[1, 0]
for label, df in data.items():
    ax.plot(df["generation"], df["mae_on_a"], color=colors[label],
            linestyle=linestyles[label], linewidth=1.5, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7, label="Phase boundary")
ax.set_xlabel("Generation")
ax.set_ylabel("MAE")
ax.set_title("MAE on Summer (A)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 4: MAE on B
ax = axes[1, 1]
for label, df in data.items():
    ax.plot(df["generation"], df["mae_on_b"], color=colors[label],
            linestyle=linestyles[label], linewidth=1.5, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7, label="Phase boundary")
ax.set_xlabel("Generation")
ax.set_ylabel("MAE")
ax.set_title("MAE on Winter (B)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "forgetting_comparison_6panel.png"), dpi=200, bbox_inches="tight")
print("\nSaved: forgetting_comparison_6panel.png")

# ============================================================
# Figure 2: Focused comparison - MSE on A before/after shift
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Catastrophic Forgetting: Summer Knowledge Retention After Distribution Shift", fontsize=13, fontweight="bold")

# Left: zoomed on post-boundary (the critical region)
ax = ax1
for label, df in data.items():
    post = df[df["generation"] >= phase_boundary]
    if len(post) > 0:
        ax.plot(post["generation"], post["mse_on_a"], color=colors[label],
                linestyle=linestyles[label], linewidth=2, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7)
ax.set_xlabel("Generation")
ax.set_ylabel("MSE on Summer (A)")
ax.set_title("Post-Shift: Does Summer MSE Degrade?")
ax.legend(fontsize=7)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Right: MSE ratio (A/B) - how balanced is performance?
ax = ax2
for label, df in data.items():
    post = df[df["generation"] >= phase_boundary]
    if len(post) > 0:
        ratio = post["mse_on_a"] / post["mse_on_b"]
        ax.plot(post["generation"], ratio, color=colors[label],
                linestyle=linestyles[label], linewidth=2, label=label, alpha=0.85)
ax.axvline(x=phase_boundary, color="gray", linestyle=":", alpha=0.7)
ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
ax.set_xlabel("Generation")
ax.set_ylabel("MSE(A) / MSE(B)")
ax.set_title("Performance Balance (ratio < 1 = better on summer)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "forgetting_post_shift_analysis.png"), dpi=200, bbox_inches="tight")
print("Saved: forgetting_post_shift_analysis.png")

# ============================================================
# Summary tables
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY: Final Performance (last eval point)")
print("=" * 90)
print(f"{'Experiment':<28} {'MSE(A)':<12} {'MSE(B)':<12} {'MAE(A)':<12} {'MAE(B)':<12} {'Phase':<10}")
print("-" * 90)
for label, df in data.items():
    last = df.iloc[-1]
    print(f"{label:<28} {last['mse_on_a']:<12.6f} {last['mse_on_b']:<12.6f} {last['mae_on_a']:<12.6f} {last['mae_on_b']:<12.6f} {last['phase']:<10}")

# Forgetting metric: compare MSE(A) at boundary vs end
print("\n" + "=" * 90)
print("FORGETTING ANALYSIS: MSE on Summer (A) at boundary vs end")
print("=" * 90)
print(f"{'Experiment':<28} {'MSE(A)@boundary':<18} {'MSE(A)@end':<18} {'Change':<12} {'Forgot?'}")
print("-" * 90)
for label, df in data.items():
    boundary_rows = df[df["generation"] >= phase_boundary - 5]
    if len(boundary_rows) > 0:
        mse_at_boundary = boundary_rows.iloc[0]["mse_on_a"]
        mse_at_end = df.iloc[-1]["mse_on_a"]
        change = (mse_at_end - mse_at_boundary) / mse_at_boundary * 100
        forgot = "YES" if change > 10 else "No"
        print(f"{label:<28} {mse_at_boundary:<18.6f} {mse_at_end:<18.6f} {change:>+8.1f}%    {forgot}")
    else:
        print(f"{label:<28} {'N/A (no boundary data)'}")

print("\nDone.")
