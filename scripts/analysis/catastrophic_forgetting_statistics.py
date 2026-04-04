#!/usr/bin/env python3
"""
Catastrophic Forgetting: GECCO Paper Statistical Analysis

Computes rigorous continual learning metrics across 5 trials for up to 6 conditions:
  1. ONE-NAS + Uniform Replay
  2. ONE-NAS + SlidingWindow (no replay)
  3. LSTM + Uniform Replay
  4. LSTM + SlidingWindow (no replay)
  5. ONE-NAS + Stratified Replay (if available)
  6. LSTM + Stratified Replay (if available)

Produces:
  - Figures 1-5 (PNG + PDF) with mean +/- std error bands
  - LaTeX summary table
  - Mann-Whitney U tests with rank-biserial effect sizes
  - Bootstrap 95% CIs for all scalar metrics
  - Bonferroni-corrected p-values

Usage:
  python3 scripts/analysis/catastrophic_forgetting_statistics.py
"""

import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..', 'results')
OUT_DIR = os.path.join(BASE, 'statistics')
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 5
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
LSTM_PARAMS = 5831  # analytical: 4*(22*22 + 22*22 + 22) + 22*1 + 1 = 5831 for 22 inputs, 22 hidden, 1 output

# ── Condition definitions ────────────────────────────────────────────────────
CONDITIONS = {
    'ONE-NAS + Uniform':     'catastrophic_forgetting',
    'ONE-NAS + SlidingWindow': 'catastrophic_forgetting_control',
    'LSTM + Uniform':        'catastrophic_forgetting_lstm_baseline',
    'LSTM + SlidingWindow':  'catastrophic_forgetting_lstm_control',
    'ONE-NAS + Stratified':  'catastrophic_forgetting_stratified',
    'LSTM + Stratified':     'catastrophic_forgetting_lstm_stratified',
}

CORE_CONDITIONS = [
    'ONE-NAS + Uniform',
    'ONE-NAS + SlidingWindow',
    'LSTM + Uniform',
    'LSTM + SlidingWindow',
]

COLORS = {
    'ONE-NAS + Uniform':       '#2196F3',
    'ONE-NAS + SlidingWindow': '#FF9800',
    'LSTM + Uniform':          '#4CAF50',
    'LSTM + SlidingWindow':    '#F44336',
    'ONE-NAS + Stratified':    '#9C27B0',
    'LSTM + Stratified':       '#795548',
}

LINESTYLES = {
    'ONE-NAS + Uniform':       '-',
    'ONE-NAS + SlidingWindow': '-',
    'LSTM + Uniform':          '--',
    'LSTM + SlidingWindow':    '--',
    'ONE-NAS + Stratified':    '-.',
    'LSTM + Stratified':       '-.',
}

MARKERS = {
    'ONE-NAS + Uniform':       'o',
    'ONE-NAS + SlidingWindow': 's',
    'LSTM + Uniform':          '^',
    'LSTM + SlidingWindow':    'v',
    'ONE-NAS + Stratified':    'D',
    'LSTM + Stratified':       'P',
}

# ── Matplotlib defaults ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_trials():
    """Load forgetting_results.csv for all available trials across all conditions."""
    data = {}  # {condition_name: [df_trial0, df_trial1, ...]}

    for cond_name, dir_name in CONDITIONS.items():
        trials = []
        for t in range(N_TRIALS):
            path = os.path.join(BASE, dir_name, str(t), 'forgetting_results.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df[df['phase'] != 'boundary'].copy()
                df = df.sort_values('generation').reset_index(drop=True)
                trials.append(df)

        if trials:
            data[cond_name] = trials
            if len(trials) < N_TRIALS:
                print(f"  WARNING: {cond_name} has only {len(trials)}/{N_TRIALS} trials")
        # silently skip conditions with 0 trials

    return data


def detect_phase_boundary(df):
    """Return the generation of the last pure_A row (phase transition point)."""
    pure_a = df[df['phase'] == 'pure_A']
    if len(pure_a) == 0:
        return None
    return pure_a['generation'].max()


# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION (per trial)
# ══════════════════════════════════════════════════════════════════════════════

def compute_trial_metrics(df):
    """Compute all scalar metrics for a single trial DataFrame."""
    boundary_gen = detect_phase_boundary(df)
    if boundary_gen is None:
        return None

    pure_a = df[df['phase'] == 'pure_A']
    mixed = df[df['phase'] == 'mixed_AB']

    if len(mixed) == 0:
        return None

    mse_a_at_transition = pure_a.iloc[-1]['mse_on_a']
    mse_b_at_transition = pure_a.iloc[-1]['mse_on_b']
    final_mse_a = mixed.iloc[-1]['mse_on_a']
    final_mse_b = mixed.iloc[-1]['mse_on_b']

    # BWT: negative = forgetting
    bwt = final_mse_a - mse_a_at_transition

    # AUFC: area under forgetting curve (trapezoidal) normalized by generation span
    mixed_gens = mixed['generation'].values
    mixed_mse_a = mixed['mse_on_a'].values
    gen_span = mixed_gens[-1] - mixed_gens[0]
    if gen_span > 0:
        aufc = np.trapezoid(mixed_mse_a, mixed_gens) / gen_span
    else:
        aufc = np.nan

    # Peak forgetting ratio
    peak_mse_a = mixed['mse_on_a'].max()
    peak_ratio = peak_mse_a / mse_a_at_transition if mse_a_at_transition > 0 else np.nan

    # Recovery time: generations from peak until MSE_A returns to <= 1.1 * transition value
    threshold = 1.1 * mse_a_at_transition
    peak_idx = mixed['mse_on_a'].idxmax()
    peak_gen = mixed.loc[peak_idx, 'generation']
    after_peak = mixed[mixed['generation'] >= peak_gen]
    recovered = after_peak[after_peak['mse_on_a'] <= threshold]
    if len(recovered) > 0:
        recovery_time = recovered.iloc[0]['generation'] - peak_gen
    else:
        recovery_time = np.nan

    # Plasticity-Stability Ratio
    improvement_b = mse_b_at_transition - final_mse_b  # positive = improved
    degradation_a = final_mse_a - mse_a_at_transition  # positive = degraded
    if degradation_a > 0:
        ps_ratio = improvement_b / degradation_a
    else:
        ps_ratio = np.inf if improvement_b > 0 else np.nan

    return {
        'boundary_gen': boundary_gen,
        'mse_a_at_transition': mse_a_at_transition,
        'mse_b_at_transition': mse_b_at_transition,
        'final_mse_a': final_mse_a,
        'final_mse_b': final_mse_b,
        'bwt': bwt,
        'aufc': aufc,
        'peak_ratio': peak_ratio,
        'recovery_time': recovery_time,
        'ps_ratio': ps_ratio,
    }


def compute_all_metrics(data):
    """Compute per-trial metrics for all conditions. Returns {cond: [dict, ...]}."""
    all_metrics = {}
    for cond_name, trials in data.items():
        cond_metrics = []
        for df in trials:
            m = compute_trial_metrics(df)
            if m is not None:
                cond_metrics.append(m)
        if cond_metrics:
            all_metrics[cond_name] = cond_metrics
    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=95, seed=BOOTSTRAP_SEED):
    """Bootstrap confidence interval for the mean."""
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) < 2:
        m = np.nanmean(values) if len(values) == 1 else np.nan
        return m, np.nan, np.nan
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return np.mean(values), lo, hi


def rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def pairwise_mannwhitney(all_metrics, metric_key):
    """Run pairwise Mann-Whitney U tests across conditions for a given metric."""
    cond_names = sorted(all_metrics.keys())
    pairs = list(itertools.combinations(cond_names, 2))
    n_comparisons = len(pairs)

    results = []
    for c1, c2 in pairs:
        vals1 = [m[metric_key] for m in all_metrics[c1] if np.isfinite(m[metric_key])]
        vals2 = [m[metric_key] for m in all_metrics[c2] if np.isfinite(m[metric_key])]

        if len(vals1) < 2 or len(vals2) < 2:
            results.append({
                'comparison': f'{c1} vs {c2}',
                'U': np.nan, 'p_raw': np.nan, 'p_corrected': np.nan,
                'r_rb': np.nan, 'n1': len(vals1), 'n2': len(vals2),
            })
            continue

        u, p = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
        p_corr = min(p * n_comparisons, 1.0)  # Bonferroni
        r = rank_biserial(u, len(vals1), len(vals2))

        results.append({
            'comparison': f'{c1} vs {c2}',
            'U': u, 'p_raw': p, 'p_corrected': p_corr,
            'r_rb': r, 'n1': len(vals1), 'n2': len(vals2),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY AGGREGATION (for plots with error bands)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_trajectories(trials, column):
    """Aggregate a column across trials onto a common generation grid.

    Returns (generations, mean, std, se, n_valid) arrays.
    Uses NaN-tolerant aggregation for trials of different lengths.
    """
    # Collect all unique generations
    all_gens = sorted(set().union(*(df['generation'].values for df in trials)))
    all_gens = np.array(all_gens)

    # Build matrix (n_gens x n_trials), NaN where trial doesn't have that gen
    matrix = np.full((len(all_gens), len(trials)), np.nan)
    for t_idx, df in enumerate(trials):
        gen_to_val = dict(zip(df['generation'], df[column]))
        for g_idx, g in enumerate(all_gens):
            if g in gen_to_val:
                matrix[g_idx, t_idx] = gen_to_val[g]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=1)
        std = np.nanstd(matrix, axis=1, ddof=1)
        n_valid = np.sum(~np.isnan(matrix), axis=1)
        se = std / np.sqrt(n_valid)

    return all_gens, mean, std, se, n_valid


def get_mixed_phase_trajectories(trials, column):
    """Like aggregate_trajectories but restricted to mixed_AB phase."""
    mixed_trials = [df[df['phase'] == 'mixed_AB'] for df in trials]
    mixed_trials = [df for df in mixed_trials if len(df) > 0]
    if not mixed_trials:
        return None, None, None, None, None
    return aggregate_trajectories(mixed_trials, column)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def add_phase_line(ax, data):
    """Add phase boundary line using the median boundary across all trials."""
    boundaries = []
    for trials in data.values():
        for df in trials:
            b = detect_phase_boundary(df)
            if b is not None:
                boundaries.append(b)
    if boundaries:
        boundary = np.median(boundaries)
        ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ypos = ax.get_ylim()[1] * 0.95
        ax.text(boundary - 20, ypos, 'Phase 1\n(pure A)', ha='right', va='top',
                fontsize=9, color='gray', style='italic')
        ax.text(boundary + 20, ypos, 'Phase 2\n(mixed A+B)', ha='left', va='top',
                fontsize=9, color='gray', style='italic')


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'))
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'))
    print(f"  Saved: {name}.png / .pdf")


def plot_trajectory(data, column, ylabel, title, filename):
    """Fig 1/2 style: single trajectory plot with error bands."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for cond_name, trials in data.items():
        gens, mean, std, se, n_valid = aggregate_trajectories(trials, column)
        ax.plot(gens, mean, color=COLORS[cond_name], linestyle=LINESTYLES[cond_name],
                linewidth=1.8, label=cond_name, alpha=0.9)
        ax.fill_between(gens, mean - se, mean + se,
                         color=COLORS[cond_name], alpha=0.15)

    add_phase_line(ax, data)
    ax.set_xlabel('Generation')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


def plot_forgetting_ratio(data, filename):
    """Fig 3: forgetting ratio during mixed phase."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for cond_name, trials in data.items():
        # Compute forgetting ratio per trial, then aggregate
        ratio_trials = []
        for df in trials:
            boundary = detect_phase_boundary(df)
            if boundary is None:
                continue
            mixed = df[df['phase'] == 'mixed_AB'].copy()
            if len(mixed) == 0:
                continue
            baseline_a = mixed.iloc[0]['mse_on_a']
            if baseline_a <= 0:
                continue
            mixed = mixed.copy()
            mixed['forgetting_ratio'] = mixed['mse_on_a'] / baseline_a
            ratio_trials.append(mixed[['generation', 'forgetting_ratio']])

        if not ratio_trials:
            continue

        gens, mean, std, se, n = aggregate_trajectories(
            [rt.rename(columns={'forgetting_ratio': 'val'}).assign(phase='mixed_AB')
             for rt in ratio_trials],
            'val'
        )
        # Workaround: aggregate_trajectories expects a column name in df
        # Just do it manually here
        all_gens = sorted(set().union(*(rt['generation'].values for rt in ratio_trials)))
        all_gens = np.array(all_gens)
        matrix = np.full((len(all_gens), len(ratio_trials)), np.nan)
        for t_idx, rt in enumerate(ratio_trials):
            gen_to_val = dict(zip(rt['generation'], rt['forgetting_ratio']))
            for g_idx, g in enumerate(all_gens):
                if g in gen_to_val:
                    matrix[g_idx, t_idx] = gen_to_val[g]

        mean = np.nanmean(matrix, axis=1)
        se = np.nanstd(matrix, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(matrix), axis=1))

        ax.plot(all_gens, mean, color=COLORS[cond_name], linestyle=LINESTYLES[cond_name],
                linewidth=1.8, label=cond_name, alpha=0.9)
        ax.fill_between(all_gens, mean - se, mean + se,
                         color=COLORS[cond_name], alpha=0.15)

    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Forgetting Ratio (MSE_A / MSE_A at transition)')
    ax.set_title('Forgetting Ratio on Distribution A During Mixed Training')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


def plot_bar_chart(all_metrics, filename):
    """Fig 4: bar chart of BWT, AUFC, Peak Ratio with bootstrap CIs + significance."""
    metrics_to_plot = ['bwt', 'aufc', 'peak_ratio']
    metric_labels = {
        'bwt': 'BWT (Backward Transfer)',
        'aufc': 'AUFC (Avg MSE_A\nduring mixed phase)',
        'peak_ratio': 'Peak Forgetting Ratio',
    }

    cond_names = [c for c in CONDITIONS if c in all_metrics]
    n_conds = len(cond_names)
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax_idx, metric_key in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        means = []
        ci_los = []
        ci_his = []
        colors_list = []

        for cond in cond_names:
            vals = [m[metric_key] for m in all_metrics[cond] if np.isfinite(m[metric_key])]
            mean, lo, hi = bootstrap_ci(vals)
            means.append(mean)
            ci_los.append(mean - lo if np.isfinite(lo) else 0)
            ci_his.append(hi - mean if np.isfinite(hi) else 0)
            colors_list.append(COLORS[cond])

        x = np.arange(n_conds)
        bars = ax.bar(x, means, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x, means, yerr=[ci_los, ci_his], fmt='none', ecolor='black',
                     capsize=4, linewidth=1.2)

        # Add significance stars for pairwise comparisons
        test_results = pairwise_mannwhitney(all_metrics, metric_key)
        sig_pairs = [(r['comparison'], r['p_corrected'])
                     for r in test_results if np.isfinite(r['p_corrected']) and r['p_corrected'] < 0.05]

        y_max = max(m + h for m, h in zip(means, ci_his)) if means else 0
        for pair_str, p in sig_pairs:
            parts = pair_str.split(' vs ')
            if parts[0] in cond_names and parts[1] in cond_names:
                i = cond_names.index(parts[0])
                j = cond_names.index(parts[1])
                star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                y_max *= 1.08
                ax.plot([i, i, j, j], [y_max * 0.95, y_max, y_max, y_max * 0.95],
                        'k-', linewidth=0.8)
                ax.text((i + j) / 2, y_max, star, ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' + ', '\n+ ') for c in cond_names],
                            fontsize=8, rotation=0)
        ax.set_title(metric_labels[metric_key], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Continual Learning Metrics (mean with 95% bootstrap CI)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


def plot_2x2_grid(data, filename):
    """Fig 5: 2x2 grid showing A vs B for each method (core 4 conditions)."""
    plot_conds = [c for c in CORE_CONDITIONS if c in data]
    n = len(plot_conds)
    if n == 0:
        print("  Skipping 2x2 grid: no core conditions with data")
        return

    nrows = (n + 1) // 2
    ncols = min(n, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), sharex=True)
    if n == 1:
        axes = np.array([[axes]])
    elif n == 2:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for idx, cond in enumerate(plot_conds):
        ax = axes_flat[idx]
        trials = data[cond]

        gens_a, mean_a, _, se_a, _ = aggregate_trajectories(trials, 'mse_on_a')
        gens_b, mean_b, _, se_b, _ = aggregate_trajectories(trials, 'mse_on_b')

        ax.plot(gens_a, mean_a, color='#D32F2F', linewidth=1.5,
                label='MSE on A (summer)', alpha=0.85)
        ax.fill_between(gens_a, mean_a - se_a, mean_a + se_a,
                         color='#D32F2F', alpha=0.12)

        ax.plot(gens_b, mean_b, color='#1565C0', linewidth=1.5,
                label='MSE on B (winter)', alpha=0.85)
        ax.fill_between(gens_b, mean_b - se_b, mean_b + se_b,
                         color='#1565C0', alpha=0.12)

        # Phase boundary
        boundaries = [detect_phase_boundary(df) for df in trials]
        boundaries = [b for b in boundaries if b is not None]
        if boundaries:
            ax.axvline(x=np.median(boundaries), color='gray',
                       linestyle=':', linewidth=1.2, alpha=0.7)

        ax.set_title(cond, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        if idx >= ncols:
            ax.set_xlabel('Generation')
        if idx % ncols == 0:
            ax.set_ylabel('MSE (held-out)')

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('MSE on Each Distribution by Method (mean ± SE across trials)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(all_metrics, test_results_by_metric, filepath):
    """Generate LaTeX table with mean±std for all metrics per condition."""
    metric_keys = ['bwt', 'aufc', 'peak_ratio', 'recovery_time', 'ps_ratio',
                   'final_mse_a', 'final_mse_b', 'mse_a_at_transition', 'mse_b_at_transition']
    metric_headers = {
        'bwt': 'BWT',
        'aufc': 'AUFC',
        'peak_ratio': 'Peak Ratio',
        'recovery_time': 'Recovery (gens)',
        'ps_ratio': 'P/S Ratio',
        'final_mse_a': 'Final MSE$_A$',
        'final_mse_b': 'Final MSE$_B$',
        'mse_a_at_transition': 'MSE$_A$ @ trans.',
        'mse_b_at_transition': 'MSE$_B$ @ trans.',
    }

    cond_names = [c for c in CONDITIONS if c in all_metrics]

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Continual learning metrics (mean $\pm$ std, $n$ trials). '
                 r'$*$/$**$/$***$: Bonferroni-corrected $p < 0.05/0.01/0.001$ (Mann-Whitney U).}')
    lines.append(r'\label{tab:forgetting_metrics}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    cols = 'l' + 'c' * len(metric_keys)
    lines.append(r'\begin{tabular}{' + cols + '}')
    lines.append(r'\toprule')

    header = 'Condition & ' + ' & '.join(metric_headers[k] for k in metric_keys) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    for cond in cond_names:
        trials_metrics = all_metrics[cond]
        n = len(trials_metrics)
        cells = [cond.replace('_', r'\_') + f' ($n$={n})']

        for mk in metric_keys:
            vals = [m[mk] for m in trials_metrics if np.isfinite(m[mk])]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0
                if mk in ('bwt', 'aufc', 'final_mse_a', 'final_mse_b',
                           'mse_a_at_transition', 'mse_b_at_transition'):
                    cells.append(f'{mean:.6f} $\\pm$ {std:.6f}')
                elif mk == 'recovery_time':
                    cells.append(f'{mean:.0f} $\\pm$ {std:.0f}')
                else:
                    cells.append(f'{mean:.3f} $\\pm$ {std:.3f}')
            else:
                cells.append('--')

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    with open(filepath, 'w') as f:
        f.write(tex)
    print(f"  Saved: {os.path.basename(filepath)}")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_metrics):
    """Print summary statistics to stdout."""
    print("\n" + "=" * 90)
    print("SUMMARY METRICS (mean ± std)")
    print("=" * 90)

    metric_keys = ['bwt', 'aufc', 'peak_ratio', 'recovery_time', 'ps_ratio',
                   'final_mse_a', 'final_mse_b']

    for cond in CONDITIONS:
        if cond not in all_metrics:
            continue
        trials_metrics = all_metrics[cond]
        n = len(trials_metrics)
        print(f"\n{'─' * 70}")
        print(f"  {cond}  (n={n})")
        print(f"{'─' * 70}")

        for mk in metric_keys:
            vals = [m[mk] for m in trials_metrics if np.isfinite(m[mk])]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0
                print(f"  {mk:25s}: {mean:10.6f} ± {std:.6f}  "
                      f"[95% CI: {lo:.6f}, {hi:.6f}]  (n={len(vals)})")
            else:
                print(f"  {mk:25s}: N/A")


def print_statistical_tests(all_metrics):
    """Print pairwise Mann-Whitney U tests."""
    test_metrics = ['bwt', 'aufc', 'final_mse_a', 'final_mse_b', 'peak_ratio']

    print("\n" + "=" * 90)
    print("PAIRWISE MANN-WHITNEY U TESTS (Bonferroni-corrected)")
    print("=" * 90)
    print(f"\nNOTE: With n=5, Mann-Whitney U has minimum achievable p ≈ 0.008.")
    print(f"      Statistical power is limited; effect sizes (rank-biserial r) are more informative.\n")

    all_test_results = {}
    for mk in test_metrics:
        results = pairwise_mannwhitney(all_metrics, mk)
        all_test_results[mk] = results

        print(f"\n  Metric: {mk}")
        print(f"  {'Comparison':<50s} {'U':>6s} {'p_raw':>10s} {'p_corr':>10s} {'r_rb':>8s} {'n1':>3s} {'n2':>3s}")
        print(f"  {'─' * 95}")

        for r in results:
            sig = ''
            if np.isfinite(r['p_corrected']):
                if r['p_corrected'] < 0.001:
                    sig = ' ***'
                elif r['p_corrected'] < 0.01:
                    sig = ' **'
                elif r['p_corrected'] < 0.05:
                    sig = ' *'

            p_raw_str = f"{r['p_raw']:.4f}" if np.isfinite(r['p_raw']) else 'N/A'
            p_corr_str = f"{r['p_corrected']:.4f}" if np.isfinite(r['p_corrected']) else 'N/A'
            u_str = f"{r['U']:.1f}" if np.isfinite(r['U']) else 'N/A'
            r_str = f"{r['r_rb']:.3f}" if np.isfinite(r['r_rb']) else 'N/A'

            print(f"  {r['comparison']:<50s} {u_str:>6s} {p_raw_str:>10s} "
                  f"{p_corr_str:>10s} {r_str:>8s} {r['n1']:>3d} {r['n2']:>3d}{sig}")

    return all_test_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Catastrophic Forgetting: Statistical Analysis for GECCO Paper")
    print("=" * 60)

    # Load data
    print("\nLoading trial data...")
    data = load_all_trials()

    if not data:
        print("ERROR: No data found. Check that results/ directory contains forgetting_results.csv files.")
        sys.exit(1)

    print(f"\nConditions loaded: {len(data)}")
    for cond, trials in data.items():
        n = len(trials)
        max_gens = [df['generation'].max() for df in trials]
        boundaries = [detect_phase_boundary(df) for df in trials]
        print(f"  {cond}: {n} trial(s), "
              f"max gen {np.mean(max_gens):.0f}, "
              f"boundary at gen {np.mean([b for b in boundaries if b]):.0f}")

    if any(len(t) < N_TRIALS for t in data.values()):
        print(f"\n  WARNING: Some conditions have fewer than {N_TRIALS} trials. "
              f"Statistical tests will have reduced power.")

    # Compute metrics
    print("\nComputing per-trial metrics...")
    all_metrics = compute_all_metrics(data)

    # Print summary
    print_summary(all_metrics)

    # Statistical tests
    all_test_results = print_statistical_tests(all_metrics)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    print("\nFig 1: MSE on A over generations")
    plot_trajectory(data, 'mse_on_a',
                    'MSE on Distribution A (held-out)',
                    'Catastrophic Forgetting: Performance on Distribution A Over Time',
                    'fig1_mse_on_a')

    print("Fig 2: MSE on B over generations")
    plot_trajectory(data, 'mse_on_b',
                    'MSE on Distribution B (held-out)',
                    'Adaptation to New Distribution: Performance on Distribution B Over Time',
                    'fig2_mse_on_b')

    print("Fig 3: Forgetting ratio during mixed phase")
    plot_forgetting_ratio(data, 'fig3_forgetting_ratio')

    print("Fig 4: Bar chart with bootstrap CIs")
    plot_bar_chart(all_metrics, 'fig4_bar_metrics')

    print("Fig 5: 2x2 grid (A vs B per method)")
    plot_2x2_grid(data, 'fig5_2x2_grid')

    # LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(all_metrics, all_test_results,
                         os.path.join(OUT_DIR, 'metrics_table.tex'))

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
