#!/usr/bin/env python3
"""
POSITIVE CONTROL for the "will architecture search help?" pre-flight diagnostic,
run on NAS-Bench-201 (Dong & Yang, ICLR 2020).

Artefact: NAS-Bench-201-v1_1-096897.pth, md5 55e847143ce1f7c2d89b676f6b096897
(last six digits 096897, matching the official README).  Reduced on Anvil to
nb201_seedlevel.csv.gz by extract_nb201.py.

Runs the IDENTICAL protocol used for the equities negative case:
  (i)   CHAMPION vs POPULATION   -> argmax-of-k vs mean-of-k / random-pick-of-k
  (ii)  OVERSEARCHING            -> argmax-of-k as a function of k
  (iii) VAL->TEST rank corr      -> Spearman / Kendall across candidates
plus a seed-noise ceiling and a synthetic SNR dial.
"""
import os
import sys
import gzip
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'nb201_seedlevel.csv.gz')
OUTJSON = os.path.join(HERE, 'results.json')

RNG_SEED = 20200126  # arXiv:2001.00326
N_DRAWS = 5000
KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 250, 500, 1000, 2500]

# (label, (val_dataset, val_evalname), (test_dataset, test_evalname), same_run)
PROTOCOLS = [
    ('CIFAR-10 (standard NAS protocol, cross-run)',
     ('cifar10-valid', 'x-valid'), ('cifar10', 'ori-test'), False),
    ('CIFAR-10 (within-run: 25k-train model)',
     ('cifar10-valid', 'x-valid'), ('cifar10-valid', 'ori-test'), True),
    ('CIFAR-100',
     ('cifar100', 'x-valid'), ('cifar100', 'x-test'), True),
    ('ImageNet16-120',
     ('ImageNet16-120', 'x-valid'), ('ImageNet16-120', 'x-test'), True),
]


# --------------------------------------------------------------------------- #
def load(hp=200):
    df = pd.read_csv(DATA)
    df = df[df.hp == hp]
    return df


def matrixify(df, dataset, evalname, n_arch):
    """-> (n_arch, max_seeds) float array with NaN padding, seeds sorted."""
    sub = df[(df.dataset == dataset) & (df.eval_name == evalname)]
    sub = sub.sort_values(['arch_index', 'seed'])
    seeds = sorted(sub.seed.unique())
    M = np.full((n_arch, len(seeds)), np.nan)
    sidx = {s: j for j, s in enumerate(seeds)}
    M[sub.arch_index.values, [sidx[s] for s in sub.seed.values]] = sub.eval_acc.values
    return M, seeds


def nan_choice(M, rng, n_draws_shape):
    """Pick one non-nan seed per (draw, arch) element. M is (n_arch, n_seed).
    n_draws_shape is the index array of arch ids; returns same-shaped values."""
    cnt = (~np.isnan(M)).sum(1)                       # (n_arch,)
    # compact each row's valid entries to the front
    order = np.argsort(np.isnan(M), axis=1, kind='stable')
    Mc = np.take_along_axis(M, order, axis=1)
    a = n_draws_shape
    j = (rng.random(a.shape) * cnt[a]).astype(np.int64)
    return Mc[a, j]


def sample_pools(rng, n_arch, k, n_draws, chunk=256):
    """Yield (chunk, k) arrays of arch indices sampled WITHOUT replacement."""
    done = 0
    while done < n_draws:
        c = min(chunk, n_draws - done)
        r = rng.random((c, n_arch))
        idx = np.argpartition(r, k - 1, axis=1)[:, :k]
        done += c
        yield idx


# --------------------------------------------------------------------------- #
def rank_corrs(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    sp = stats.spearmanr(x[ok], y[ok])
    kt = stats.kendalltau(x[ok], y[ok])
    pe = stats.pearsonr(x[ok], y[ok])
    return dict(spearman=float(sp.statistic), spearman_p=float(sp.pvalue),
                kendall=float(kt.statistic), kendall_p=float(kt.pvalue),
                pearson=float(pe[0]), n=int(ok.sum()))


def seed_noise(M):
    """Variance decomposition over seeds. M (n_arch, n_seed) with NaNs."""
    cnt = (~np.isnan(M)).sum(1)
    use = cnt >= 2
    within = np.nanvar(M[use], axis=1, ddof=1)            # per-arch seed var
    sigma2_e = float(np.mean(within))                     # noise var (1 seed)
    mean_all = np.nanmean(M, axis=1)
    nbar = float(np.mean(cnt[cnt > 0]))
    var_of_means = float(np.var(mean_all[cnt > 0], ddof=1))
    sigma2_a = max(var_of_means - sigma2_e / nbar, 0.0)   # true between-arch var
    icc1 = sigma2_a / (sigma2_a + sigma2_e)               # reliability, 1 seed
    return dict(sigma_between=float(np.sqrt(sigma2_a)),
                sigma_seed=float(np.sqrt(sigma2_e)),
                icc_single_seed=float(icc1),
                mean_n_seeds=nbar,
                frac_arch_ge2_seeds=float(np.mean(use)))


def split_half_ceiling(M, rng, n_rep=200):
    """Spearman between two DISJOINT single-seed measurements of the same
    quantity -> the empirical ceiling on any val/test rank correlation."""
    cnt = (~np.isnan(M)).sum(1)
    use = np.where(cnt >= 2)[0]
    if len(use) < 100:
        return None
    order = np.argsort(np.isnan(M), axis=1, kind='stable')
    Mc = np.take_along_axis(M, order, axis=1)
    out = []
    for _ in range(n_rep):
        j1 = (rng.random(len(use)) * cnt[use]).astype(np.int64)
        j2 = (rng.random(len(use)) * (cnt[use] - 1)).astype(np.int64)
        j2 = j2 + (j2 >= j1)                              # ensure j2 != j1
        a = Mc[use, j1]
        b = Mc[use, j2]
        out.append(stats.spearmanr(a, b).statistic)
    return dict(spearman_mean=float(np.mean(out)), spearman_sd=float(np.std(out)),
                n_arch=int(len(use)))


# --------------------------------------------------------------------------- #
def overseach_sweep(VAL, TEST, rng, label, val_mode):
    """val_mode: 'seedmean' or 'single'."""
    n_arch = VAL.shape[0]
    valid = (~np.isnan(VAL).all(1)) & (~np.isnan(TEST).all(1))
    ids = np.where(valid)[0]
    Vm = np.nanmean(VAL, axis=1)
    Tm = np.nanmean(TEST, axis=1)            # GROUND TRUTH test accuracy
    rows = []
    for k in KS:
        if k > len(ids):
            continue
        am, mm, rm = [], [], []
        for pool_local in sample_pools(rng, len(ids), k, N_DRAWS):
            pool = ids[pool_local]
            if val_mode == 'seedmean':
                v = Vm[pool]
            else:
                v = nan_choice(VAL, rng, pool)
            t = Tm[pool]
            sel = np.argmax(v, axis=1)
            am.append(t[np.arange(len(pool)), sel])
            mm.append(t.mean(axis=1))
            rp = rng.integers(0, k, size=len(pool))
            rm.append(t[np.arange(len(pool)), rp])
        am = np.concatenate(am); mm = np.concatenate(mm); rm = np.concatenate(rm)
        rows.append(dict(k=k,
                         argmax_mean=float(am.mean()), argmax_se=float(am.std(ddof=1) / np.sqrt(len(am))),
                         mean_mean=float(mm.mean()), mean_se=float(mm.std(ddof=1) / np.sqrt(len(mm))),
                         rand_mean=float(rm.mean()), rand_se=float(rm.std(ddof=1) / np.sqrt(len(rm))),
                         argmax_minus_mean=float((am - mm).mean()),
                         argmax_minus_mean_se=float((am - mm).std(ddof=1) / np.sqrt(len(am))),
                         argmax_minus_rand=float((am - rm).mean()),
                         argmax_minus_rand_se=float((am - rm).std(ddof=1) / np.sqrt(len(am))),
                         n_draws=int(len(am))))
    return rows


# --------------------------------------------------------------------------- #
def synthetic_snr(TEST_TRUE, rng, alphas, n_draws=4000, ks=(1, 10, 100, 1000)):
    """SNR dial. True architecture quality t = real NB-201 CIFAR-10 test acc.
    The experimenter observes a noisy validation score
        v_i = a * z(t_i) + sqrt(1-a^2) * eps_i,
    with eps DRAWN FRESH FOR EACH SEARCH RUN (this is the point: each run sees
    its own noise realisation, exactly as a real search does).
    Reports all three diagnostic measurements as a function of a."""
    t = TEST_TRUE[~np.isnan(TEST_TRUE)]
    z = (t - t.mean()) / t.std()
    n = len(z)
    out = []
    for a in alphas:
        b = np.sqrt(max(1 - a * a, 0.0))
        # (iii) measured rank corr: what the experimenter would compute from one
        # observed validation sweep over all candidates
        sps = [stats.spearmanr(a * z + b * rng.standard_normal(n), t).statistic
               for _ in range(20)]
        row = dict(alpha=float(a), spearman=float(np.mean(sps)))
        for k in ks:
            am, mm, rm = [], [], []
            for pool in sample_pools(rng, n, k, n_draws):
                tt = t[pool]
                vv = a * z[pool] + b * rng.standard_normal(pool.shape)  # fresh noise
                sel = np.argmax(vv, axis=1)
                am.append(tt[np.arange(len(pool)), sel])
                mm.append(tt.mean(axis=1))
                rm.append(tt[np.arange(len(pool)), rng.integers(0, k, size=len(pool))])
            am = np.concatenate(am); mm = np.concatenate(mm); rm = np.concatenate(rm)
            row['argmax_k%d' % k] = float(am.mean())
            row['argmax_k%d_se' % k] = float(am.std(ddof=1) / np.sqrt(len(am)))
            row['mean_k%d' % k] = float(mm.mean())
            row['rand_k%d' % k] = float(rm.mean())
            row['amax_minus_mean_k%d' % k] = float((am - mm).mean())
            row['amax_minus_rand_k%d' % k] = float((am - rm).mean())
        row['oversearch_1_to_%d' % ks[-1]] = row['argmax_k%d' % ks[-1]] - row['argmax_k1']
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(RNG_SEED)
    df = load(hp=200)
    n_arch = int(df.arch_index.max()) + 1
    print('n_arch =', n_arch, ' rows =', len(df))
    print('datasets:', sorted(df.dataset.unique()))
    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        print('  %-16s evals=%s seeds=%s n_arch=%d'
              % (ds, sorted(sub.eval_name.unique()), sorted(sub.seed.unique()),
                 sub.arch_index.nunique()))

    results = {'artefact': {
        'file': 'NAS-Bench-201-v1_1-096897.pth',
        'md5': '55e847143ce1f7c2d89b676f6b096897',
        'md5_last6_matches_official_name': True,
        'paper': 'Dong & Yang, ICLR 2020, arXiv:2001.00326',
        'hp': 200, 'n_arch': n_arch}, 'protocols': {}}

    for label, (vds, vev), (tds, tev), same_run in PROTOCOLS:
        print('\n' + '=' * 78)
        print(label)
        print('  val  =', vds, '/', vev, '   test =', tds, '/', tev)
        VAL, vseeds = matrixify(df, vds, vev, n_arch)
        TEST, tseeds = matrixify(df, tds, tev, n_arch)
        print('  val seeds', [int(s) for s in vseeds], ' test seeds', [int(s) for s in tseeds])
        Vm, Tm = np.nanmean(VAL, axis=1), np.nanmean(TEST, axis=1)
        ok = ~(np.isnan(Vm) | np.isnan(Tm))
        print('  n usable archs', ok.sum())
        pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        qs = np.nanpercentile(Tm, pcts)
        print('  test acc: mean %.3f  sd %.3f  max %.3f  min %.3f'
              % (np.nanmean(Tm), np.nanstd(Tm), np.nanmax(Tm), np.nanmin(Tm)))
        print('  test acc percentiles ' + '  '.join('p%d=%.2f' % (p, q) for p, q in zip(pcts, qs)))
        chance = 100.0 / {'cifar10': 10, 'cifar10-valid': 10, 'cifar100': 100,
                          'ImageNet16-120': 120}[tds]
        frac_near_chance = float(np.nanmean(Tm < 2 * chance))
        print('  fraction of the space within 2x chance accuracy: %.3f' % frac_near_chance)

        R = {}
        R['n_arch_usable'] = int(ok.sum())
        R['val_seeds'] = [int(s) for s in vseeds]
        R['test_seeds'] = [int(s) for s in tseeds]
        R['test_acc_stats'] = dict(mean=float(np.nanmean(Tm)), sd=float(np.nanstd(Tm)),
                                   max=float(np.nanmax(Tm)), min=float(np.nanmin(Tm)),
                                   percentiles={str(p): float(q) for p, q in zip(pcts, qs)},
                                   chance=chance, frac_within_2x_chance=frac_near_chance)

        # ---- P1 rank correlation --------------------------------------------
        R['P1_seedmean'] = rank_corrs(Vm, Tm)
        v1 = nan_choice(VAL, rng, np.where(ok)[0])
        R['P1_singleseed_val'] = rank_corrs(v1, Tm[ok])
        print('  P1 spearman (seed-mean val vs true test) = %.4f  kendall = %.4f'
              % (R['P1_seedmean']['spearman'], R['P1_seedmean']['kendall']))
        print('  P1 spearman (SINGLE-seed val)            = %.4f  kendall = %.4f'
              % (R['P1_singleseed_val']['spearman'], R['P1_singleseed_val']['kendall']))

        # ---- noise ceiling ---------------------------------------------------
        R['noise_val'] = seed_noise(VAL)
        R['noise_test'] = seed_noise(TEST)
        R['ceiling_val_splithalf'] = split_half_ceiling(VAL, rng)
        R['ceiling_test_splithalf'] = split_half_ceiling(TEST, rng)
        print('  noise: val sigma_seed=%.3f icc=%.3f | test sigma_seed=%.3f icc=%.3f'
              % (R['noise_val']['sigma_seed'], R['noise_val']['icc_single_seed'],
                 R['noise_test']['sigma_seed'], R['noise_test']['icc_single_seed']))
        if R['ceiling_test_splithalf']:
            print('  split-half ceiling (test seed A vs seed B) spearman = %.4f'
                  % R['ceiling_test_splithalf']['spearman_mean'])

        # ---- P2/P3/P4 sweeps -------------------------------------------------
        for mode in ('seedmean', 'single'):
            rows = overseach_sweep(VAL, TEST, rng, label, mode)
            R['sweep_' + mode] = rows
            print('  --- argmax-of-k sweep, val=%s ---' % mode)
            print('     k   argmax(SE)      mean(SE)       rand(SE)     amax-mean   amax-rand')
            for r in rows:
                print('  %5d  %6.3f(%.3f)  %6.3f(%.3f)  %6.3f(%.3f)  %+7.3f    %+7.3f'
                      % (r['k'], r['argmax_mean'], r['argmax_se'],
                         r['mean_mean'], r['mean_se'], r['rand_mean'], r['rand_se'],
                         r['argmax_minus_mean'], r['argmax_minus_rand']))
        # ---- P1..P4 verdict summary -----------------------------------------
        sw = {r['k']: r for r in R['sweep_single']}
        kmax = max(sw)
        d2 = sw[kmax]['argmax_mean'] - sw[1]['argmax_mean']
        s2 = np.hypot(sw[kmax]['argmax_se'], sw[1]['argmax_se'])
        V = dict(
            P1_spearman=R['P1_singleseed_val']['spearman'],
            P1_kendall=R['P1_singleseed_val']['kendall'],
            P1_pass=bool(R['P1_singleseed_val']['spearman'] > 0.5),
            P2_delta=float(d2), P2_t=float(d2 / s2), P2_kmax=int(kmax),
            P2_pass=bool(d2 / s2 > 2),
            P3_delta=sw[kmax]['argmax_minus_mean'],
            P3_t=sw[kmax]['argmax_minus_mean'] / sw[kmax]['argmax_minus_mean_se'],
            P3_pass=bool(sw[kmax]['argmax_minus_mean'] > 0),
            P4_delta=sw[kmax]['argmax_minus_rand'],
            P4_t=sw[kmax]['argmax_minus_rand'] / sw[kmax]['argmax_minus_rand_se'],
            P4_pass=bool(sw[kmax]['argmax_minus_rand'] > 0))
        R['verdict'] = V
        print('  VERDICT  P1 rho=%.3f [%s] | P2 argmax@1->@%d %+0.3f t=%.1f [%s] | '
              'P3 amax-mean %+0.3f t=%.1f [%s] | P4 amax-rand %+0.3f t=%.1f [%s]'
              % (V['P1_spearman'], 'PASS' if V['P1_pass'] else 'FAIL',
                 kmax, V['P2_delta'], V['P2_t'], 'PASS' if V['P2_pass'] else 'FAIL',
                 V['P3_delta'], V['P3_t'], 'PASS' if V['P3_pass'] else 'FAIL',
                 V['P4_delta'], V['P4_t'], 'PASS' if V['P4_pass'] else 'FAIL'))
        results['protocols'][label] = R

    # ---- hp=12 low-fidelity proxy (noisier regime) --------------------------
    print('\n' + '=' * 78)
    print('LOW-FIDELITY PROXY: 12-epoch validation -> 200-epoch true test')
    df12 = load(hp=12)
    lf = {}
    for label, (vds, vev), (tds, tev), _ in PROTOCOLS:
        try:
            VAL12, _ = matrixify(df12, vds, vev, n_arch)
        except Exception as e:
            print('  skip', label, e); continue
        TEST, _ = matrixify(df, tds, tev, n_arch)
        Vm, Tm = np.nanmean(VAL12, axis=1), np.nanmean(TEST, axis=1)
        if np.isnan(Vm).all():
            continue
        rc = rank_corrs(Vm, Tm)
        rows = overseach_sweep(VAL12, TEST, rng, label, 'seedmean')
        lf[label] = dict(P1=rc, sweep=rows)
        print('  %-45s spearman=%.4f  argmax k=1 %.3f -> k=%d %.3f (%+.3f)'
              % (label, rc['spearman'], rows[0]['argmax_mean'],
                 rows[-1]['k'], rows[-1]['argmax_mean'],
                 rows[-1]['argmax_mean'] - rows[0]['argmax_mean']))
    results['low_fidelity_hp12'] = lf

    # ---- synthetic SNR dial --------------------------------------------------
    print('\n' + '=' * 78)
    print('SYNTHETIC SNR DIAL (true architecture quality = NB-201 CIFAR-10 test acc)')
    TESTc, _ = matrixify(df, 'cifar10', 'ori-test', n_arch)
    Tm = np.nanmean(TESTc, axis=1)
    alphas = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0]
    snr = synthetic_snr(Tm, rng, alphas)
    results['synthetic_snr'] = snr
    print('  alpha  spearman   argmax@1  argmax@10 argmax@100 argmax@1000  rand@1000  mean@1000  overs(1->1000)  amax-rand@1000')
    for r in snr:
        print('  %5.2f  %+8.4f   %7.3f  %8.3f  %9.3f  %10.3f  %9.3f  %9.3f  %+8.3f  %+8.3f'
              % (r['alpha'], r['spearman'], r['argmax_k1'], r['argmax_k10'],
                 r['argmax_k100'], r['argmax_k1000'], r['rand_k1000'],
                 r['mean_k1000'], r['oversearch_1_to_1000'],
                 r['amax_minus_rand_k1000']))

    with open(OUTJSON, 'w') as f:
        json.dump(results, f, indent=1)
    print('\nwrote', OUTJSON)


if __name__ == '__main__':
    main()
