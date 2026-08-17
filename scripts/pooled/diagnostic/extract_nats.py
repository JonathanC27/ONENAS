#!/usr/bin/env python
"""
Low-memory fallback / independent cross-check extractor.

Source artefact: NATS-tss-v1_0-3ffb9-simple.tar  (NATS-Bench topology search
space = the identical 15,625-architecture NAS-Bench-201 space; Dong, Liu,
Musial & Gabrys, "NATS-Bench: Benchmarking NAS Algorithms for Architecture
Topology and Size", IEEE TPAMI 2021; the NAS-Bench-201 repo README states
NAS-Bench-201 "has been extended to NATS-Bench").

One bz2'd pickle per architecture, so this streams with ~0 memory and runs on a
login node.  Emits the same schema as extract_nb201.py.
"""
import os
import sys
import bz2
import pickle
import gzip
import csv
import glob

ROOT = '/anvil/scratch/x-jchang5/nb201/nats'
OUT = '/anvil/scratch/x-jchang5/nb201/nats_seedlevel.csv.gz'


def log(*a):
    print(*a, flush=True)


def pload(p):
    with bz2.open(p, 'rb') as f:
        return pickle.load(f)


def main():
    cands = glob.glob(os.path.join(ROOT, '**', '*.pickle.pbz2'), recursive=True)
    log('== found', len(cands), 'pickle.pbz2 files')
    dirs = sorted(set(os.path.dirname(c) for c in cands))
    log('== dirs:', dirs[:5])
    arch_files = [c for c in cands if os.path.basename(c)[0].isdigit()]
    log('== numbered arch files:', len(arch_files))
    arch_files.sort(key=lambda p: int(os.path.basename(p).split('.')[0]))

    x = pload(arch_files[0])
    log('== per-arch top keys:', list(x.keys()) if isinstance(x, dict) else type(x))
    hpk = sorted(x.keys(), key=str)
    sd = x[hpk[-1]]
    log('== ArchResults keys:', sorted(sd.keys()))
    log('== arch_str:', sd.get('arch_str'))
    log('== dataset_seed:', sd.get('dataset_seed'))
    rk = sorted(sd['all_results'].keys(), key=str)
    log('== all_results keys:', rk)
    rc = sd['all_results'][rk[0]]
    log('== ResultsCount keys:', sorted(rc.keys()))
    log('== eval_names:', rc.get('eval_names'), 'epochs:', rc.get('epochs'))
    ek = list(rc['eval_acc1es'].keys())
    log('== eval_acc1es n=%d first=%s last=%s' % (len(ek), ek[:4], ek[-4:]))

    fh = gzip.open(OUT, 'wt', newline='')
    w = csv.writer(fh)
    w.writerow(['arch_index', 'hp', 'dataset', 'seed', 'epochs',
                'params_MB', 'flops_M', 'latency',
                'train_acc', 'eval_name', 'eval_acc', 'eval_loss'])
    nrow = 0
    for n, p in enumerate(arch_files):
        idx = int(os.path.basename(p).split('.')[0])
        x = pload(p)
        for hpkey, sd in x.items():
            try:
                hp = int(hpkey)
            except (TypeError, ValueError):
                hp = hpkey
            for (ds, seed), rc in sd['all_results'].items():
                ep = rc['epochs']
                last = ep - 1
                lat = rc.get('latency')
                if isinstance(lat, (list, tuple)):
                    lat = (sum(lat) / len(lat)) if len(lat) else None
                ta = rc.get('train_acc1es')
                train_acc = ta.get(last) if isinstance(ta, dict) else None
                for name in rc['eval_names']:
                    key = '{:}@{:}'.format(name, last)
                    acc = rc['eval_acc1es'].get(key)
                    if acc is None:
                        continue
                    w.writerow([idx, hp, ds, seed, ep, rc.get('params'), rc.get('flop'),
                                lat, train_acc, name, acc, rc['eval_losses'].get(key)])
                    nrow += 1
        if (n + 1) % 2000 == 0:
            log('   ... %d/%d, %d rows' % (n + 1, len(arch_files), nrow))
    fh.close()
    log('== DONE rows =', nrow, 'size =', os.path.getsize(OUT))


if __name__ == '__main__':
    main()
