#!/usr/bin/env python
"""
Extract a compact (validation accuracy, test accuracy) table from the OFFICIAL
NAS-Bench-201 benchmark file.

Source artefact: NAS-Bench-201-v1_1-096897.pth
  Dong & Yang, "NAS-Bench-201: Extending the Scope of Reproducible Neural
  Architecture Search", ICLR 2020 (arXiv:2001.00326).
  Official download listed in https://github.com/D-X-Y/NAS-Bench-201 README:
    "[2020.03.16] APIv1.3/FILEv1.1: NAS-Bench-201-v1_1-096897.pth (4.7G),
     where 096897 is the last six digits for this file."
  Verified md5 = 55e847143ce1f7c2d89b676f6b096897 -> last six digits 096897. OK.

We do NOT use the nas_201_api package; we parse the raw state-dicts directly
(they are plain nested python dicts), which avoids constructing 31k ArchResults
objects.  Semantics replicated from nas_201_api/api.py:
  ArchResults.state_dict() -> {'arch_index','arch_str','all_results',
                               'dataset_seed','clear_net_done'}
      all_results : {(dataset, seed): ResultsCount.state_dict()}
  ResultsCount.get_eval(name, iepoch) reads  eval_acc1es['{name}@{iepoch}']
      with iepoch defaulting to epochs-1.
"""
import os
import sys
import gzip
import csv
import gc
import torch

PATH = '/anvil/scratch/x-jchang5/nb201/NAS-Bench-201-v1_1-096897.pth'
OUT = '/anvil/scratch/x-jchang5/nb201/nb201_seedlevel.csv.gz'


def log(*a):
    print(*a, flush=True)


def main():
    log('== loading', PATH, os.path.getsize(PATH), 'bytes')
    d = torch.load(PATH, map_location='cpu', weights_only=False)
    log('== top-level keys:', sorted(d.keys()))

    meta = d['meta_archs']
    evaluated = sorted(d['evaluated_indexes'])
    a2i = d['arch2infos']
    log('== n_meta_archs =', len(meta))
    log('== n_evaluated  =', len(evaluated))
    log('== n_arch2infos =', len(a2i))

    # ---- structural probe on the first architecture -------------------------
    k0 = evaluated[0]
    log('== arch2infos[%d] keys: %s' % (k0, sorted(a2i[k0].keys())))
    sd0 = a2i[k0]['full']
    log('== ArchResults(full) keys:', sorted(sd0.keys()))
    log('== arch_str:', sd0['arch_str'])
    log('== dataset_seed:', sd0['dataset_seed'])
    rk = sorted(sd0['all_results'].keys(), key=str)
    log('== all_results keys:', rk)
    rc0 = sd0['all_results'][rk[0]]
    log('== ResultsCount keys:', sorted(rc0.keys()))
    log('== eval_names:', rc0.get('eval_names'), ' epochs:', rc0.get('epochs'))
    ek = list(rc0['eval_acc1es'].keys())
    log('== eval_acc1es n=%d first5=%s last5=%s' % (len(ek), ek[:5], ek[-5:]))
    sdl = a2i[k0]['less']
    rkl = sorted(sdl['all_results'].keys(), key=str)
    log('== LESS all_results keys:', rkl)
    rcl = sdl['all_results'][rkl[0]]
    log('== LESS eval_names:', rcl.get('eval_names'), ' epochs:', rcl.get('epochs'))

    # ---- full extraction ----------------------------------------------------
    fh = gzip.open(OUT, 'wt', newline='')
    w = csv.writer(fh)
    w.writerow(['arch_index', 'hp', 'dataset', 'seed', 'epochs',
                'params_MB', 'flops_M', 'latency',
                'train_acc', 'eval_name', 'eval_acc', 'eval_loss'])

    arch_fh = gzip.open('/anvil/scratch/x-jchang5/nb201/nb201_archs.csv.gz', 'wt', newline='')
    aw = csv.writer(arch_fh)
    aw.writerow(['arch_index', 'arch_str'])

    nrow = 0
    for n, idx in enumerate(evaluated):
        entry = a2i[idx]
        wrote_arch = False
        for tag, hp in (('less', 12), ('full', 200)):
            if tag not in entry or entry[tag] is None:
                continue
            sd = entry[tag]
            if not wrote_arch:
                aw.writerow([idx, sd['arch_str']])
                wrote_arch = True
            for (ds, seed), rc in sd['all_results'].items():
                ep = rc['epochs']
                last = ep - 1
                params = rc.get('params')
                flop = rc.get('flop')
                lat = rc.get('latency')
                if isinstance(lat, (list, tuple)):
                    lat = (sum(lat) / len(lat)) if len(lat) else None
                ta = rc.get('train_acc1es')
                train_acc = ta.get(last) if isinstance(ta, dict) else None
                for name in rc['eval_names']:
                    key = '{:}@{:}'.format(name, last)
                    acc = rc['eval_acc1es'].get(key)
                    loss = rc['eval_losses'].get(key)
                    if acc is None:
                        continue
                    w.writerow([idx, hp, ds, seed, ep, params, flop, lat,
                                train_acc, name, acc, loss])
                    nrow += 1
        # free as we go
        a2i[idx] = None
        if (n + 1) % 2000 == 0:
            log('   ... %d/%d archs, %d rows' % (n + 1, len(evaluated), nrow))
            gc.collect()

    fh.close()
    arch_fh.close()
    log('== DONE rows =', nrow)
    log('== out size  =', os.path.getsize(OUT))


if __name__ == '__main__':
    main()
