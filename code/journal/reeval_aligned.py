"""Re-evaluate trained runs with all BN configurations on identical noise draws.

Why this exists
---------------
``protocol_v3.py`` evaluates the as-trained model with ``seed_base=10000`` and
the three BN-recalibrated models with ``seed_base=40000``.  The means are still
unbiased over draws, but the as-trained and BN rows are not paired at the
window level, so the 2x2 table cannot use paired tests and cannot feed the
switching analysis.

This script reloads every saved ``.pt`` (written before any recalibration),
rebuilds the same fold, and evaluates four configurations on the *same* noise
draws (``seed_base=50000``):

    at          training-time BN statistics, no recalibration
    bn_clean    BN statistics estimated on the clean training windows
    bn_noise_0  BN statistics estimated on 0 dB noisy training windows
    bn_noise_4  BN statistics estimated on 4 dB noisy training windows

By default only clean and 0 dB test conditions are evaluated: clean detects
whether recalibration damages clean accuracy, and 0 dB is the point used by
the 2x2 BN table and switching analysis.  The full five-SNR curves already
exist in the M1 ``pre`` results.  Pass ``--snrs 0 4`` to extend the grid.

No retraining.  Outputs go to ``results/journal/aligned``.

Usage
-----
    python reeval_aligned.py --jobs 4 --threads 1
    python reeval_aligned.py --folds load0 --variants noPad --conditions noise
    python reeval_aligned.py --snrs 0 4
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_v3 import (  # noqa: E402
    BASE, build, build_split, evaluate_condition, noisy_calibration_set,
    recalibrate_bn,
)

RESULTS = BASE / 'results' / 'journal'
RUNS = RESULTS / 'runs'
OUT = RESULTS / 'aligned'
EVAL_SEED = 50000
EVAL_SNRS = [0.0]
KEYS = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
        'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
SECTIONS = ['at', 'bn_clean', 'bn_noise_0', 'bn_noise_4']


def _fresh(variant: str, pt: Path):
    model = build(variant)
    state = torch.load(pt, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def eval_one(stem: str, out_dir: Path, threads: int, force: bool = False,
             snrs=None):
    torch.set_num_threads(threads)
    snrs = EVAL_SNRS if snrs is None else snrs
    meta = json.loads((RUNS / f'{stem}.json').read_text())
    out_path = out_dir / f'{stem}.json'
    npz_path = out_dir / 'preds' / f'{stem}.npz'
    if out_path.exists() and not force:
        if json.loads(out_path.read_text()).get('complete'):
            return stem, 'skip'

    pt = RUNS / f'{stem}.pt'
    if not pt.exists():
        return stem, 'missing-weights'

    s = build_split(meta['fold'])
    xtr, xte, yte = s['xtr'], s['xte'], s['yte']
    variant = meta['variant']
    payload = {'fold': meta['fold'], 'variant': variant,
               'cond': meta['cond'], 'seed': meta['seed'],
               'eval_seed': EVAL_SEED, 'eval_snrs': snrs,
               'sections': {}, 'complete': True}
    arrays = {'y_test': yte.numpy().astype(np.int8)}

    configs = [('at', None), ('bn_clean', xtr)]
    for snr in (0.0, 4.0):
        configs.append((f'bn_noise_{snr:g}',
                        noisy_calibration_set(xtr, snr)))

    for name, cal in configs:
        model = _fresh(variant, pt)
        if cal is not None:
            recalibrate_bn(model, cal)
        metrics, corr = evaluate_condition(model, xte, yte,
                                           seed_base=EVAL_SEED, snrs=snrs)
        payload['sections'][name] = metrics
        for k, v in corr.items():
            arrays[f'{name}_{k}'] = v

    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_path)
    np.savez_compressed(npz_path, **arrays)
    return stem, 'done'


def _stats(values):
    a = np.asarray(values, dtype=float)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    h = 0.0
    if len(a) > 1:
        from scipy.stats import t
        h = float(t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a)))
    return {'mean': round(m, 2), 'sd': round(sd, 2),
            'ci95_low': round(m - h, 2), 'ci95_high': round(m + h, 2)}


def aggregate(out_dir: Path):
    rows = []
    for p in sorted(out_dir.glob('*.json')):
        if p.name == 'aligned_summary.json':
            continue
        d = json.loads(p.read_text())
        if 'sections' in d and 'variant' in d:
            rows.append(d)
    if not rows:
        return None
    variants = sorted({r['variant'] for r in rows})
    conditions = sorted({r['cond'] for r in rows})
    sections = [s for s in SECTIONS if s in rows[0]['sections']]
    keys = [k for k in KEYS
            if all(k in rows[0]['sections'][s] for s in sections)]
    summary = {'eval_seed': EVAL_SEED,
               'eval_snrs': rows[0].get('eval_snrs', EVAL_SNRS),
               'runs': len(rows), 'groups': {}}

    for variant in variants:
        for cond in conditions:
            q = [r for r in rows
                 if r['variant'] == variant and r['cond'] == cond]
            if not q:
                continue
            folds = sorted({r['fold'] for r in q})
            lobo = [f for f in folds if f.startswith('load')]
            g = {'n': len(q), 'folds': folds, 'sections': {}}
            for section in sections:
                if section not in q[0]['sections']:
                    continue
                sec = {}
                for k in keys:
                    fold_means = {
                        f: float(np.mean([r['sections'][section][k] for r in q
                                          if r['fold'] == f]))
                        for f in folds
                    }
                    seed_sds = [
                        float(np.std([r['sections'][section][k] for r in q
                                      if r['fold'] == f], ddof=1))
                        if sum(r['fold'] == f for r in q) > 1 else 0.0
                        for f in lobo
                    ]
                    entry = _stats([r['sections'][section][k] for r in q])
                    entry['fold_means'] = {f: round(v, 2)
                                           for f, v in fold_means.items()}
                    entry['mean_lobo'] = round(
                        float(np.mean([fold_means[f] for f in lobo])), 2
                    ) if lobo else None
                    entry['sd_across_lobo_folds'] = round(
                        float(np.std([fold_means[f] for f in lobo], ddof=1)), 2
                    ) if len(lobo) > 1 else 0.0
                    entry['random_mean'] = (
                        round(fold_means['random'], 2)
                        if 'random' in fold_means else None)
                    entry['mean_within_fold_sd'] = round(
                        float(np.mean(seed_sds)), 2) if seed_sds else 0.0
                    sec[k] = entry
                g['sections'][section] = sec
            summary['groups'][f'{variant}|{cond}'] = g

    (out_dir / 'aligned_summary.json').write_text(
        json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', default=None)
    ap.add_argument('--variants', nargs='+', default=None)
    ap.add_argument('--conditions', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--snrs', nargs='+', type=float, default=EVAL_SNRS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preds').mkdir(parents=True, exist_ok=True)

    metas = []
    for p in sorted(RUNS.glob('*.json')):
        m = json.loads(p.read_text())
        if not m.get('complete'):
            continue
        if not (RUNS / (p.stem + '.pt')).exists():
            continue
        if args.folds and m['fold'] not in args.folds:
            continue
        if args.variants and m['variant'] not in args.variants:
            continue
        if args.conditions and m['cond'] not in args.conditions:
            continue
        if args.seeds and m['seed'] not in args.seeds:
            continue
        metas.append(p.stem)

    print(f'queued {len(metas)} runs')
    if not args.aggregate_only:
        if args.jobs == 1:
            for stem in metas:
                print(eval_one(stem, OUT, args.threads, args.force,
                               args.snrs), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(eval_one, s, OUT, args.threads, args.force,
                                  args.snrs) for s in metas]
                for f in as_completed(futs):
                    print(f.result(), flush=True)

    summary = aggregate(OUT)
    if summary is None:
        print('no aligned results yet')
        return
    print(json.dumps({'runs': summary['runs'],
                      'groups': len(summary['groups'])}, indent=1))


if __name__ == '__main__':
    main()
