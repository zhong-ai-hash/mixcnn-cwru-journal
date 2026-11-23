"""Re-evaluate the five-SNR curve of every M1 checkpoint in one environment.

The M1 matrix was produced under the original Python 3.13 environment.  The
aligned BN re-evaluation and all later experiments use the pinned Python 3.12
environment, where the same checkpoint can differ by 1--3 points on individual
seeds.  Mixing the two sets in one table is not defensible, so this script
recomputes the as-trained curve (clean + every SNR) from the saved ``.pt``
files without retraining.

Outputs ``results/journal/curve/<stem>.json`` and a machine-readable
``curve/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_v3 import (  # noqa: E402
    SNRS, build, build_split, evaluate_condition,
)

BASE = Path(__file__).resolve().parents[2]
RUNS = BASE / 'results' / 'journal' / 'runs'
OUT = BASE / 'results' / 'journal' / 'curve'
EVAL_SEED = 50000
METRICS = ['acc_clean', 'f1_clean'] + [
    f'acc_snr_{snr:g}' for snr in SNRS
] + [f'std_snr_{snr:g}' for snr in SNRS]


def stems(folds, variants, conditions, seeds):
    for fold in folds:
        for variant in variants:
            for cond in conditions:
                for seed in seeds:
                    yield f'{fold}_{variant}_{cond}_seed{seed}'


def eval_one(stem: str, threads: int, force: bool = False):
    torch.set_num_threads(threads)
    src = RUNS / f'{stem}.json'
    pt = RUNS / f'{stem}.pt'
    out_path = OUT / f'{stem}.json'
    if not src.exists() or not pt.exists():
        return stem, 'missing'
    if out_path.exists() and not force:
        if json.loads(out_path.read_text()).get('complete'):
            return stem, 'skip'

    meta = json.loads(src.read_text())
    split = build_split(meta['fold'])
    model = build(meta['variant'])
    model.load_state_dict(torch.load(pt, map_location='cpu', weights_only=True))
    model.eval()

    t0 = time.time()
    metrics, _ = evaluate_condition(
        model, split['xte'], split['yte'], seed_base=EVAL_SEED, snrs=SNRS)
    payload = {
        'fold': meta['fold'], 'variant': meta['variant'],
        'cond': meta['cond'], 'seed': meta['seed'],
        'eval_seed': EVAL_SEED, 'eval_snrs': SNRS,
        'n_test': int(len(split['xte'])), 'pre': metrics,
        'secs': round(time.time() - t0, 1), 'complete': True,
    }
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_path)
    return stem, 'done'


def _mean_sd(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return round(float(a.mean()), 2), round(sd, 2)


def aggregate():
    rows = []
    for path in sorted(OUT.glob('*.json')):
        if path.name == 'summary.json':
            continue
        payload = json.loads(path.read_text())
        if payload.get('complete'):
            rows.append(payload)
    groups = {}
    for payload in rows:
        key = f"{payload['variant']}|{payload['cond']}"
        group = groups.setdefault(key, {'runs': 0, 'folds': {}})
        group['runs'] += 1
        group['folds'].setdefault(payload['fold'], []).append(payload['pre'])

    summary = {'runs': len(rows), 'eval_seed': EVAL_SEED,
               'eval_snrs': SNRS, 'groups': {}}
    for key, group in sorted(groups.items()):
        metrics = {}
        for name in METRICS:
            per_fold = {}
            for fold, items in group['folds'].items():
                per_fold[fold] = _mean_sd([it[name] for it in items])[0]
            lobo = [v for f, v in per_fold.items() if f != 'random']
            mean_lobo, sd_lobo = _mean_sd(lobo)
            entry = {
                'fold_means': per_fold,
                'mean_lobo': mean_lobo,
                'sd_across_lobo_folds': sd_lobo,
            }
            if 'random' in per_fold:
                entry['random_mean'] = per_fold['random']
                entry['leakage_gap'] = round(
                    per_fold['random'] - mean_lobo, 2)
            metrics[name] = entry
        summary['groups'][key] = {
            'runs': group['runs'],
            'metrics': metrics,
        }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+',
                    default=['load0', 'load1', 'load2', 'load3', 'random'])
    ap.add_argument('--variants', nargs='+', default=['base', 'noPad'])
    ap.add_argument('--conditions', nargs='+', default=['clean', 'noise'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        tasks = list(stems(args.folds, args.variants, args.conditions,
                           args.seeds))
        done = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(eval_one, stem, args.threads, args.force): stem
                       for stem in tasks}
            for future in as_completed(futures):
                stem, status = future.result()
                done += 1
                print(f'[{done}/{len(tasks)}] {stem} {status} '
                      f'({time.time() - t0:.0f}s)', flush=True)
    summary = aggregate()
    print(json.dumps({'runs': summary['runs'],
                      'groups': len(summary['groups'])}, indent=1))


if __name__ == '__main__':
    main()
