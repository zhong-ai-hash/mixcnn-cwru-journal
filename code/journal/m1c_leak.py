"""M1c: isolate same-recording leakage from cross-load generalization.

Both arms use exactly the same test set: the last 25 windows of every load-0
recording (10 recordings x 25 = 250 windows).  Arm A is trained on all 1,500
windows from loads 1--3, so no test recording appears in training.  Arm B
replaces 250 of those windows with the first 25 windows of each load-0
recording, so every test recording contributes earlier, disjoint windows to
training.  Both arms therefore have 1,500 training windows and the identical
250-window test set; Arm B - Arm A is the net effect of same-recording
leakage, with the load composition change held to the minimum needed for a
balanced replacement.

The 250 donors removed from Arm A are the first 25 windows of each load-3
recording.  This matches the replacement windows by class and by position in
the recording.

Outputs
-------
``results/journal/m1c/``
    runs/*.json, runs/*.pt, preds/*.npz
    split_meta.json
    summary.json

Usage
-----
    python m1c_leak.py --dry-run
    python m1c_leak.py --jobs 4 --threads 1 --save-state
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
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_v3 import (  # noqa: E402
    BASE, BATCH, DRAWS, EPOCHS, LR, SNR_TRAIN, SNRS, add_noise,
)
from variants import build  # noqa: E402

OUT = BASE / 'results' / 'journal' / 'm1c'
DATA = BASE / 'data' / 'journal' / 'cwru_raw_windows.npz'
ARMS = ['no_overlap', 'recording_leak']
VARIANTS = ['base', 'noPad']
CONDITIONS = ['clean', 'noise']
SEEDS = [0, 1, 2, 3, 4]
TEST_LOAD = 0
DONOR_LOAD = 3
HALF = 25
EVAL_SNRS = [0.0]
_CACHE = {}


def load_arrays():
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


def _per_file_order(d, load_value):
    """Return first/last 25 indices for every recording at one load."""
    first, last = [], []
    file_ids = d['file_id'][d['load'] == load_value]
    for fid in sorted(np.unique(file_ids).tolist()):
        idx = np.flatnonzero((d['load'] == load_value) & (d['file_id'] == fid))
        if len(idx) != 2 * HALF:
            raise RuntimeError(
                f'file {fid} at load {load_value} has {len(idx)} windows; '
                f'expected {2 * HALF}')
        first.extend(idx[:HALF].tolist())
        last.extend(idx[HALF:].tolist())
    return np.asarray(first, dtype=np.int64), np.asarray(last, dtype=np.int64)


def build_split(arm):
    if arm in _CACHE:
        return _CACHE[arm]
    d = load_arrays()
    X_raw = d['X'].astype(np.float64)
    y = d['y'].astype(np.int64)
    load = d['load'].astype(np.int64)
    _, idx_test = _per_file_order(d, TEST_LOAD)
    leak_train, _ = _per_file_order(d, TEST_LOAD)
    donor_remove, _ = _per_file_order(d, DONOR_LOAD)

    idx_a = np.flatnonzero(load != TEST_LOAD)
    if arm == 'no_overlap':
        idx_tr = idx_a
    elif arm == 'recording_leak':
        idx_tr = np.setdiff1d(idx_a, donor_remove, assume_unique=True)
        idx_tr = np.concatenate([idx_tr, leak_train])
    else:
        raise ValueError(f'unknown arm: {arm}')

    if len(idx_tr) != 1500:
        raise RuntimeError(f'{arm}: expected 1500 training windows, got {len(idx_tr)}')
    if len(idx_test) != 250:
        raise RuntimeError(f'expected 250 test windows, got {len(idx_test)}')
    if len(np.intersect1d(idx_tr, idx_test)) != 0:
        raise RuntimeError(f'{arm}: train/test index overlap')
    if set(np.unique(y[idx_tr]).tolist()) != set(range(10)):
        raise RuntimeError(f'{arm}: not all classes in training')

    scaler = StandardScaler().fit(X_raw[idx_tr])
    X = scaler.transform(X_raw).astype(np.float32)

    def tensor(ii):
        return torch.from_numpy(np.expand_dims(X[ii], axis=1))

    split = {
        'xtr': tensor(idx_tr), 'ytr': torch.from_numpy(y[idx_tr]),
        'xte': tensor(idx_test), 'yte': torch.from_numpy(y[idx_test]),
        'meta': {
            'arm': arm,
            'test_load': TEST_LOAD,
            'donor_load': DONOR_LOAD,
            'n_train': int(len(idx_tr)),
            'n_test': int(len(idx_test)),
            'class_counts_train': np.bincount(y[idx_tr], minlength=10).tolist(),
            'class_counts_test': np.bincount(y[idx_test], minlength=10).tolist(),
            'train_loads': sorted(set(load[idx_tr].tolist())),
            'test_loads': sorted(set(load[idx_test].tolist())),
            'train_recording_ids': sorted(set(d['file_id'][idx_tr].tolist())),
            'test_recording_ids': sorted(set(d['file_id'][idx_test].tolist())),
            'train_test_recording_overlap': sorted(
                set(d['file_id'][idx_tr].tolist())
                & set(d['file_id'][idx_test].tolist())),
            'leak_windows': int(len(np.intersect1d(idx_tr, leak_train))),
            'scaler': 'StandardScaler fit on each arm training split only',
            'test_last_indices': idx_test.tolist(),
        },
    }
    _CACHE[arm] = split
    return split


@torch.no_grad()
def evaluate(model, x, y, seed_base=10000, snrs=None):
    snrs = EVAL_SNRS if snrs is None else snrs
    model.eval()
    y_np = y.numpy()
    pred = model(x).argmax(1).numpy()
    metrics = {
        'acc_clean': round(float(accuracy_score(y_np, pred)) * 100, 2),
        'f1_clean': round(float(f1_score(y_np, pred, average='macro')) * 100, 2),
    }
    arrays = {'clean': (pred == y_np).astype(np.int8)}
    for snr in snrs:
        accs, rows = [], []
        for draw in range(DRAWS):
            gen = torch.Generator().manual_seed(
                seed_base + int(round((snr + 20) * 100)) + draw)
            preds = []
            for i in range(0, len(x), 64):
                xb = add_noise(x[i:i + 64], float(snr), generator=gen)
                preds.append(model(xb).argmax(1))
            p = torch.cat(preds).numpy()
            accs.append(accuracy_score(y_np, p))
            rows.append((p == y_np).astype(np.int8))
        key = f'{snr:g}'
        metrics[f'acc_snr_{key}'] = round(float(np.mean(accs)) * 100, 2)
        metrics[f'std_snr_{key}'] = round(float(np.std(accs)) * 100, 2)
        arrays[f'snr_{key}'] = np.stack(rows, axis=0)
    return metrics, arrays


def train(arm, variant, cond, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = build_split(arm)
    xtr, ytr = s['xtr'], s['ytr']
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            xb = xtr[idx]
            if cond == 'noise':
                snr = torch.empty(len(idx), dtype=xb.dtype).uniform_(*SNR_TRAIN)
                xb = add_noise(xb, snr)
            opt.zero_grad()
            loss = crit(model(xb), ytr[idx])
            loss.backward()
            opt.step()
    return model, round(time.time() - t0, 1)


def run_one(arm, variant, cond, seed, out_dir, threads, save_state,
            force=False, snrs=None):
    snrs = EVAL_SNRS if snrs is None else snrs
    torch.set_num_threads(threads)
    run_dir = Path(out_dir) / 'runs'
    pred_dir = Path(out_dir) / 'preds'
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{arm}_{variant}_{cond}_seed{seed}'
    out_path = run_dir / f'{stem}.json'
    pred_path = pred_dir / f'{stem}.npz'
    state_path = run_dir / f'{stem}.pt'
    if out_path.exists() and not force:
        old = json.loads(out_path.read_text())
        if (old.get('complete') and pred_path.exists()
                and (not save_state or state_path.exists())):
            return stem, 'skip'

    s = build_split(arm)
    model, secs = train(arm, variant, cond, seed)
    if save_state:
        torch.save(model.state_dict(), state_path)
    metrics, arrays = evaluate(model, s['xte'], s['yte'], snrs=snrs)
    payload = {
        'arm': arm, 'variant': variant, 'cond': cond, 'seed': seed,
        'eval_snrs': snrs,
        'n_train': int(len(s['xtr'])), 'n_test': int(len(s['xte'])),
        'params': int(sum(p.numel() for p in model.parameters())),
        'secs': secs, 'pre': metrics,
        'test_recording_overlap': len(
            s['meta']['train_test_recording_overlap']),
        'complete': True,
    }
    arrays['y_test'] = s['yte'].numpy().astype(np.int8)
    np.savez_compressed(pred_path, **arrays)
    out_path.write_text(json.dumps(payload, indent=1))
    return stem, 'done'


def _mean_sd(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return round(float(a.mean()), 2), round(sd, 2)


def aggregate(out_dir):
    rows = [json.loads(p.read_text())
            for p in sorted((Path(out_dir) / 'runs').glob('*.json'))]
    all_keys = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
                'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
    keys = [k for k in all_keys if all(k in r['pre'] for r in rows)]
    summary = {'runs': len(rows), 'eval_snrs': EVAL_SNRS,
               'effects': {}, 'groups': {}}
    for variant in VARIANTS:
        for cond in CONDITIONS:
            group = [r for r in rows
                     if r['variant'] == variant and r['cond'] == cond]
            if not group:
                continue
            sections = {}
            for arm in ARMS:
                sec = [r for r in group if r['arm'] == arm]
                sections[arm] = {
                    k: dict(zip(('mean', 'sd'), _mean_sd([r['pre'][k] for r in sec])))
                    for k in keys
                }
            summary['groups'][f'{variant}|{cond}'] = {
                'n': len(group), 'sections': sections}
            for k in keys:
                a = {r['seed']: r['pre'][k] for r in group if r['arm'] == ARMS[0]}
                b = {r['seed']: r['pre'][k] for r in group if r['arm'] == ARMS[1]}
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 2:
                    continue
                diffs = np.asarray([b[s] - a[s] for s in seeds], dtype=float)
                mean, sd = _mean_sd(diffs)
                test = ttest_rel([b[s] for s in seeds], [a[s] for s in seeds])
                summary['effects'][f'{variant}|{cond}|{k}'] = {
                    'contrast': 'recording_leak - no_overlap',
                    'seeds': seeds, 'diffs': [round(float(x), 2) for x in diffs],
                    'mean': mean, 'sd': sd,
                    't': round(float(test.statistic), 2),
                    'p': round(float(test.pvalue), 4),
                }
    (Path(out_dir) / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--snrs', nargs='+', type=float, default=EVAL_SNRS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--save-state', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        meta = {}
        for arm in ARMS:
            m = build_split(arm)['meta']
            meta[arm] = m
            print(f"{arm:15s} train={m['n_train']:4d} test={m['n_test']:3d} "
                  f"train_loads={m['train_loads']} test_loads={m['test_loads']} "
                  f"recording_overlap={len(m['train_test_recording_overlap'])} "
                  f"leak_windows={m['leak_windows']}")
        (out_dir / 'split_meta.json').write_text(json.dumps(meta, indent=1))
        print(f"runs queued: {len(ARMS) * len(args.variants) * len(args.conditions) * len(args.seeds)}")
        return

    if not args.aggregate_only:
        meta = {arm: build_split(arm)['meta'] for arm in ARMS}
        (out_dir / 'split_meta.json').write_text(json.dumps(meta, indent=1))
        tasks = [(arm, v, c, s, str(out_dir), args.threads, args.save_state,
                  args.force, args.snrs)
                 for arm in ARMS for v in args.variants
                 for c in args.conditions for s in args.seeds]
        if args.jobs == 1:
            for task in tasks:
                print(run_one(*task), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_one, *task) for task in tasks]
                for fut in as_completed(futs):
                    print(fut.result(), flush=True)

    summary = aggregate(out_dir)
    print(json.dumps({'runs': summary['runs'],
                      'effects': len(summary['effects'])}, indent=1))


if __name__ == '__main__':
    main()
