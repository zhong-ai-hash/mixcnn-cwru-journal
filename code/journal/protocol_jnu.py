"""JNU external validation under a leave-one-speed-out protocol.

Default protocol
----------------
Train on 600 and 800 rpm, test on 1000 rpm.  The scaler is fit on training
windows only.  One fixed outer test set is used for all runs.  Windows are
4096 points at 50 kHz (82 ms), close to the 85 ms CWRU window.

The script also supports the other two held-out speeds for a three-fold
robustness check.  It deliberately keeps the model, optimizer, epoch budget,
noise-training rule, and evaluation grid identical to the CWRU protocol.

Outputs
-------
``results/journal/jnu/``
    runs/*.json, runs/*.pt, preds/*.npz
    split_meta_<speed>.json
    summary.json

Usage
-----
    python protocol_jnu.py --dry-run
    python protocol_jnu.py --jobs 4 --threads 1 --save-state
    python protocol_jnu.py --folds 600 800 1000 --seeds 0 1 2
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_v3 import (  # noqa: E402
    BATCH, DRAWS, EPOCHS, LR, SNR_TRAIN, SNRS, add_noise,
)
from variants import build  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / 'data' / 'journal' / 'jnu_windows.npz'
META = BASE / 'data' / 'journal' / 'jnu_meta.json'
OUT = BASE / 'results' / 'journal' / 'jnu'
SPLIT_SEED = 2026
SEEDS = [0, 1, 2]
VARIANTS = ['base', 'noPad']
CONDITIONS = ['clean', 'noise']
FOLDS = [1000]
ALL_FOLDS = [1000, 800, 600]
_CACHE = {}


def load_meta():
    return json.loads(META.read_text())


def build_split(held_out):
    held_out = int(held_out)
    if held_out in _CACHE:
        return _CACHE[held_out]
    d = np.load(DATA)
    X_raw = d['X'].astype(np.float64)
    y = d['y'].astype(np.int64)
    speed = d['speed'].astype(np.int64)
    file_id = d['file_id'].astype(np.int64)
    idx = np.arange(len(y))
    idx_te = idx[speed == held_out]
    rest = idx[speed != held_out]
    idx_tr, idx_va = train_test_split(
        rest, test_size=0.10, random_state=SPLIT_SEED, stratify=y[rest])
    if len(idx_te) == 0:
        raise RuntimeError(f'no JNU windows at held-out speed {held_out}')

    scaler = StandardScaler().fit(X_raw[idx_tr])
    X = scaler.transform(X_raw).astype(np.float32)

    def tensor(ii):
        return torch.from_numpy(np.expand_dims(X[ii], axis=1))

    split = {
        'xtr': tensor(idx_tr), 'ytr': torch.from_numpy(y[idx_tr]),
        'xva': tensor(idx_va), 'yva': torch.from_numpy(y[idx_va]),
        'xte': tensor(idx_te), 'yte': torch.from_numpy(y[idx_te]),
        'meta': {
            'fold': str(held_out),
            'held_out_speed': held_out,
            'split_seed': SPLIT_SEED,
            'n_train': int(len(idx_tr)),
            'n_val': int(len(idx_va)),
            'n_test': int(len(idx_te)),
            'class_counts_train': np.bincount(y[idx_tr], minlength=4).tolist(),
            'class_counts_val': np.bincount(y[idx_va], minlength=4).tolist(),
            'class_counts_test': np.bincount(y[idx_te], minlength=4).tolist(),
            'train_speeds': sorted(set(speed[idx_tr].tolist())),
            'test_speeds': sorted(set(speed[idx_te].tolist())),
            'train_recording_ids': sorted(set(file_id[idx_tr].tolist())),
            'test_recording_ids': sorted(set(file_id[idx_te].tolist())),
            'train_test_recording_overlap': sorted(
                set(file_id[idx_tr].tolist()) & set(file_id[idx_te].tolist())),
            'window': int(X_raw.shape[1]),
            'sampling_rate_hz': load_meta()['sampling_rate_hz'],
            'scaler': 'StandardScaler fit on the training split only',
        },
    }
    _CACHE[held_out] = split
    return split


@torch.no_grad()
def evaluate(model, x, y, seed_base=10000):
    model.eval()
    y_np = y.numpy()
    pred = model(x).argmax(1).numpy()
    metrics = {
        'acc_clean': round(float(accuracy_score(y_np, pred)) * 100, 2),
        'f1_clean': round(float(f1_score(y_np, pred, average='macro')) * 100, 2),
    }
    arrays = {'clean': (pred == y_np).astype(np.int8)}
    for snr in SNRS:
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


def train(variant, cond, seed, held_out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = build_split(held_out)
    xtr, ytr = s['xtr'], s['ytr']
    model = build(variant, n_cls=4)
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


def flops_for_input(model, length):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, 1, length)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(x)
        return int(counter.get_total_flops())
    except Exception:
        return None


def run_one(variant, cond, seed, held_out, out_dir, threads, save_state,
            force=False):
    torch.set_num_threads(threads)
    run_dir = Path(out_dir) / 'runs'
    pred_dir = Path(out_dir) / 'preds'
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f'speed{held_out}_{variant}_{cond}_seed{seed}'
    out_path = run_dir / f'{stem}.json'
    pred_path = pred_dir / f'{stem}.npz'
    state_path = run_dir / f'{stem}.pt'
    if out_path.exists() and not force:
        old = json.loads(out_path.read_text())
        if (old.get('complete') and pred_path.exists()
                and (not save_state or state_path.exists())):
            return stem, 'skip'

    s = build_split(held_out)
    model, secs = train(variant, cond, seed, held_out)
    if save_state:
        torch.save(model.state_dict(), state_path)
    metrics, arrays = evaluate(model, s['xte'], s['yte'])
    payload = {
        'fold': str(held_out), 'held_out_speed': int(held_out),
        'variant': variant, 'cond': cond, 'seed': seed,
        'n_train': int(len(s['xtr'])), 'n_test': int(len(s['xte'])),
        'params': int(sum(p.numel() for p in model.parameters())),
        'flops': flops_for_input(model, int(s['xte'].shape[-1])),
        'secs': secs, 'pre': metrics,
        'test_recording_overlap': len(
            s['meta']['train_test_recording_overlap']),
        'complete': True,
    }
    arrays['y_test'] = s['yte'].numpy().astype(np.int8)
    np.savez_compressed(pred_path, **arrays)
    out_path.write_text(json.dumps(payload, indent=1))
    return stem, 'done'


def _stats(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    h = 0.0
    if len(a) > 1:
        from scipy.stats import t
        h = float(t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a)))
    return {'mean': round(mean, 2), 'sd': round(sd, 2),
            'ci95_low': round(mean - h, 2), 'ci95_high': round(mean + h, 2)}


def aggregate(out_dir):
    out_dir = Path(out_dir)
    rows = [json.loads(p.read_text())
            for p in sorted((out_dir / 'runs').glob('*.json'))]
    keys = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
            'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
    summary = {'runs': len(rows), 'groups': {}}
    for variant in sorted({r['variant'] for r in rows}):
        for cond in sorted({r['cond'] for r in rows if r['variant'] == variant}):
            q = [r for r in rows
                 if r['variant'] == variant and r['cond'] == cond]
            folds = sorted({r['fold'] for r in q})
            g = {'n': len(q), 'folds': folds, 'metrics': {}}
            for k in keys:
                fold_means = {
                    f: float(np.mean([r['pre'][k] for r in q
                                      if r['fold'] == f]))
                    for f in folds
                }
                entry = _stats([r['pre'][k] for r in q])
                entry['fold_means'] = {f: round(v, 2)
                                       for f, v in fold_means.items()}
                entry['mean_speed'] = round(
                    float(np.mean(list(fold_means.values()))), 2)
                entry['sd_across_speeds'] = round(
                    float(np.std(list(fold_means.values()), ddof=1)), 2
                ) if len(folds) > 1 else 0.0
                g['metrics'][k] = entry
            summary['groups'][f'{variant}|{cond}'] = g
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=FOLDS,
                    choices=ALL_FOLDS)
    ap.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
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
        for fold in args.folds:
            s = build_split(fold)
            m = s['meta']
            (out_dir / f'split_meta_{fold}.json').write_text(
                json.dumps(m, indent=1))
            print(f"speed{fold}: train={m['n_train']:4d} val={m['n_val']:3d} "
                  f"test={m['n_test']:3d} train_speeds={m['train_speeds']} "
                  f"test_speeds={m['test_speeds']}")
        queued = (len(args.folds) * len(args.variants)
                  * len(args.conditions) * len(args.seeds))
        print(f"runs queued: {queued}")
        return

    if not args.aggregate_only:
        for fold in args.folds:
            (out_dir / f'split_meta_{fold}.json').write_text(
                json.dumps(build_split(fold)['meta'], indent=1))
        tasks = [(v, c, s, f, str(out_dir), args.threads, args.save_state,
                  args.force)
                 for f in args.folds for v in args.variants
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
                      'groups': len(summary['groups'])}, indent=1))


if __name__ == '__main__':
    main()
