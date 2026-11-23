"""Protocol v3: recording-level (leave-one-load-out) evaluation.

Differences from protocol v2
----------------------------
* The split is by load, not by random window.  ``load0``--``load3`` hold out
  every window recorded at that load, so no recording contributes to both
  training and test.  ``random`` reproduces the conference protocol on the
  same windows and exists only to quantify the leakage gap.
* Three initialization seeds per fold.  Five folds x three seeds gives both a
  split variance and a seed variance, which v2 could not separate.
* Every run records its fold, split sizes, and per-class counts.

Each run is written atomically under ``results/journal/runs``.  A completed
(fold, variant, condition, seed) run is skipped unless --force is given.

Usage
-----
    python protocol_v3.py --dry-run
    python protocol_v3.py --folds load0 load1 load2 load3 random \\
        --variants base noPad --conditions clean noise --seeds 0 1 2 \\
        --jobs 4 --threads 1
"""
from __future__ import annotations

import argparse
import hashlib
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

# variants.py lives in code/, one level above this journal/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from variants import build  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'data' / 'journal'
DEFAULT_OUT = BASE / 'results' / 'journal'

SPLIT_SEED = 2026
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ['base', 'noPad']
CONDITIONS = ['clean', 'noise']
FOLDS = ['load0', 'load1', 'load2', 'load3', 'random']
SNRS = [-4.0, 0.0, 4.0, 8.0, 12.0]
BN_SNRS = [0.0, 4.0]
DRAWS = 10
EPOCHS = 20
BATCH = 32
LR = 1e-4
SNR_TRAIN = (-4.0, 10.0)

_CACHE: dict[str, dict] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_arrays():
    npz = DATA_DIR / 'cwru_raw_windows.npz'
    if not npz.exists():
        raise SystemExit(
            f'{npz} not found. Run raw_cwru.py first and check its metadata.')
    d = np.load(npz)
    return {k: d[k] for k in d.files}, npz


def build_split(fold: str):
    """Return tensors for one fold.  Cached per process."""
    if fold in _CACHE:
        return _CACHE[fold]

    d, npz = load_arrays()
    X_raw = d['X'].astype(np.float64)
    y = d['y'].astype(np.int64)
    load = d['load'].astype(np.int64)
    idx = np.arange(len(y))

    if fold == 'random':
        idx_tr, idx_tmp = train_test_split(
            idx, test_size=0.30, random_state=SPLIT_SEED, stratify=y)
        idx_va, idx_te = train_test_split(
            idx_tmp, test_size=2.0 / 3.0, random_state=SPLIT_SEED,
            stratify=y[idx_tmp])
        held_out = None
    elif fold.startswith('load'):
        k = int(fold[-1])
        idx_te = idx[load == k]
        rest = idx[load != k]
        idx_tr, idx_va = train_test_split(
            rest, test_size=0.10, random_state=SPLIT_SEED, stratify=y[rest])
        held_out = k
    else:
        raise ValueError(f'unknown fold: {fold}')

    scaler = StandardScaler().fit(X_raw[idx_tr])
    X = scaler.transform(X_raw).astype(np.float32)

    def tensor(ii):
        return torch.from_numpy(np.expand_dims(X[ii], axis=1))

    split = {
        'xtr': tensor(idx_tr), 'ytr': torch.from_numpy(y[idx_tr]),
        'xva': tensor(idx_va), 'yva': torch.from_numpy(y[idx_va]),
        'xte': tensor(idx_te), 'yte': torch.from_numpy(y[idx_te]),
        'meta': {
            'fold': fold,
            'held_out_load': held_out,
            'split_seed': SPLIT_SEED,
            'n_total': int(len(y)),
            'n_train': int(len(idx_tr)),
            'n_val': int(len(idx_va)),
            'n_test': int(len(idx_te)),
            'class_counts_train': np.bincount(y[idx_tr], minlength=10).tolist(),
            'class_counts_val': np.bincount(y[idx_va], minlength=10).tolist(),
            'class_counts_test': np.bincount(y[idx_te], minlength=10).tolist(),
            'train_loads': sorted(set(load[idx_tr].tolist())),
            'test_loads': sorted(set(load[idx_te].tolist())),
            'scaler': 'StandardScaler fit on the training split only',
            'data_sha256': _sha256(npz),
        },
    }
    _CACHE[fold] = split
    return split


def add_noise(x, snr_db, generator=None):
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(snr_db, dtype=x.dtype)
    if snr_db.ndim == 0:
        snr_db = snr_db.expand(x.shape[0])
    snr_db = snr_db.reshape(-1, 1, 1)
    p = x.pow(2).mean(dim=2, keepdim=True).sqrt()
    sigma = p / (10 ** (snr_db / 20.0))
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    return x + noise * sigma


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None


@torch.no_grad()
def recalibrate_bn(model, x, bs=64):
    reset_bn(model)
    model.train()
    for i in range(0, len(x), bs):
        model(x[i:i + bs])
    model.eval()


@torch.no_grad()
def predict(model, x, bs=64):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs]).argmax(1))
    return torch.cat(out).numpy()


@torch.no_grad()
def evaluate_condition(model, x, y, seed_base=10000, snrs=None):
    snrs = SNRS if snrs is None else snrs
    y_np = y.numpy()
    metrics, correctness = {}, {}

    pred = predict(model, x)
    metrics['acc_clean'] = round(float(accuracy_score(y_np, pred)) * 100, 2)
    metrics['f1_clean'] = round(float(f1_score(y_np, pred, average='macro')) * 100, 2)
    correctness['clean'] = (pred == y_np).astype(np.int8)

    for snr in snrs:
        accs, rows = [], []
        for d in range(DRAWS):
            gen = torch.Generator().manual_seed(
                seed_base + int(round((snr + 20) * 100)) + d)
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
        correctness[f'snr_{key}'] = np.stack(rows, axis=0)
    return metrics, correctness


def noisy_calibration_set(xtr, snr, draws=5, seed_base=30000):
    parts = []
    for d in range(draws):
        gen = torch.Generator().manual_seed(
            seed_base + int(round((snr + 20) * 100)) + d)
        parts.append(add_noise(xtr, float(snr), generator=gen))
    return torch.cat(parts, dim=0)


def flops_for(model):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, 1, 1024)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(x)
        return int(counter.get_total_flops())
    except Exception:
        return None


def train_model(variant, seed, cond, fold):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = build_split(fold)
    xtr, ytr = s['xtr'], s['ytr']
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = len(xtr)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
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


def run_one(fold, variant, cond, seed, out_dir, threads, force=False,
            save_state=False):
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    run_dir, pred_dir = out_dir / 'runs', out_dir / 'preds'
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    stem = f'{fold}_{variant}_{cond}_seed{seed}'
    out_path = run_dir / f'{stem}.json'
    pred_path = pred_dir / f'{stem}.npz'
    state_path = run_dir / f'{stem}.pt'
    if out_path.exists() and not force:
        old = json.loads(out_path.read_text())
        if (old.get('complete') and pred_path.exists()
                and (not save_state or state_path.exists())):
            return stem, 'skip'

    s = build_split(fold)
    xtr, xte, yte = s['xtr'], s['xte'], s['yte']
    model = build(variant)
    resumed = False
    if save_state and state_path.exists() and not force:
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        secs = 0.0
        resumed = True
    else:
        model, secs = train_model(variant, seed, cond, fold)
        # Save the weights with their *training* BN statistics, before any
        # recalibration below mutates the running estimates.
        if save_state:
            tmp_state = state_path.with_suffix('.pt.tmp')
            torch.save(model.state_dict(), tmp_state)
            tmp_state.replace(state_path)
    params = sum(p.numel() for p in model.parameters())
    flops = flops_for(model)

    pre, pre_correct = evaluate_condition(model, xte, yte)
    payload = {
        'fold': fold,
        'variant': variant,
        'cond': cond,
        'seed': seed,
        'split_seed': SPLIT_SEED,
        'n_train': int(len(xtr)),
        'n_test': int(len(xte)),
        'params': int(params),
        'flops': flops,
        'secs': secs,
        'resumed': resumed,
        'pre': pre,
        'bn_clean': None,
        'bn_noise_0': None,
        'bn_noise_4': None,
        'complete': True,
    }
    arrays = {'y_test': yte.numpy().astype(np.int8)}
    for k, v in pre_correct.items():
        arrays[f'pre_{k}'] = v

    if cond == 'noise':
        recalibrate_bn(model, xtr)
        # Same seed base as the noisy BN sections below, so all three BN
        # libraries see identical noise draws and can be paired window by
        # window.
        bn_clean, c = evaluate_condition(model, xte, yte, seed_base=40000)
        payload['bn_clean'] = bn_clean
        for k, v in c.items():
            arrays[f'bn_clean_{k}'] = v

    for snr in BN_SNRS:
        cal = noisy_calibration_set(xtr, snr)
        recalibrate_bn(model, cal)
        bn, c = evaluate_condition(model, xte, yte, seed_base=40000)
        key = f'bn_noise_{snr:g}'
        payload[key] = bn
        for k, v in c.items():
            arrays[f'{key}_{k}'] = v

    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_path)
    np.savez_compressed(pred_path, **arrays)
    return stem, 'done'


def _stats(values):
    a = np.asarray(values, dtype=float)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    if len(a) > 1:
        from scipy.stats import t
        h = float(t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a)))
    else:
        h = 0.0
    return {'mean': round(m, 2), 'sd': round(sd, 2),
            'ci95_low': round(m - h, 2), 'ci95_high': round(m + h, 2)}


def aggregate(out_dir):
    out_dir = Path(out_dir)
    runs = [json.loads(p.read_text())
            for p in sorted((out_dir / 'runs').glob('*.json'))]
    keys = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
            'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
    summary = {'protocol': {
        'split_seed': SPLIT_SEED,
        'seeds': sorted({r['seed'] for r in runs}),
        'variants': sorted({r['variant'] for r in runs}),
        'conditions': sorted({r['cond'] for r in runs}),
        'folds': sorted({r['fold'] for r in runs}),
        'snrs': SNRS, 'bn_snrs': BN_SNRS, 'draws': DRAWS,
        'epochs': EPOCHS, 'batch': BATCH, 'lr': LR,
    }, 'groups': {}}

    for variant in VARIANTS:
        for cond in CONDITIONS:
            q = [r for r in runs if r['variant'] == variant and r['cond'] == cond]
            if not q:
                continue
            folds = sorted({r['fold'] for r in q})
            g = {'n': len(q), 'folds': folds, 'sections': {}}
            for section in ['pre', 'bn_clean', 'bn_noise_0', 'bn_noise_4']:
                if q[0].get(section) is None:
                    continue
                sec = {}
                for k in keys:
                    vals = [r[section][k] for r in q]
                    entry = _stats(vals)
                    # Fold means are kept per fold so the leaky random split
                    # never enters the LOBO variance.
                    fold_means = {
                        f: float(np.mean([r[section][k] for r in q
                                          if r['fold'] == f]))
                        for f in folds
                    }
                    lobo = [f for f in folds if f.startswith('load')]
                    seed_sds = [
                        float(np.std([r[section][k] for r in q if r['fold'] == f],
                                     ddof=1))
                        if sum(r['fold'] == f for r in q) > 1 else 0.0
                        for f in lobo
                    ]
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
                    entry['leakage_gap'] = (
                        round(entry['random_mean'] - entry['mean_lobo'], 2)
                        if entry['random_mean'] is not None
                        and entry['mean_lobo'] is not None else None)
                    entry['mean_within_fold_sd'] = round(
                        float(np.mean(seed_sds)), 2) if seed_sds else 0.0
                    sec[k] = entry
                g['sections'][section] = sec
            summary['groups'][f'{variant}|{cond}'] = g

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', default=FOLDS, choices=FOLDS)
    ap.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--save-state', action='store_true',
                    help='save model weights (needed by bn_switch.py)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for fold in args.folds:
            s = build_split(fold)
            m = s['meta']
            print(f"{fold:7s} train={m['n_train']:5d} val={m['n_val']:4d} "
                  f"test={m['n_test']:5d} train_loads={m['train_loads']} "
                  f"test_loads={m['test_loads']}")
        print(f"runs queued: {len(args.folds) * len(args.variants) * len(args.conditions) * len(args.seeds)}")
        return

    if not args.aggregate_only:
        for fold in args.folds:
            s = build_split(fold)
            (out_dir / f'split_meta_{fold}.json').write_text(
                json.dumps(s['meta'], indent=1))

        tasks = [(f, v, c, s, str(out_dir), args.threads, args.force,
                  args.save_state)
                 for f in args.folds for v in args.variants
                 for c in args.conditions for s in args.seeds]
        if args.jobs == 1:
            for t in tasks:
                print(run_one(*t), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_one, *t) for t in tasks]
                for f in as_completed(futs):
                    print(f.result(), flush=True)

    summary = aggregate(out_dir)
    print(json.dumps({
        'groups': len(summary['groups']),
        'runs': sum(g['n'] for g in summary['groups'].values()),
        'out_dir': str(out_dir),
    }, indent=1), flush=True)


if __name__ == '__main__':
    main()
