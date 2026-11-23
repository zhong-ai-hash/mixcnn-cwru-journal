"""Protocol v2 for the bearing-fault paper.

Design choices that differ from the legacy scripts:
  * one fixed outer train/validation/test split;
  * StandardScaler fitted on the training split only;
  * five initialization seeds on the same split;
  * per-window AWGN during training and evaluation;
  * clean/noise BN statistics crossed with clean/noise-trained weights;
  * five SNR levels and ten noise draws per level.

Each run is written atomically under results_v2/runs.  A rerun skips a
completed (variant, condition, seed) run unless --force is given.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from variants import build

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / 'data'
DEFAULT_OUT = ROOT.parent / 'results_v2'

SPLIT_SEED = 2026
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ['base', 'nores', 'noPad', 'soft', 'multi_pad', 'multi']
CONDITIONS = ['clean', 'noise']
SNRS = [-4.0, 0.0, 4.0, 8.0, 12.0]
BN_SNRS = [0.0, 4.0]
DRAWS = 10
EPOCHS = 20
BATCH = 32
LR = 1e-4
SNR_TRAIN = (-4.0, 10.0)

_SPLIT = None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_split():
    """Load arrays once per worker and build the fixed split."""
    global _SPLIT
    if _SPLIT is not None:
        return _SPLIT

    data_path = DATA_DIR / 'cwru_data.npy'
    label_path = DATA_DIR / 'cwru_label.npy'
    X_raw = np.load(data_path)
    y = np.load(label_path).astype(np.int64)

    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, random_state=SPLIT_SEED, stratify=y)
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=2.0 / 3.0, random_state=SPLIT_SEED,
        stratify=y[idx_tmp])

    scaler = StandardScaler().fit(X_raw[idx_tr])
    X = scaler.transform(X_raw).astype(np.float32)

    def as_tensor(ii, add_channel=False):
        z = X[ii]
        if add_channel:
            z = np.expand_dims(z, axis=1)
        return torch.from_numpy(z)

    _SPLIT = {
        'xtr': as_tensor(idx_tr, True),
        'ytr': torch.from_numpy(y[idx_tr]),
        'xva': as_tensor(idx_va, True),
        'yva': torch.from_numpy(y[idx_va]),
        'xte': as_tensor(idx_te, True),
        'yte': torch.from_numpy(y[idx_te]),
        'meta': {
            'split_seed': SPLIT_SEED,
            'n_total': int(len(y)),
            'n_train': int(len(idx_tr)),
            'n_val': int(len(idx_va)),
            'n_test': int(len(idx_te)),
            'class_counts_train': np.bincount(y[idx_tr], minlength=10).tolist(),
            'class_counts_val': np.bincount(y[idx_va], minlength=10).tolist(),
            'class_counts_test': np.bincount(y[idx_te], minlength=10).tolist(),
            'scaler': 'StandardScaler fit on train split only; per-position',
            'data_sha256': _sha256(data_path),
            'label_sha256': _sha256(label_path),
        },
    }
    return _SPLIT


def add_noise(x, snr_db, generator=None):
    """Add window-wise AWGN.

    `snr_db` may be a scalar or a vector with one value per window.  The
    signal power is computed per window, so the requested SNR is verifiable
    sample by sample rather than only at batch level.
    """
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
    """Estimate BN running statistics on an unlabeled calibration set."""
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
def evaluate_condition(model, x, y, seed_base=10000):
    """Evaluate clean and every noise level, returning metrics and per-window correctness."""
    y_np = y.numpy()
    metrics = {}
    correctness = {}

    pred = predict(model, x)
    metrics['acc_clean'] = round(float(accuracy_score(y_np, pred)) * 100, 2)
    metrics['f1_clean'] = round(float(f1_score(y_np, pred, average='macro')) * 100, 2)
    correctness['clean'] = (pred == y_np).astype(np.int8)

    for snr in SNRS:
        accs = []
        rows = []
        for d in range(DRAWS):
            gen = torch.Generator().manual_seed(seed_base + int(round((snr + 20) * 100)) + d)
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
        gen = torch.Generator().manual_seed(seed_base + int(round((snr + 20) * 100)) + d)
        parts.append(add_noise(xtr, float(snr), generator=gen))
    return torch.cat(parts, dim=0)


def flops_for(model):
    """Small wrapper kept separate so a profiling failure cannot kill a run."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, 1, 1024)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(x)
        return int(counter.get_total_flops())
    except Exception:
        return None


def train_model(variant, seed, cond):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = load_split()
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


def run_one(variant, cond, seed, out_dir, threads):
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    run_dir = out_dir / 'runs'
    pred_dir = out_dir / 'preds'
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    stem = f'{variant}_{cond}_seed{seed}'
    out_path = run_dir / f'{stem}.json'
    pred_path = pred_dir / f'{stem}.npz'
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old.get('complete') and pred_path.exists():
            return stem, 'skip'

    s = load_split()
    xtr, xte, yte = s['xtr'], s['xte'], s['yte']
    model, secs = train_model(variant, seed, cond)
    params = sum(p.numel() for p in model.parameters())
    flops = flops_for(model)

    pre, pre_correct = evaluate_condition(model, xte, yte)
    payload = {
        'variant': variant,
        'cond': cond,
        'seed': seed,
        'split_seed': SPLIT_SEED,
        'n_train': int(len(xtr)),
        'n_test': int(len(xte)),
        'params': int(params),
        'flops': flops,
        'secs': secs,
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
        bn_clean, c = evaluate_condition(model, xte, yte, seed_base=20000)
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


def _metric_ci(values):
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
    runs = [json.loads(p.read_text()) for p in sorted((out_dir / 'runs').glob('*.json'))]
    summary = {'protocol': {
        'split_seed': SPLIT_SEED,
        'seeds': SEEDS,
        'variants': VARIANTS,
        'conditions': CONDITIONS,
        'snrs': SNRS,
        'bn_snrs': BN_SNRS,
        'draws': DRAWS,
        'epochs': EPOCHS,
        'batch': BATCH,
        'lr': LR,
    }, 'groups': {}}

    keys = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
            'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
    for variant in VARIANTS:
        for cond in CONDITIONS:
            q = [r for r in runs if r['variant'] == variant and r['cond'] == cond]
            if not q:
                continue
            g = {'n': len(q), 'seeds': sorted(r['seed'] for r in q), 'sections': {}}
            for section in ['pre', 'bn_clean', 'bn_noise_0', 'bn_noise_4']:
                if section == 'bn_clean' and q[0].get('bn_clean') is None:
                    continue
                if q[0].get(section) is None:
                    continue
                g['sections'][section] = {
                    k: _metric_ci([r[section][k] for r in q]) for k in keys
                }
            summary['groups'][f'{variant}|{cond}'] = g
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        split = load_split()
        (out_dir / 'split_meta.json').write_text(json.dumps(split['meta'], indent=1))

        tasks = [(v, c, s, str(out_dir), args.threads)
                 for v in args.variants for c in args.conditions for s in args.seeds]
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
