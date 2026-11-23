"""External baselines under the v3 protocol.

Models
------
wdcnn      Zhang et al. (2017), the standard wide-kernel CNN baseline
resnet1d   1-D ResNet with four basic residual blocks
drsn1d     Zhao et al. (2020) deep residual shrinkage network, CW variant,
           simplified to four blocks with channel-wise thresholds

All three use the same split, optimizer, epoch budget, noise-training rule,
and evaluation grid as MIXCNN in ``protocol_v3.py``.  No per-baseline tuning is
performed; this is stated as a limitation in the paper.

Purpose
-------
The baselines answer two reviewer questions: (1) is the leakage gap specific
to the six MIXCNN variants, or does it appear for unrelated architectures;
(2) are the noisy-training conclusions an artifact of one model family.

Outputs go to ``results/journal/baselines`` with the same run schema as
``protocol_v3.py`` so the analysis scripts can be shared.

Usage
-----
    python baselines.py --dry-run
    python baselines.py --models wdcnn resnet1d --jobs 4 --threads 1
    python baselines.py --models drsn1d --folds load0 random --seeds 0 1 2
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
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_v3 import (  # noqa: E402
    BASE, EPOCHS, BATCH, LR, SNR_TRAIN, SPLIT_SEED, SEEDS, FOLDS,
    add_noise, build_split, evaluate_condition, flops_for,
)

OUT = BASE / 'results' / 'journal' / 'baselines'
MODELS = ['wdcnn', 'resnet1d', 'drsn1d']
CONDITIONS = ['clean', 'noise']


class WDCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=16, padding=24),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(2),
        )
        self.head = nn.Linear(64 * 2, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class ResBlock1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        z = torch.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return torch.relu(x + z)


class ResNet1D(nn.Module):
    def __init__(self, n_classes=10, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.blocks = nn.Sequential(*[ResBlock1d(ch) for _ in range(blocks)])
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        z = self.blocks(self.stem(x))
        return self.head(z.mean(dim=2))


class DRSNBlock(nn.Module):
    """Residual block with channel-wise soft thresholding (DRSN-CW)."""

    def __init__(self, ch):
        super().__init__()
        hidden = max(8, ch // 8)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):
        z = torch.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        a = z.abs().mean(dim=2)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(a))))
        thr = s.unsqueeze(2) * z.abs().mean(dim=2, keepdim=True)
        z = torch.sign(z) * torch.relu(z.abs() - thr)
        return torch.relu(x + z)


class DRSN1D(nn.Module):
    def __init__(self, n_classes=10, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 15, padding=7, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[DRSNBlock(ch) for _ in range(blocks)])
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        z = self.blocks(self.stem(x))
        return self.head(z.mean(dim=2))


def build_model(name: str):
    if name == 'wdcnn':
        return WDCNN()
    if name == 'resnet1d':
        return ResNet1D()
    if name == 'drsn1d':
        return DRSN1D()
    raise ValueError(f'unknown model: {name}')


def train_model(name, seed, cond, fold):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = build_split(fold)
    xtr, ytr = s['xtr'], s['ytr']
    model = build_model(name)
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


def run_one(model_name, fold, cond, seed, out_dir, threads, force=False):
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    run_dir, pred_dir = out_dir / 'runs', out_dir / 'preds'
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    stem = f'{model_name}_{fold}_{cond}_seed{seed}'
    out_path = run_dir / f'{stem}.json'
    pred_path = pred_dir / f'{stem}.npz'
    if out_path.exists() and not force:
        if json.loads(out_path.read_text()).get('complete') and pred_path.exists():
            return stem, 'skip'

    s = build_split(fold)
    xtr, xte, yte = s['xtr'], s['xte'], s['yte']
    model, secs = train_model(model_name, seed, cond, fold)
    params = sum(p.numel() for p in model.parameters())
    flops = flops_for(model)
    pre, corr = evaluate_condition(model, xte, yte)

    payload = {
        'model': model_name, 'fold': fold, 'cond': cond, 'seed': seed,
        'split_seed': SPLIT_SEED, 'n_train': int(len(xtr)),
        'n_test': int(len(xte)), 'params': int(params), 'flops': flops,
        'secs': secs, 'pre': pre, 'complete': True,
    }
    arrays = {'y_test': yte.numpy().astype(np.int8)}
    for k, v in corr.items():
        arrays[f'pre_{k}'] = v

    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_path)
    np.savez_compressed(pred_path, **arrays)
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


def aggregate(out_dir):
    out_dir = Path(out_dir)
    rows = [json.loads(p.read_text())
            for p in sorted((out_dir / 'runs').glob('*.json'))]
    keys = ['acc_clean', 'f1_clean', 'acc_snr_-4', 'acc_snr_0',
            'acc_snr_4', 'acc_snr_8', 'acc_snr_12']
    summary = {'runs': len(rows), 'groups': {}}
    for model_name in sorted({r['model'] for r in rows}):
        for cond in sorted({r['cond'] for r in rows if r['model'] == model_name}):
            q = [r for r in rows
                 if r['model'] == model_name and r['cond'] == cond]
            folds = sorted({r['fold'] for r in q})
            lobo = [f for f in folds if f.startswith('load')]
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
                g['metrics'][k] = entry
            summary['groups'][f'{model_name}|{cond}'] = g
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--folds', nargs='+', default=FOLDS, choices=FOLDS)
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        for name in args.models:
            m = build_model(name)
            print(f'{name:9s} params={sum(p.numel() for p in m.parameters()):>8d} '
                  f'flops={flops_for(m)}')
        print(f'runs queued: {len(args.models) * len(args.folds) * len(args.conditions) * len(args.seeds)}')
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        tasks = [(name, f, c, s, str(out_dir), args.threads, args.force)
                 for name in args.models for f in args.folds
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
    print(json.dumps({'runs': summary['runs'],
                      'groups': len(summary['groups'])}, indent=1))


if __name__ == '__main__':
    main()
