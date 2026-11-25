"""Deployable noise-level detector and BN-bank switch, evaluated offline.

The GPU suite stores per-window correctness for every BN bank and every test
SNR.  This script re-derives a cheap noise-level statistic from the same
standardised test windows, calibrates it on the training split with synthetic
noise, and then asks: if we pick a BN bank from the estimated noise level,
how much accuracy do we lose relative to the oracle bank?

No model is retrained.  Output:

    results/gpu/switch/switch_runs.csv
    results/gpu/switch/switch_summary.json|csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from data import split_for

_CACHE: dict = {}


def features(x: np.ndarray) -> np.ndarray:
    """Scale-invariant noise statistics for (N, L) windows."""
    x = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    eps = 1e-12
    flat = np.exp(np.mean(np.log(power + eps), axis=1)) / (
        np.mean(power, axis=1) + eps)
    cut = max(1, int(power.shape[1] * 0.25))
    hf = power[:, cut:].sum(axis=1) / (power.sum(axis=1) + eps)
    kurt = (x ** 4).mean(axis=1) / ((x ** 2).mean(axis=1) ** 2 + eps)
    return np.stack([np.log(flat + eps), hf, np.log(kurt + eps)], axis=1)


def add_noise_np(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Single-draw noise for calibrating the level estimator."""
    import torch
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(xt.shape, generator=gen, dtype=torch.float32)
    rms = xt.pow(2).mean(dim=2, keepdim=True).sqrt()
    sigma = rms / (10 ** (snr_db / 20.0))
    return (xt + noise * sigma).numpy()


def add_noise_draw(x: np.ndarray, snr_db: float, draw: int, draws: int,
                   seed_base: int) -> np.ndarray:
    """Replay the exact draw the engine used for ``(snr_db, draw)``.

    The engine stacks ``g`` draws into one tensor before adding noise, so the
    RNG stream depends on the group, not on the individual draw.  Reproducing
    the group and slicing it is the only way to compare switch decisions with
    per-window correctness.
    """
    import torch
    from engine import noise_seed
    n = x.shape[0]
    g = max(1, min(int(draws), 8192 // max(n, 1)))
    d0 = (int(draw) // g) * g
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    xs = xt.repeat(g, 1, 1) if g > 1 else xt
    gen = torch.Generator().manual_seed(
        noise_seed(int(seed_base), float(snr_db), d0))
    noise = torch.randn(xs.shape, generator=gen, dtype=torch.float32)
    rms = xs.pow(2).mean(dim=2, keepdim=True).sqrt()
    sigma = rms / (10 ** (snr_db / 20.0))
    noisy = (xs + noise * sigma).numpy()
    return noisy[(int(draw) - d0) * n:(int(draw) - d0 + 1) * n]


def calibrate(xtr: np.ndarray, snrs, eval_seed: int):
    rows, labels = [features(xtr)], ['clean']
    for snr in snrs:
        rows.append(features(add_noise_np(
            xtr, float(snr), eval_seed + int(round((snr + 20) * 100)))))
        labels.append(f'{snr:g}')
    f = np.concatenate(rows, axis=0)
    mu, sd = f.mean(axis=0), f.std(axis=0)
    sd[sd < 1e-8] = 1.0
    f = (f - mu) / sd
    centroids = {}
    start = 0
    for label, block in zip(labels, rows):
        centroids[label] = f[start:start + len(block)].mean(axis=0)
        start += len(block)
    return {'mu': mu, 'sd': sd, 'centroids': centroids,
            'labels': labels}


def estimate(f: np.ndarray, model) -> np.ndarray:
    z = (f - model['mu']) / model['sd']
    names = model['labels']
    cents = np.stack([model['centroids'][n] for n in names])
    d = ((z[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(names)[d.argmin(axis=1)]


def pick_bank(label: str, banks) -> str:
    numeric = sorted(float(b) for b in banks if b != 'clean')
    if label == 'clean' or not numeric:
        return 'clean'
    v = float(label)
    if v >= numeric[-1] + 2:
        return 'clean'
    return f'{min(numeric, key=lambda b: abs(b - v)):g}'


def prepared(dataset: str, split_name: str, snrs, draws: int,
             eval_seed: int) -> dict:
    key = (dataset, split_name, tuple(snrs), int(draws), int(eval_seed))
    if key in _CACHE:
        return _CACHE[key]
    split = split_for(dataset, split_name)
    xtr = split['xtr'].squeeze(1).numpy()
    xte = split['xte'].squeeze(1).numpy()
    cal = calibrate(xtr, snrs, eval_seed)
    est = {}
    for snr in snrs:
        rows = []
        for d in range(draws):
            noisy = add_noise_draw(xte, float(snr), d, draws, eval_seed)
            rows.append(estimate(features(noisy), cal))
        est[f'{snr:g}'] = np.stack(rows)
    if len(_CACHE) >= 3:
        _CACHE.clear()
    _CACHE[key] = est
    return est


def evaluate_run(path: Path, preds: Path) -> dict | None:
    payload = json.loads(path.read_text())
    if not payload.get('complete') or not payload.get('bn_banks'):
        return None
    if len(payload.get('snrs', [])) == 0:
        return None
    data = np.load(preds)
    est_by_snr = prepared(payload['dataset'], payload['split'],
                          payload['snrs'], int(payload.get('draws', 1)),
                          payload.get('eval_seed', 50000))

    banks = payload['bn_banks']
    out = {'dataset': payload['dataset'], 'split': payload['split'],
           'model': payload['model'], 'cond': payload['cond'],
           'seed': payload['seed'], 'banks': banks,
           'bucket_exact': {}, 'bucket_within1': {},
           'deployed': {}, 'oracle': {}, 'fixed': {},
           'switch_agree': {}}
    for snr in payload['snrs']:
        key = f'{snr:g}'
        corr = {b: data[f'bn_{b}_snr_{key}'] for b in banks
                if f'bn_{b}_snr_{key}' in data.files}
        if not corr:
            continue
        bank_list = list(corr)
        stack = np.stack([corr[b] for b in bank_list], axis=0)
        oracle = stack.max(axis=0).mean() * 100
        out['oracle'][key] = round(float(oracle), 2)
        out['fixed'][key] = {
            b: round(float(arr.mean() * 100), 2) for b, arr in corr.items()}

        # Estimated bucket and bank selection are per (draw, window).
        est = est_by_snr[key]
        chosen = np.vectorize(lambda e: pick_bank(e, banks))(est)
        deployed = 0
        agree = 0
        for bi, b in enumerate(bank_list):
            m = chosen == b
            deployed += corr[b][m].sum()
            agree += ((stack.argmax(axis=0) == bi) & m).sum()
        n = stack.shape[1] * stack.shape[2]
        out['deployed'][key] = round(float(deployed / n * 100), 2)
        out['switch_agree'][key] = round(float(agree / n), 3)
        out['bucket_exact'][key] = round(float(np.mean(
            est == f'{snr:g}')), 3)
        numeric = np.full(est.shape, 99.0)
        mask = est != 'clean'
        numeric[mask] = est[mask].astype(float)
        out['bucket_within1'][key] = round(float(np.mean(
            np.abs(numeric - float(snr)) <= 4)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suites', nargs='+',
                    default=['main', 'baselines', 'paderborn', 'seu'])
    ap.add_argument('--out', default=str(RESULTS / 'switch'))
    args = ap.parse_args()

    rows = []
    for suite in args.suites:
        root = RESULTS / suite
        if not root.is_dir():
            continue
        for path in sorted(root.glob('*.json')):
            if path.name in ('summary.json', 'manifest.json'):
                continue
            preds = root / 'preds' / f'{path.stem}.npz'
            if not preds.is_file():
                continue
            try:
                row = evaluate_run(path, preds)
            except Exception as exc:
                print(f'skip {path.name}: {exc}', flush=True)
                continue
            if row:
                row['suite'] = suite
                rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        for snr in r['deployed']:
            flat.append({
                'suite': r['suite'], 'dataset': r['dataset'],
                'model': r['model'], 'cond': r['cond'],
                'split': r['split'], 'seed': r['seed'], 'snr': snr,
                'deployed': r['deployed'][snr], 'oracle': r['oracle'][snr],
                'regret': round(r['oracle'][snr] - r['deployed'][snr], 2),
                'switch_agree': r['switch_agree'][snr],
                'bucket_exact': r['bucket_exact'][snr],
                'bucket_within1': r['bucket_within1'][snr],
            })
    df = pd.DataFrame(flat)
    df.to_csv(out / 'switch_runs.csv', index=False)
    if len(df):
        summary = (df.groupby(['suite', 'model', 'cond', 'snr'])
                   [['deployed', 'oracle', 'regret', 'switch_agree',
                     'bucket_exact', 'bucket_within1']]
                   .agg(['mean', 'std']).round(3))
        summary.to_csv(out / 'switch_summary.csv')
        print(summary.to_string())
    (out / 'switch_summary.json').write_text(json.dumps({
        'runs': len(rows), 'rows': len(flat),
        'by_suite': df.groupby('suite').size().to_dict() if len(df) else {},
    }, indent=1))


if __name__ == '__main__':
    main()
