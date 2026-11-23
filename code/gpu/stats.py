"""Paired contrasts, TOST, bootstrap CIs and variance components.

Reads the per-run JSON files written by ``suite.py``; no model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS


def load_runs(suite: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / suite).glob('*.json')):
        if path.name == 'summary.json':
            continue
        payload = json.loads(path.read_text())
        if not payload.get('complete'):
            continue
        row = {
            'model': payload['model'], 'cond': payload['cond'],
            'split': payload['split'],
            'kind': payload.get('split_meta', {}).get('kind', 'unknown'),
            'seed': payload['seed'],
        }
        row.update(payload['pre'])
        rows.append(row)
    if not rows:
        raise SystemExit(f'no completed runs under {RESULTS / suite}')
    return pd.DataFrame(rows)


def paired_diff(df, metric, cond_a, cond_b):
    """Return per-group paired differences a-b on matched (split, seed)."""
    out = []
    for (model, kind), group in df.groupby(['model', 'kind']):
        a = group[group.cond == cond_a].set_index(['split', 'seed'])[metric]
        b = group[group.cond == cond_b].set_index(['split', 'seed'])[metric]
        idx = a.index.intersection(b.index)
        if len(idx) < 2:
            continue
        diffs = (a.loc[idx] - b.loc[idx]).to_numpy(dtype=float)
        test = stats.ttest_rel(a.loc[idx], b.loc[idx])
        half = stats.t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(
            len(diffs))
        out.append({
            'model': model, 'kind': kind, 'n_pairs': len(diffs),
            'mean': round(float(diffs.mean()), 3),
            'sd': round(float(diffs.std(ddof=1)), 3),
            'ci95_low': round(float(diffs.mean() - half), 3),
            'ci95_high': round(float(diffs.mean() + half), 3),
            't': round(float(test.statistic), 3),
            'p': round(float(test.pvalue), 5),
            'dz': round(float(diffs.mean() / diffs.std(ddof=1)), 3)
            if diffs.std(ddof=1) > 0 else 0.0,
        })
    return out


def tost(diffs, margin):
    """Two one-sided tests for equivalence within +/- margin."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {'p': 0.0 if abs(d.mean()) < margin else 1.0,
                'equivalent': bool(abs(d.mean()) < margin)}
    t_low = (d.mean() + margin) / se
    t_high = (d.mean() - margin) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    p = max(p_low, p_high)
    return {'margin': margin, 'p': round(float(p), 5),
            'equivalent': bool(p < 0.05)}


def bootstrap_ci(values, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(draws, 2.5)), 3),
            round(float(np.percentile(draws, 97.5)), 3)]


def variance_components(df, metric, model, cond):
    """Two-way random effects (seed x split), one observation per cell."""
    sub = df[(df.model == model) & (df.cond == cond)]
    pivot = sub.pivot_table(index='seed', columns='split', values=metric)
    pivot = pivot.dropna(axis=0, how='any').dropna(axis=1, how='any')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    x = pivot.to_numpy(dtype=float)
    ns, nsp = x.shape
    grand = x.mean()
    seed_means = x.mean(axis=1, keepdims=True)
    split_means = x.mean(axis=0, keepdims=True)
    ss_seed = nsp * ((seed_means - grand) ** 2).sum()
    ss_split = ns * ((split_means - grand) ** 2).sum()
    resid = x - seed_means - split_means + grand
    ss_resid = (resid ** 2).sum()
    df_seed, df_split = ns - 1, nsp - 1
    df_resid = df_seed * df_split
    ms_seed = ss_seed / df_seed
    ms_split = ss_split / df_split
    ms_resid = ss_resid / max(df_resid, 1)
    var_seed = max(0.0, (ms_seed - ms_resid) / nsp)
    var_split = max(0.0, (ms_split - ms_resid) / ns)
    var_resid = max(0.0, ms_resid)
    total = var_seed + var_split + var_resid or 1.0
    return {
        'n_seeds': ns, 'n_splits': nsp,
        'var_seed': round(float(var_seed), 4),
        'var_split': round(float(var_split), 4),
        'var_resid': round(float(var_resid), 4),
        'share_seed': round(float(var_seed / total), 3),
        'share_split': round(float(var_split / total), 3),
        'share_resid': round(float(var_resid / total), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='main')
    ap.add_argument('--metric', default='acc_snr_0')
    ap.add_argument('--cond-a', default='noise')
    ap.add_argument('--cond-b', default='clean')
    ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = load_runs(args.suite)
    contrasts = paired_diff(df, args.metric, args.cond_a, args.cond_b)
    for c in contrasts:
        group = df[(df.model == c['model']) & (df.kind == c['kind'])]
        a = group[group.cond == args.cond_a].set_index(['split', 'seed'])[
            args.metric]
        b = group[group.cond == args.cond_b].set_index(['split', 'seed'])[
            args.metric]
        idx = a.index.intersection(b.index)
        diffs = (a.loc[idx] - b.loc[idx]).to_numpy(dtype=float)
        c['tost'] = tost(diffs, args.margin)
        c['bootstrap_ci'] = bootstrap_ci(diffs)
        c['variance'] = variance_components(
            df, args.metric, c['model'], args.cond_a)

    payload = {'suite': args.suite, 'metric': args.metric,
               'contrast': f'{args.cond_a} - {args.cond_b}',
               'runs': int(len(df)), 'contrasts': contrasts}
    out = Path(args.out) if args.out else RESULTS / args.suite / 'stats.json'
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == '__main__':
    main()
