#!/usr/bin/env python3
"""Rebuild the v7 tables from the per-run JSON payloads.

Why this exists
---------------
``tables.py`` has three defects that change published numbers:

* ``table_split`` always reads ``acc_snr_0`` while the caption is silent
  about the test condition, and the last column subtracts
  recording-random, not window-random;
* ``table_2x2`` / ``table_bn`` average the 24 splits into one cell, which
  hides the 1.3--4.6 point leakage gap;
* ``table7_seu`` looks for bearing/file kinds that SEU does not have, so it
  writes a header with no rows.

This script reads the run JSONs directly (``runs.csv`` keeps only two
metrics) and rebuilds everything with explicit protocol columns, cluster
bootstrap CIs, a per-kind variance decomposition, the M1c arm contrast,
per-class recall and the JNU per-speed breakdown.  No ``preds/*.npz`` is
required: every metric used here is already in the run JSON.

Usage
-----
    python make_gpu_tables.py --root <repo>/results/gpu
    python make_gpu_tables.py --root <extract>/results/gpu --bootstrap 10000
    python make_gpu_tables.py --root <repo>/results/gpu --suites main m1c

Outputs go to ``<root>/tables_rebuilt``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:  # Windows consoles default to GBK and cannot print ± / − / →.
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CWRU_CLASSES = ['N', 'IR007', 'IR014', 'IR021', 'B007', 'B014', 'B021',
                'OR007', 'OR014', 'OR021']
JNU_CLASSES = ['N', 'IB', 'OB', 'TB']
SKIP = {'summary.json', 'manifest.json', 'runtime.json', 'stats.json',
        'benchmark.json'}
PROTOCOLS = [('lobo', 'LOBO'), ('recording-random', 'recording'),
             ('window-random', 'window')]


# --------------------------------------------------------------------- io
def load_suite(root: Path, suite: str) -> list[dict]:
    rows = []
    folder = root / suite
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob('*.json')):
        if path.name in SKIP:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not payload.get('complete'):
            continue
        payload['_kind'] = payload.get('split_meta', {}).get(
            'kind', 'unknown')
        rows.append(payload)
    return rows


def metric(row: dict, key: str, bank: str | None = None) -> float | None:
    block = row.get('pre', {}) if bank is None else row.get(f'bn_{bank}', {})
    value = block.get(key)
    return None if value is None else float(value)


# ---------------------------------------------------------------- helpers
def split_means(rows, key, bank=None) -> dict[str, float]:
    acc = defaultdict(list)
    for row in rows:
        value = metric(row, key, bank)
        if value is not None:
            acc[row['split']].append(value)
    return {s: float(np.mean(v)) for s, v in acc.items() if v}


def mean_sd(values) -> tuple[float, float]:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return float('nan'), float('nan')
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def cell(means: dict[str, float]) -> str:
    m, s = mean_sd(means.values())
    return f'{m:.2f} ± {s:.2f}'


def cell_seeds(rows, key) -> str:
    """Mean ± SD across seeds, for a single-split arm such as M1c."""
    vals = [metric(r, key) for r in rows]
    m, s = mean_sd([v for v in vals if v is not None])
    return f'{m:.2f} ± {s:.2f}'


def boot_values(values, n_boot=5000, seed=0) -> tuple[float, float, float]:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    draws = a[idx].mean(axis=1)
    return (float(a.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def boot_diff(a_values, b_values, n_boot=5000, seed=0):
    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    if not a.size or not b.size:
        return None
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    draws = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return {'mean': float(a.mean() - b.mean()),
            'lo': float(np.percentile(draws, 2.5)),
            'hi': float(np.percentile(draws, 97.5))}


def paired_by_split(rows_a, rows_b, key, bank_a=None, bank_b=None):
    a = {(r['split'], r['seed']): metric(r, key, bank_a) for r in rows_a}
    b = {(r['split'], r['seed']): metric(r, key, bank_b) for r in rows_b}
    common = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    out = defaultdict(list)
    for k in common:
        out[k[0]].append(a[k] - b[k])
    return out


def paired_by_seed(rows_a, rows_b, key, bank_a=None, bank_b=None):
    """Pair two arms that use different split names but the same seeds."""
    a = {r['seed']: metric(r, key, bank_a) for r in rows_a}
    b = {r['seed']: metric(r, key, bank_b) for r in rows_b}
    common = [s for s in a if s in b and a[s] is not None and b[s] is not None]
    return {'paired': [a[s] - b[s] for s in common]} if common else {}


def boot_paired(by_split, n_boot=5000, seed=0):
    if not by_split:
        return None
    splits = list(by_split)
    rng = np.random.default_rng(seed)
    all_values = np.concatenate([np.asarray(by_split[s], float)
                                 for s in splits])
    if len(splits) > 1:
        draws = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, len(splits), len(splits))
            vals = np.concatenate([np.asarray(by_split[splits[j]], float)
                                   for j in pick])
            draws[i] = vals.mean()
    else:
        v = np.asarray(by_split[splits[0]], float)
        idx = rng.integers(0, len(v), (n_boot, len(v)))
        draws = v[idx].mean(axis=1)
    return {'mean': float(all_values.mean()),
            'lo': float(np.percentile(draws, 2.5)),
            'hi': float(np.percentile(draws, 97.5)),
            'n': int(all_values.size), 'n_clusters': len(splits)}


def anova(rows, key):
    """Two-way random effects (seed x split) for a complete rectangular grid."""
    pivot = {}
    for row in rows:
        value = metric(row, key)
        if value is None:
            continue
        pivot.setdefault(int(row['seed']), {})[row['split']] = value
    splits = sorted({s for d in pivot.values() for s in d})
    seeds = [s for s in sorted(pivot) if all(sp in pivot[s] for sp in splits)]
    if len(seeds) < 2 or len(splits) < 2:
        return None
    x = np.asarray([[pivot[s][sp] for sp in splits] for s in seeds], float)
    ns, nsp = x.shape
    grand = x.mean()
    seed_means = x.mean(axis=1, keepdims=True)
    split_means_ = x.mean(axis=0, keepdims=True)
    ss_seed = nsp * ((seed_means - grand) ** 2).sum()
    ss_split = ns * ((split_means_ - grand) ** 2).sum()
    resid = x - seed_means - split_means_ + grand
    ss_resid = (resid ** 2).sum()
    ms_seed = ss_seed / (ns - 1)
    ms_split = ss_split / (nsp - 1)
    ms_resid = ss_resid / max((ns - 1) * (nsp - 1), 1)
    var_seed = max(0.0, (ms_seed - ms_resid) / nsp)
    var_split = max(0.0, (ms_split - ms_resid) / ns)
    var_resid = max(0.0, ms_resid)
    total = var_seed + var_split + var_resid or 1.0
    return {'n_splits': nsp, 'n_seeds': ns,
            'share_split': var_split / total,
            'share_seed': var_seed / total,
            'share_resid': var_resid / total,
            'var_split': var_split, 'var_seed': var_seed,
            'var_resid': var_resid}


def sel(rows, model=None, cond=None, kind=None, split=None):
    out = rows
    if model is not None:
        out = [r for r in out if r['model'] == model]
    if cond is not None:
        out = [r for r in out if r['cond'] == cond]
    if kind is not None:
        out = [r for r in out if r['_kind'] == kind]
    if split is not None:
        out = [r for r in out if r['split'] == split]
    return out


def fmt_ci(d, digits=2, sign=True):
    if d is None:
        return '—'
    if sign:
        return (f'{d["mean"]:+.{digits}f} '
                f'[{d["lo"]:+.{digits}f}, {d["hi"]:+.{digits}f}]')
    return f'{d["mean"]:.{digits}f} [{d["lo"]:.{digits}f}, {d["hi"]:.{digits}f}]'


# ----------------------------------------------------------------- tables
def table_protocol(rows, models):
    head = ('| model | LOBO clean | LOBO 0 dB | recording clean | '
            'recording 0 dB | window clean | window 0 dB |')
    lines = [head, '|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        cells = []
        for kind, _ in PROTOCOLS:
            for key in ('acc_clean', 'acc_snr_0'):
                cells.append(cell(split_means(
                    sel(rows, model, kind=kind), key)))
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table_leakage(rows, models, n_boot):
    lines = ['| model | train | window − LOBO | recording − LOBO | '
             'window − recording |', '|---|---|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            base = split_means(sel(rows, model, cond, 'lobo'), 'acc_clean')
            rec = split_means(sel(rows, model, cond, 'recording-random'),
                              'acc_clean')
            win = split_means(sel(rows, model, cond, 'window-random'),
                              'acc_clean')
            d1 = boot_diff(win.values(), base.values(), n_boot)
            d2 = boot_diff(rec.values(), base.values(), n_boot)
            d3 = boot_diff(win.values(), rec.values(), n_boot)
            lines.append(f'| {model} | {cond} | {fmt_ci(d1)} | '
                         f'{fmt_ci(d2)} | {fmt_ci(d3)} |')
    lines += ['', 'All contrasts are on clean-test accuracy, in points; '
              'brackets are 95% percentile bootstrap CIs over splits.']
    return '\n'.join(lines)


def table_variance(rows, models):
    lines = ['| model | train | pooling | n_splits | split | seed | resid |',
             '|---|---|---|---:|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            base = sel(rows, model, cond)
            pooled = anova(base, 'acc_clean')
            if pooled:
                lines.append(
                    f'| {model} | {cond} | protocol-pooled | '
                    f'{pooled["n_splits"]} | {pooled["share_split"]:.2f} | '
                    f'{pooled["share_seed"]:.2f} | '
                    f'{pooled["share_resid"]:.2f} |')
            for kind, label in PROTOCOLS:
                a = anova(sel(base, kind=kind), 'acc_clean')
                if a:
                    lines.append(
                        f'| {model} | {cond} | {label} | {a["n_splits"]} | '
                        f'{a["share_split"]:.2f} | {a["share_seed"]:.2f} | '
                        f'{a["share_resid"]:.2f} |')
    return '\n'.join(lines)


def table_m1c(rows, models, n_boot):
    lines = ['| model | train | metric | Arm A no-overlap | '
             'Arm B same-recording | B − A | n |',
             '|---|---|---|---:|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            a_rows = sel(rows, model, cond, split='m1c_noov')
            b_rows = sel(rows, model, cond, split='m1c_leak')
            if not a_rows or not b_rows:
                continue
            for key, label in (('acc_clean', 'clean'),
                               ('acc_snr_0', '0 dB')):
                d = boot_paired(paired_by_seed(b_rows, a_rows, key), n_boot)
                lines.append(
                    f'| {model} | {cond} | {label} | '
                    f'{cell_seeds(a_rows, key)} | '
                    f'{cell_seeds(b_rows, key)} | {fmt_ci(d)} | '
                    f'{d["n"] if d else 0} |')
    return '\n'.join(lines)


def table_perclass(rows, models, key, label):
    head = '| model | ' + ' | '.join(CWRU_CLASSES) + ' | min |'
    lines = [f'### per-class recall, {label} (LOBO, mean over folds)', '',
             head, '|---|' + '---:|' * len(CWRU_CLASSES) + '---:|']
    for model in models:
        per_fold = defaultdict(list)
        for row in sel(rows, model, kind='lobo'):
            vec = row.get('pre', {}).get(key)
            if vec is None:
                continue
            per_fold[row['split']].append(np.asarray(vec, float))
        if not per_fold:
            continue
        fold_means = [np.mean(v, axis=0) for v in per_fold.values()]
        mean = np.mean(fold_means, axis=0)
        worst = int(np.argmin(mean))
        lines.append(f'| {model} | ' + ' | '.join(f'{v:.1f}' for v in mean)
                     + f' | {CWRU_CLASSES[worst]} ({mean[worst]:.1f}) |')
    return '\n'.join(lines)


def table_bn(rows, models, n_boot):
    lines = ['> BN recalibration batches are drawn in file order; see the '
             'audit before quoting these numbers.', '',
             '| model | train | pre | BN clean | BN 0 dB | BN 4 dB | '
             'self-Δ (clean bank − pre) | matched-Δ (0 dB bank − pre) |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            base = sel(rows, model, cond, 'lobo')
            if not base or 'bn_clean' not in base[0]:
                continue
            pre = cell(split_means(base, 'acc_snr_0'))
            banks = []
            for bank in ('clean', '0', '4'):
                means = split_means(base, 'acc_snr_0', bank)
                banks.append(cell(means) if means else '—')
            d_self = boot_paired(
                paired_by_split(base, base, 'acc_snr_0', 'clean', None),
                n_boot)
            d_match = boot_paired(
                paired_by_split(base, base, 'acc_snr_0', '0', None),
                n_boot) if 'bn_0' in base[0] else None
            lines.append(f'| {model} | {cond} | {pre} | ' + ' | '.join(banks)
                         + f' | {fmt_ci(d_self)} | {fmt_ci(d_match)} |')
    return '\n'.join(lines)


def table_baselines(rows, models):
    lines = ['| model | LOBO clean→clean | LOBO clean→0 dB | '
             'LOBO noise→clean | LOBO noise→0 dB |',
             '|---|---:|---:|---:|---:|']
    for model in models:
        cells = []
        for cond, key in (('clean', 'acc_clean'), ('clean', 'acc_snr_0'),
                          ('noise', 'acc_clean'), ('noise', 'acc_snr_0')):
            cells.append(cell(split_means(
                sel(rows, model, cond, 'lobo'), key)))
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table_dataset(rows, models, kinds, labels, key, n_boot, title):
    lines = [f'### {title}, {key}', '',
             f'| model | train | {labels[0]} | {labels[1]} | Δ |',
             '|---|---|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            a = split_means(sel(rows, model, cond, kinds[0]), key)
            b = split_means(sel(rows, model, cond, kinds[1]), key)
            if not a or not b:
                continue
            d = boot_diff(b.values(), a.values(), n_boot)
            lines.append(f'| {model} | {cond} | {cell(a)} | {cell(b)} | '
                         f'{fmt_ci(d)} |')
    return '\n'.join(lines)


def table_jnu(rows, models, n_boot):
    speeds = ['600', '800', '1000']
    lines = ['### JNU cross-speed, clean test (per held-out speed)', '',
             '| model | train | ' + ' | '.join(speeds) + ' | overall |',
             '|---|---|' + '---:|' * len(speeds) + '---:|']
    out = {}
    for model in models:
        for cond in ('clean', 'noise'):
            cells = []
            all_means = []
            for speed in speeds:
                means = split_means(
                    sel(rows, model, cond, split=speed), 'acc_clean')
                cells.append(cell(means))
                all_means.extend(means.values())
            overall = mean_sd(all_means)
            lines.append(f'| {model} | {cond} | ' + ' | '.join(cells)
                         + f' | {overall[0]:.2f} ± {overall[1]:.2f} |')
            out[(model, cond)] = overall
    lines += ['', '### JNU seed SD within each speed (shows whether the '
              'large overall SD is between-speed or between-seed)', '',
              '| model | train | ' + ' | '.join(speeds) + ' |',
              '|---|---|' + '---:|' * len(speeds)]
    for model in models:
        for cond in ('clean', 'noise'):
            cells = []
            for speed in speeds:
                vals = [metric(r, 'acc_clean') for r in
                        sel(rows, model, cond, split=speed)]
                vals = [v for v in vals if v is not None]
                cells.append(f'{mean_sd(vals)[1]:.2f}' if vals else '—')
            lines.append(f'| {model} | {cond} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table_jnu_perclass(rows, models):
    head = '| model | train | ' + ' | '.join(JNU_CLASSES) + ' |'
    lines = ['### JNU per-class recall, clean test (all speeds)', '',
             head, '|---|' + '---:|' * len(JNU_CLASSES)]
    for model in models:
        for cond in ('clean', 'noise'):
            vecs = [np.asarray(r['pre']['recall_clean'], float)
                    for r in sel(rows, model, cond)
                    if r.get('pre', {}).get('recall_clean')]
            if not vecs:
                continue
            mean = np.mean(vecs, axis=0)
            lines.append(f'| {model} | {cond} | '
                         + ' | '.join(f'{v:.1f}' for v in mean) + ' |')
    return '\n'.join(lines)


def jnu_collapse(rows, models):
    lines = ['### JNU dominant-predicted-class share (1.0 = collapsed)', '',
             '| model | train | speed 600 | speed 800 | speed 1000 |',
             '|---|---|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            cells = []
            for speed in ('600', '800', '1000'):
                shares = []
                for row in sel(rows, model, cond, split=speed):
                    conf = row.get('pre', {}).get('confusion_clean')
                    if not conf:
                        continue
                    c = np.asarray(conf, float)
                    shares.append(float(c.sum(axis=0).max() / c.sum()))
                cells.append(f'{np.mean(shares):.2f}' if shares else '—')
            lines.append(f'| {model} | {cond} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='results/gpu (repo) or .../extract/results/gpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--bootstrap', type=int, default=5000)
    ap.add_argument('--suites', nargs='+',
                    default=['main', 'baselines', 'm1c', 'jnu',
                             'paderborn', 'seu'])
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'tables_rebuilt'
    out.mkdir(parents=True, exist_ok=True)
    data = {suite: load_suite(root, suite) for suite in args.suites}
    for suite, rows in data.items():
        print(f'{suite:10s} {len(rows):5d} complete runs')

    written = []

    def emit(name, text):
        path = out / name
        path.write_text(text + '\n', encoding='utf-8')
        written.append(name)
        print(f'--- {name} ---')
        print(text)

    summary = {}

    main_rows = data.get('main', [])
    models = sorted({r['model'] for r in main_rows})
    if main_rows:
        emit('t1_protocol_2x2.md', table_protocol(main_rows, models))
        emit('t2_leakage.md', table_leakage(main_rows, models,
                                            args.bootstrap))
        emit('t3_variance.md', table_variance(main_rows, models))
        blocks = [table_perclass(main_rows, models, 'recall_clean',
                                 'clean test'),
                  table_perclass(main_rows, models, 'recall_snr_0', '0 dB')]
        emit('t4_perclass.md', '\n\n'.join(blocks))
        emit('t5_bn.md', table_bn(main_rows, models, args.bootstrap))
        # compact headline numbers
        for model in models:
            for cond in ('clean', 'noise'):
                base = split_means(sel(main_rows, model, cond, 'lobo'),
                                   'acc_clean')
                win = split_means(
                    sel(main_rows, model, cond, 'window-random'),
                    'acc_clean')
                rec = split_means(
                    sel(main_rows, model, cond, 'recording-random'),
                    'acc_clean')
                summary[f'{model}|{cond}|window_minus_lobo'] = boot_diff(
                    win.values(), base.values(), args.bootstrap)
                summary[f'{model}|{cond}|recording_minus_lobo'] = boot_diff(
                    rec.values(), base.values(), args.bootstrap)

    base_rows = data.get('baselines', [])
    if base_rows:
        emit('t6_baselines.md',
             table_baselines(base_rows, sorted({r['model'] for r in base_rows})))

    m1c_rows = data.get('m1c', [])
    if m1c_rows:
        emit('t7_m1c.md', table_m1c(
            m1c_rows, sorted({r['model'] for r in m1c_rows}), args.bootstrap))

    pader = data.get('paderborn', [])
    if pader:
        emit('t8_paderborn.md', table_dataset(
            pader, sorted({r['model'] for r in pader}),
            ('bearing-grouped', 'file-random'), ('bearing-grouped',
                                                 'file-random'),
            'acc_clean', args.bootstrap, 'Paderborn bearing shift'))

    seu = data.get('seu', [])
    if seu:
        lines = ['### SEU, sample-grouped (sanity check)', '',
                 '| model | train | acc clean | acc 0 dB |',
                 '|---|---|---:|---:|']
        for model in sorted({r['model'] for r in seu}):
            for cond in ('clean', 'noise'):
                lines.append(
                    f'| {model} | {cond} | '
                    f'{cell(split_means(sel(seu, model, cond), "acc_clean"))} | '
                    f'{cell(split_means(sel(seu, model, cond), "acc_snr_0"))} |')
        emit('t9_seu.md', '\n'.join(lines))

    jnu = data.get('jnu', [])
    if jnu:
        jmodels = sorted({r['model'] for r in jnu})
        emit('t10_jnu.md', '\n\n'.join([
            table_jnu(jnu, jmodels, args.bootstrap),
            table_jnu_perclass(jnu, jmodels),
            jnu_collapse(jnu, jmodels)]))

    (out / 'headline.json').write_text(json.dumps(summary, indent=1))
    (out / 'README.md').write_text(
        '# Rebuilt tables\n\n'
        f'Source: `{root}`\n\n'
        'Every cell is the mean over splits of the per-split mean over seeds. '
        'The ± is the SD across splits, not across seeds. CIs are percentile '
        'bootstrap over splits (unpaired contrasts) or over splits then seeds '
        '(paired contrasts).\n\n'
        'Generated by `code/gpu/make_gpu_tables.py`.\n', encoding='utf-8')
    print(f'\nwrote {len(written)} tables to {out}')


if __name__ == '__main__':
    main()
