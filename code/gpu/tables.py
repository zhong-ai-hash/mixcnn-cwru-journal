"""Turn per-run JSON into the tables the manuscript needs.

Every number in the paper comes from one GPU matrix; this script is the only
place that aggregates it, so the tables and the text cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import RESULTS


def load(suite: str) -> list[dict]:
    rows = []
    for path in sorted((RESULTS / suite).glob('*.json')):
        if path.name in ('summary.json', 'manifest.json', 'runtime.json'):
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get('complete'):
            rows.append(payload)
    return rows


def _mean_sd(values) -> tuple[float, float, int]:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return float('nan'), float('nan'), 0
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(a.mean()), sd, len(a)


def by_split(rows: list[dict], metric: str) -> dict[str, list[float]]:
    out = defaultdict(list)
    for r in rows:
        value = r.get('pre', {}).get(metric)
        if value is not None:
            out[r['split']].append(float(value))
    return out


def cell(rows: list[dict], metric: str) -> str:
    per_split = by_split(rows, metric)
    means = [float(np.mean(v)) for v in per_split.values() if v]
    mean, sd, n = _mean_sd(means)
    return f'{mean:.2f} ± {sd:.2f} ({n})'


def table_2x2(rows: list[dict], models: list[str]) -> str:
    lines = ['| model | train clean → clean | train clean → 0 dB | '
             'train noise → clean | train noise → 0 dB |',
             '|---|---:|---:|---:|---:|']
    for model in models:
        cells = []
        for cond, metric in (('clean', 'acc_clean'), ('clean', 'acc_snr_0'),
                             ('noise', 'acc_clean'), ('noise', 'acc_snr_0')):
            sel = [r for r in rows if r['model'] == model
                   and r['cond'] == cond]
            cells.append(cell(sel, metric))
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table_split(rows: list[dict], models: list[str]) -> str:
    lines = ['| model | cond | LOBO | recording-random | window-random | '
             'random − LOBO |',
             '|---|---|---:|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            sel = [r for r in rows if r['model'] == model and r['cond'] == cond]
            vals = {}
            for kind in ('lobo', 'recording-random', 'window-random'):
                sub = [r for r in sel
                       if r.get('split_meta', {}).get('kind') == kind]
                per_split = by_split(sub, 'acc_snr_0')
                means = [float(np.mean(v)) for v in per_split.values() if v]
                vals[kind] = _mean_sd(means)
            delta = vals['recording-random'][0] - vals['lobo'][0]
            lines.append(
                f"| {model} | {cond} | "
                f"{vals['lobo'][0]:.2f} ± {vals['lobo'][1]:.2f} | "
                f"{vals['recording-random'][0]:.2f} ± "
                f"{vals['recording-random'][1]:.2f} | "
                f"{vals['window-random'][0]:.2f} ± "
                f"{vals['window-random'][1]:.2f} | {delta:+.2f} |")
    return '\n'.join(lines)


def table_bn(rows: list[dict], models: list[str]) -> str:
    lines = ['| model | train | pre | BN clean | BN 0 dB | BN 4 dB |',
             '|---|---|---:|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            sel = [r for r in rows if r['model'] == model and r['cond'] == cond]
            cells = [cell(sel, 'acc_snr_0')]
            for bank in ('clean', '0', '4'):
                sub = [dict(r, pre=r.get(f'bn_{bank}', {})) for r in sel]
                cells.append(cell(sub, 'acc_snr_0'))
            lines.append(f'| {model} | {cond} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table_dataset(rows: list[dict], models: list[str]) -> str:
    lines = ['| model | cond | grouped | random | Δ |',
             '|---|---|---:|---:|---:|']
    for model in models:
        for cond in ('clean', 'noise'):
            sel = [r for r in rows if r['model'] == model and r['cond'] == cond]
            vals = {}
            for kind, label in (('bearing-grouped', 'grouped'),
                                ('file-random', 'random')):
                sub = [r for r in sel
                       if r.get('split_meta', {}).get('kind') == kind]
                if sub:
                    vals[label] = _mean_sd(
                        [float(np.mean(v)) for v in
                         by_split(sub, 'acc_clean').values() if v])
            if len(vals) == 2:
                lines.append(
                    f"| {model} | {cond} | {vals['grouped'][0]:.2f} ± "
                    f"{vals['grouped'][1]:.2f} | {vals['random'][0]:.2f} ± "
                    f"{vals['random'][1]:.2f} | "
                    f"{vals['random'][0] - vals['grouped'][0]:+.2f} |")
    return '\n'.join(lines)


def overhead(rows: list[dict]) -> str:
    train = [r.get('train_secs') or 0 for r in rows]
    ev = [r.get('eval_secs') or 0 for r in rows]
    peak = [r['peak_alloc_mb'] for r in rows if r.get('peak_alloc_mb')]
    tput = [r['train_windows_per_s'] for r in rows
            if r.get('train_windows_per_s')]
    lines = ['| metric | value |', '|---|---:|',
             f'| runs | {len(rows)} |',
             f'| train s / run | {np.mean(train):.2f} |',
             f'| eval s / run | {np.mean(ev):.2f} |',
             f'| GPU s total | {sum(train) + sum(ev):.0f} |']
    if peak:
        lines.append(f'| peak VRAM MB | {max(peak):.0f} |')
    if tput:
        lines.append(f'| train windows/s | {np.mean(tput):.0f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else RESULTS / 'tables'
    out.mkdir(parents=True, exist_ok=True)

    main_rows = load('main')
    models = sorted({r['model'] for r in main_rows})
    blocks = {
        'table2_2x2.md': table_2x2(main_rows, models),
        'table3_split.md': table_split(main_rows, models),
        'table4_bn.md': table_bn(main_rows, models),
        'table_overhead.md': overhead(main_rows),
    }
    base_rows = load('baselines')
    if base_rows:
        base_models = sorted({r['model'] for r in base_rows})
        blocks['table5_baselines.md'] = table_2x2(base_rows, base_models)
        blocks['table5_baselines_split.md'] = table_split(base_rows,
                                                          base_models)
    for suite, name in (('paderborn', 'table6_paderborn.md'),
                        ('seu', 'table7_seu.md')):
        rows = load(suite)
        if rows:
            blocks[name] = table_dataset(rows, sorted({r['model'] for r in rows}))

    for name, text in blocks.items():
        (out / name).write_text(text + '\n')
        print(f'--- {name} ---')
        print(text)


if __name__ == '__main__':
    main()
