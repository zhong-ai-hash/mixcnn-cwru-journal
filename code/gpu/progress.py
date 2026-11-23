"""Scan completed run JSONs and write a progress table (CSV + JSON).

Safe to run while the matrix is training; it only reads files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import RESULTS

SUITES = ['main', 'baselines', 'm1c', 'jnu', 'paderborn', 'seu']
DEFAULT_EXPECTED = {'main': 2880, 'baselines': 2400, 'm1c': 480,
                    'jnu': 360, 'paderborn': 1100, 'seu': 300}


def scan(suite: str):
    root = RESULTS / suite
    rows, done = [], 0
    if not root.is_dir():
        return rows, done
    for path in root.glob('*.json'):
        if path.name in ('summary.json', 'manifest.json'):
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not payload.get('complete'):
            continue
        done += 1
        rows.append({
            'suite': suite, 'model': payload.get('model'),
            'cond': payload.get('cond'), 'split': payload.get('split'),
            'kind': payload.get('split_meta', {}).get('kind'),
            'seed': payload.get('seed'),
            'train_secs': payload.get('train_secs'),
            'eval_secs': payload.get('eval_secs'),
            'acc_clean': payload.get('pre', {}).get('acc_clean'),
            'acc_snr_0': payload.get('pre', {}).get('acc_snr_0'),
        })
    return rows, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(RESULTS / 'progress.csv'))
    args = ap.parse_args()

    all_rows = []
    summary = []
    for suite in SUITES:
        rows, done = scan(suite)
        all_rows.extend(rows)
        manifest = RESULTS / suite / 'manifest.json'
        expected = DEFAULT_EXPECTED.get(suite)
        if manifest.is_file():
            expected = (json.loads(manifest.read_text()).get('jobs_requested')
                        or expected)
        summary.append({
            'suite': suite, 'done': done, 'expected': expected,
            'remaining': (expected - done) if expected is not None else None,
            'train_hours': round(sum(
                (r['train_secs'] or 0) for r in rows) / 3600, 2),
            'eval_hours': round(sum(
                (r['eval_secs'] or 0) for r in rows) / 3600, 2),
            'fails': sum(1 for _ in (RESULTS / suite / 'failures.jsonl')
                         .open()) if (RESULTS / suite / 'failures.jsonl').is_file()
            else 0,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(out, index=False)
        pd.DataFrame(summary).to_csv(out.with_name('progress_summary.csv'),
                                     index=False)
    except Exception as exc:
        print(f'csv skipped: {exc}')
    (out.with_suffix('.json')).write_text(json.dumps(summary, indent=1))
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
