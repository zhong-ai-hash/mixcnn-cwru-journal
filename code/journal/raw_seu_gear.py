"""Build a windowed SEU gearbox dataset from the three labelled .npy files.

The local copy provides ``normal_samples.npy``, ``light_pit_samples.npy`` and
``heavy_pit_samples.npy``, each shaped (312, 3600).  The original split script
derived the class order from the signal standard deviation, which is fragile;
these three files already carry the class assignment, so they are used
directly.

There is no recording identifier in this dataset.  Windows cut from the same
original sample are therefore kept in the same group, and all reported splits
are grouped by that sample id.  This dataset is a cross-component sanity check,
not evidence about recording-level leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
DEFAULT_OUT = BASE / 'data' / 'journal' / 'seu_gear_windows.npz'
DEFAULT_META = BASE / 'data' / 'journal' / 'seu_gear_meta.json'
FILES = [('normal_samples.npy', 0, 'normal'),
         ('light_pit_samples.npy', 1, 'light_pitting'),
         ('heavy_pit_samples.npy', 2, 'heavy_pitting')]


def default_src() -> Path:
    for cand in (Path(r'D:\SEU_gear_db'), BASE.parent / 'SEU_gear_db',
                 BASE / 'data' / 'seu_gear'):
        if (cand / FILES[0][0]).is_file():
            return cand
    return Path(r'D:\SEU_gear_db')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', default=str(default_src()))
    ap.add_argument('--window', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=512)
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--meta', default=str(DEFAULT_META))
    args = ap.parse_args()

    src = Path(args.src_dir)
    X, y, group, sample = [], [], [], []
    counts, shapes = {}, {}
    for fname, label, name in FILES:
        path = src / fname
        if not path.is_file():
            raise SystemExit(f'missing {path}')
        arr = np.load(path)
        shapes[name] = list(arr.shape)
        if arr.ndim != 2:
            raise SystemExit(f'{path} must be 2-D, got {arr.shape}')
        for i in range(arr.shape[0]):
            sig = np.asarray(arr[i], dtype=np.float32).ravel()
            if len(sig) < args.window:
                continue
            sid = label * 100000 + i
            starts = range(0, len(sig) - args.window + 1, args.stride)
            for start in starts:
                X.append(sig[start:start + args.window])
                y.append(label)
                group.append(sid)
                sample.append(i)
        counts[name] = int(sum(1 for v in y if v == label))
        print(f'{fname}: {arr.shape} -> {counts[name]} windows', flush=True)

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    group = np.asarray(group, dtype=np.int64)
    sample = np.asarray(sample, dtype=np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, group_id=group, sample_id=sample)
    meta = {
        'source': str(src), 'files': [f for f, _, _ in FILES],
        'source_shapes': shapes, 'window': args.window,
        'stride': args.stride,
        'sampling_rate_hz': None,
        'sampling_rate_note': ('not documented in the local copy; the '
                               'dataset is used only as a cross-component '
                               'sanity check'),
        'classes': [name for _, _, name in FILES],
        'n_windows': int(len(X)), 'n_samples': int(len(np.unique(group))),
        'counts': counts,
        'windows_per_sample': int(round(len(X) / max(len(np.unique(group)), 1))),
        'split_unit': 'original sample (group_id); windows of one sample never split',
    }
    Path(args.meta).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)[:1200])


if __name__ == '__main__':
    main()
