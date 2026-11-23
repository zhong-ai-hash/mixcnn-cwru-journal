"""Build the JNU external-validation window set.

Dataset
-------
JNU bearing dataset, vertical accelerometer, 50 kHz, 600/800/1000 rpm,
four classes: normal (N), inner race (IB), outer race (OB), rolling element
(TB).  One CSV per (class, speed); each file is a single signal column.

Window choice
-------------
The CWRU protocol uses 1024 samples at 12 kHz, i.e. 85 ms per window.  At
50 kHz the same physical duration is 4096 samples, so the JNU windows use
4096 points.  This keeps the observation time comparable and preserves the
native 0--25 kHz band; the trade-off is that the fixed kernels span four
times less physical time than on CWRU, which is stated as a limitation in
the paper.  Resampling JNU to 12 kHz is the alternative and would change
the diagnostic band, so it is not the default.

Output
------
``data/journal/jnu_windows.npz``
    X        float32 (n, 4096)
    y        int8    class id 0--3 (N, IB, OB, TB)
    speed    int16   600/800/1000 rpm
    file_id  int16   index into the file table
    rec_id   int16   equals file_id (one file is one recording)

``data/journal/jnu_meta.json``
    file table, per-class/per-speed counts, signal lengths, sampling rate.

Usage
-----
    python raw_jnu.py --max-per-file 50
    python raw_jnu.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
JNU_DIR = BASE / 'data' / 'journal' / 'jnu'
OUT_DIR = BASE / 'data' / 'journal'

CLASS_MAP = {'n': 0, 'ib': 1, 'ob': 2, 'tb': 3}
CLASS_NAMES = ['N', 'IB', 'OB', 'TB']
SPEEDS = [600, 800, 1000]
FS = 50_000


def parse_name(path: Path):
    m = re.match(r'^(n|ib|ob|tb)(\d{3,4})_', path.stem, flags=re.IGNORECASE)
    if not m:
        return None
    cls_s, speed_s = m.group(1).lower(), m.group(2)
    if cls_s not in CLASS_MAP or int(speed_s) not in SPEEDS:
        return None
    return CLASS_MAP[cls_s], int(speed_s)


def read_signal(path: Path):
    try:
        a = np.loadtxt(path, delimiter=',')
    except Exception as exc:  # noqa: BLE001
        print(f'  ! cannot read {path.name}: {exc}', file=sys.stderr)
        return None, None
    a = np.asarray(a)
    if a.ndim == 1:
        return a.astype(np.float64), 1
    # multi-column files: use the first column and report the width
    return a[:, 0].astype(np.float64), int(a.shape[1])


def windows(sig: np.ndarray, length: int, stride: int) -> np.ndarray:
    if sig.size < length:
        return np.empty((0, length), dtype=np.float64)
    n = 1 + (sig.size - length) // stride
    idx = np.arange(length)[None, :] + stride * np.arange(n)[:, None]
    return sig[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=None,
                    help='default: non-overlapping, equals --window')
    ap.add_argument('--max-per-file', type=int, default=50)
    ap.add_argument('--detrend', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    stride = args.stride or args.window

    files = []
    for p in sorted(JNU_DIR.glob('*.csv')):
        parsed = parse_name(p)
        if parsed is None:
            continue
        files.append((p, *parsed))
    if not files:
        sys.exit(f'no JNU csv files found under {JNU_DIR}')

    print(f'scanning {len(files)} csv files under {JNU_DIR}')
    rows = []
    X, y, speed, file_id = [], [], [], []
    for fid, (path, cls, rpm) in enumerate(files):
        sig, n_cols = read_signal(path)
        if sig is None:
            continue
        w = windows(sig, args.window, stride)
        if args.max_per_file and len(w) > args.max_per_file:
            pick = np.linspace(0, len(w) - 1, args.max_per_file).round().astype(int)
            w = w[pick]
        if args.detrend and len(w):
            w = w - w.mean(axis=1, keepdims=True)
        n = len(w)
        if n == 0:
            print(f'  ! {path.name}: shorter than one window, skipped')
            continue
        X.append(w.astype(np.float32))
        y.append(np.full(n, cls, dtype=np.int8))
        speed.append(np.full(n, rpm, dtype=np.int16))
        file_id.append(np.full(n, fid, dtype=np.int16))
        rows.append({
            'file': path.name, 'class': CLASS_NAMES[cls], 'speed': rpm,
            'samples': int(sig.size), 'seconds': round(sig.size / FS, 2),
            'columns': n_cols, 'windows_available': int(len(windows(sig, args.window, stride))),
            'windows_used': int(n),
        })
        print(f'  + {path.name:16s} {CLASS_NAMES[cls]:2s} {rpm:4d} rpm '
              f'{sig.size/FS:5.1f}s windows={n}')

    if not X:
        sys.exit('no windows built')

    X = np.concatenate(X)
    y = np.concatenate(y)
    speed = np.concatenate(speed)
    file_id = np.concatenate(file_id)

    if args.dry_run:
        print(f'dry run: {len(y)} windows, shape {X.shape}, no files written')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / 'jnu_windows.npz'
    np.savez_compressed(npz_path, X=X, y=y, speed=speed,
                        file_id=file_id, rec_id=file_id)
    meta = {
        'source': 'JNU bearing dataset',
        'sampling_rate_hz': FS,
        'class_names': CLASS_NAMES,
        'speeds_rpm': SPEEDS,
        'window': args.window,
        'stride': stride,
        'duration_ms': round(args.window / FS * 1000, 1),
        'max_per_file': args.max_per_file,
        'detrend': bool(args.detrend),
        'n_windows': int(len(y)),
        'class_counts': np.bincount(y, minlength=4).tolist(),
        'speed_counts': {str(s): int((speed == s).sum()) for s in SPEEDS},
        'files': rows,
        'note': ('4096 samples at 50 kHz gives 82 ms, matching the 85 ms '
                 'CWRU window; native bandwidth is preserved.'),
    }
    meta_path = OUT_DIR / 'jnu_meta.json'
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f'\nclass counts: {dict(zip(CLASS_NAMES, meta["class_counts"]))}')
    print(f'speed counts: {meta["speed_counts"]}')
    print(f'wrote {npz_path} ({len(y)} windows)')
    print(f'wrote {meta_path}')


if __name__ == '__main__':
    main()
