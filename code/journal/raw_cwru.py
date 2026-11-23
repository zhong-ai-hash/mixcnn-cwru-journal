"""Rebuild a metadata-rich CWRU window set from the original .mat files.

Why this exists
---------------
``data/cwru_data.npy`` is a class-balanced 1010 x 1024 array with no recording
identifiers.  Any split of it is a window-level split, so windows from the same
recording appear in both training and test.  The journal version needs the
opposite: a split by load / recording.  This script recovers the metadata that
the packaged array dropped.

Scope
-----
CWRU 12k drive-end, 10 classes, 0.007--0.021 inch faults plus normal, four
loads (0--3 hp), one file per (class, load).  File numbers follow the standard
CWRU naming; the mapping below is the one used in this work and is written to
the output metadata so it is never implicit again.

Output
------
``data/journal/cwru_raw_windows.npz``
    X        float32 (n, 1024)
    y        int8    class id 0--9 (order in CLASS_NAMES)
    load     int8    0--3 hp
    size     float32 fault diameter in inches (0.0 for normal)
    file_id  int16   CWRU file number (e.g. 105)
    rec_id   int16   recording id; equals file_id here
    rpm      float32 shaft speed read from the .mat file

``data/journal/cwru_raw_meta.json``
    class names, file table, per-file window counts, per-class/per-load
    counts, window length, stride, and missing-file report.

Usage
-----
    python raw_cwru.py --raw-dir /path/to/CWRU --stride 1024
    python raw_cwru.py --raw-dir /path/to/CWRU --max-per-file 20   # smoke test

Run the smoke test first and check that every class has four files and that
the inferred loads match the RPM values before launching a full run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Class order used everywhere in v7.  Do not reorder: tables depend on it.
CLASS_NAMES = [
    'N', 'IR007', 'IR014', 'IR021',
    'B007', 'B014', 'B021',
    'OR007', 'OR014', 'OR021',
]

# CWRU file number -> (class id, nominal load index 0..3).
# The nominal load is only a fallback; when RPM is present the load is
# inferred from the shaft speed, which is the more reliable signal.
FILE_TABLE = {
    # normal baseline
    97: (0, 0), 98: (0, 1), 99: (0, 2), 100: (0, 3),
    # inner race
    105: (1, 0), 106: (1, 1), 107: (1, 2), 108: (1, 3),
    169: (2, 0), 170: (2, 1), 171: (2, 2), 172: (2, 3),
    209: (3, 0), 210: (3, 1), 211: (3, 2), 212: (3, 3),
    # ball
    118: (4, 0), 119: (4, 1), 120: (4, 2), 121: (4, 3),
    185: (5, 0), 186: (5, 1), 187: (5, 2), 188: (5, 3),
    222: (6, 0), 223: (6, 1), 224: (6, 2), 225: (6, 3),
    # outer race
    130: (7, 0), 131: (7, 1), 132: (7, 2), 133: (7, 3),
    197: (8, 0), 198: (8, 1), 199: (8, 2), 200: (8, 3),
    234: (9, 0), 235: (9, 1), 236: (9, 2), 237: (9, 3),
}

# Standard semantic filenames used by several CWRU mirrors.  The nested
# directory names are not part of this map because they vary across mirrors.
SEMANTIC_CLASS = {
    'normal': 0,
    'IR007': 1, 'IR014': 2, 'IR021': 3,
    'B007': 4, 'B014': 5, 'B021': 6,
    'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9,
}
FILE_BY_CLASS_LOAD = {(cls, load): num for num, (cls, load) in FILE_TABLE.items()}

FAULT_SIZE = {
    0: 0.0,
    1: 0.007, 2: 0.014, 3: 0.021,
    4: 0.007, 5: 0.014, 6: 0.021,
    7: 0.007, 8: 0.014, 9: 0.021,
}

# Shaft speeds at 0/1/2/3 hp for the 12k drive-end runs.
RPM_BY_LOAD = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

EXPECTED_FILES = sorted(FILE_TABLE)


def _loadmat(path: Path):
    try:
        from scipy.io import loadmat
    except ImportError:
        sys.exit('scipy is required: pip install scipy')
    try:
        return loadmat(str(path))
    except NotImplementedError:
        sys.exit(f'{path.name}: MAT v7.3 is not supported by scipy; '
                 're-save as v5 or install h5py and adapt this loader.')
    except Exception as exc:  # noqa: BLE001 - report the file, do not crash the scan
        print(f'  ! cannot read {path.name}: {exc}', file=sys.stderr)
        return None


def _drive_end_key(mat: dict, expected_num: int | None = None) -> str | None:
    keys = [k for k in mat if not k.startswith('__')]
    if expected_num is not None:
        exact = f'X{expected_num:03d}_DE_time'
        if exact in keys:
            return exact
    de = sorted(k for k in keys if k.endswith('_DE_time'))
    if de:
        return de[0]
    loose = sorted(k for k in keys if 'DE_time' in k)
    if loose:
        return loose[0]
    return None


def _rpm(mat: dict) -> float | None:
    rpm_keys = sorted(k for k in mat
                      if not k.startswith('__') and k.upper().endswith('RPM'))
    if not rpm_keys:
        if 'RPM' not in mat:
            return None
        rpm_keys = ['RPM']
    v = np.asarray(mat[rpm_keys[0]]).ravel()
    if v.size == 0:
        return None
    return float(v[0])


def _infer_load(rpm: float | None, fallback: int) -> tuple[int, str]:
    if rpm is None or rpm <= 0:
        return fallback, 'file_table'
    load = min(RPM_BY_LOAD, key=lambda k: abs(RPM_BY_LOAD[k] - rpm))
    return load, 'rpm'


def _windows(sig: np.ndarray, length: int, stride: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size < length:
        return np.empty((0, length), dtype=np.float64)
    n = 1 + (sig.size - length) // stride
    idx = np.arange(length)[None, :] + stride * np.arange(n)[:, None]
    return sig[idx]


def _file_number_from_name(path: Path) -> int | None:
    """Map a numeric or standard semantic filename to a CWRU file number."""
    stem = path.stem
    if stem.isdigit():
        return int(stem) if int(stem) in FILE_TABLE else None
    head = stem.split('_')[0]
    if head.isdigit():
        return int(head) if int(head) in FILE_TABLE else None
    m = re.fullmatch(r'(.+)_([0-3])', stem)
    if not m:
        return None
    prefix, load = m.groups()
    cls = SEMANTIC_CLASS.get(prefix)
    if cls is None:
        return None
    return FILE_BY_CLASS_LOAD.get((cls, int(load)))


def _file_number_from_key(key: str) -> int | None:
    m = re.fullmatch(r'X(\d+)_DE_time', key)
    return int(m.group(1)) if m else None


def scan(raw_dir: Path) -> list[Path]:
    files = []
    for p in sorted(raw_dir.rglob('*.mat')):
        parts = [part.lower() for part in p.parts]
        if any('48k' in part for part in parts):
            continue
        if any('fan end' in part for part in parts):
            continue
        if _file_number_from_name(p) is not None:
            files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--window', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=1024)
    ap.add_argument('--max-per-file', type=int, default=0,
                    help='cap windows per file; 0 means no cap')
    ap.add_argument('--detrend', action='store_true',
                    help='subtract the per-window mean (default: keep raw)')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        sys.exit(f'raw dir not found: {raw_dir}')
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir
               else Path(__file__).resolve().parents[2] / 'data' / 'journal')
    out_dir.mkdir(parents=True, exist_ok=True)

    files = scan(raw_dir)
    if not files:
        sys.exit(f'no .mat files with numeric names under {raw_dir}')

    print(f'scanning {len(files)} recognized 12k drive-end .mat files under {raw_dir}')
    X, y, load, size, file_id, rec_id, rpm_col = [], [], [], [], [], [], []
    per_file = []
    unknown = []
    missing = []

    for path in files:
        expected_num = _file_number_from_name(path)
        mat = _loadmat(path)
        if mat is None:
            continue
        key = _drive_end_key(mat, expected_num)
        if key is None:
            print(f'  ! {path.name}: no *_DE_time variable, skipped')
            continue
        num = _file_number_from_key(key) or expected_num
        if num is None or num not in FILE_TABLE:
            unknown.append(num if num is not None else -1)
            print(f'  ? {path.name}: cannot resolve a CWRU file number, skipped')
            continue
        if expected_num is not None and num != expected_num:
            print(f'  ! {path.name}: filename implies file {expected_num}, '
                  f'but signal key implies {num}; using the signal key')

        cls, nominal_load = FILE_TABLE[num]

        sig = np.asarray(mat[key]).ravel()
        rpm = _rpm(mat)
        load_idx, load_src = _infer_load(rpm, nominal_load)
        w = _windows(sig, args.window, args.stride)
        if args.max_per_file and len(w) > args.max_per_file:
            # evenly spaced subset keeps the whole recording represented
            pick = np.linspace(0, len(w) - 1, args.max_per_file).round().astype(int)
            w = w[pick]
        if args.detrend and len(w):
            w = w - w.mean(axis=1, keepdims=True)
        if len(w) == 0:
            print(f'  ! {path.name}: signal shorter than one window, skipped')
            continue

        n = len(w)
        X.append(w.astype(np.float32))
        y.append(np.full(n, cls, dtype=np.int8))
        load.append(np.full(n, load_idx, dtype=np.int8))
        size.append(np.full(n, FAULT_SIZE[cls], dtype=np.float32))
        file_id.append(np.full(n, num, dtype=np.int16))
        rec_id.append(np.full(n, num, dtype=np.int16))
        rpm_col.append(np.full(n, rpm if rpm is not None else np.nan, dtype=np.float32))
        per_file.append({
            'file': num, 'class': CLASS_NAMES[cls], 'load': int(load_idx),
            'load_source': load_src, 'rpm': rpm, 'n_windows': int(n),
            'signal_key': key, 'signal_len': int(sig.size),
        })
        print(f'  + {path.name}: {CLASS_NAMES[cls]:5s} load={load_idx} '
              f'rpm={rpm} windows={n}')

    found = {r['file'] for r in per_file}
    missing = [f for f in EXPECTED_FILES if f not in found]
    if unknown:
        print(f'unknown file numbers (ignored): {sorted(set(unknown))}')
    if missing:
        print(f'! missing expected files: {missing}')

    if not X:
        sys.exit('no windows were built; check variable names and file numbers')

    X = np.concatenate(X)
    y = np.concatenate(y)
    load = np.concatenate(load)
    size = np.concatenate(size)
    file_id = np.concatenate(file_id)
    rec_id = np.concatenate(rec_id)
    rpm_col = np.concatenate(rpm_col)

    npz_path = out_dir / 'cwru_raw_windows.npz'
    np.savez_compressed(
        npz_path, X=X, y=y, load=load, size=size,
        file_id=file_id, rec_id=rec_id, rpm=rpm_col,
    )

    cls_counts = np.bincount(y, minlength=len(CLASS_NAMES)).tolist()
    load_counts = np.bincount(load, minlength=4).tolist()
    meta = {
        'source': 'CWRU 12k drive-end',
        'class_names': CLASS_NAMES,
        'file_table': {str(k): {'class': CLASS_NAMES[v[0]], 'nominal_load': v[1]}
                       for k, v in FILE_TABLE.items()},
        'fault_size_inch': {CLASS_NAMES[k]: v for k, v in FAULT_SIZE.items()},
        'window': args.window,
        'stride': args.stride,
        'detrend': bool(args.detrend),
        'n_windows': int(len(y)),
        'class_counts': cls_counts,
        'load_counts': load_counts,
        'per_file': per_file,
        'missing_files': missing,
        'unknown_files': sorted(set(unknown)),
        'load_inference': ('RPM nearest to 1797/1772/1750/1730; '
                           'file_table fallback when RPM is absent'),
        'note': 'rec_id equals file_id; each file is one recording.',
    }
    meta_path = out_dir / 'cwru_raw_meta.json'
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    print('\nclass counts:', dict(zip(CLASS_NAMES, cls_counts)))
    print('load counts :', dict(zip([0, 1, 2, 3], load_counts)))
    print(f'wrote {npz_path} ({len(y)} windows)')
    print(f'wrote {meta_path}')
    if missing:
        print('WARNING: the window set is incomplete; fix the missing files '
              'before running the main matrix.')


if __name__ == '__main__':
    main()
