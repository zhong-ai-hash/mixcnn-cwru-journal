"""Build a windowed Paderborn dataset from the extracted .mat files.

Vibration is stored in ``Y`` as ``vibration_1`` (64 kHz, ~4 s).  Bearing ID
carries the label: K=healthy, KA=outer ring, KI=inner ring, KB=ball.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

BASE = Path(__file__).resolve().parents[2]
DEFAULT_OUT = BASE / 'data' / 'journal' / 'paderborn_windows.npz'
DEFAULT_META = BASE / 'data' / 'journal' / 'paderborn_meta.json'


def default_raw() -> Path:
    """Locate the extracted Paderborn tree without hard-coding one layout."""
    for cand in (BASE / 'data' / 'paderborn_raw',
                 BASE.parent / 'data' / 'paderborn_raw'):
        if cand.is_dir():
            return cand
    return BASE / 'data' / 'paderborn_raw'

LABELS = {'K': 0, 'KA': 1, 'KI': 2, 'KB': 3}
NAMES = ['healthy', 'outer', 'inner', 'ball']


def bearing_label(bearing: str) -> int:
    if bearing.startswith('KA'):
        return LABELS['KA']
    if bearing.startswith('KI'):
        return LABELS['KI']
    if bearing.startswith('KB'):
        return LABELS['KB']
    return LABELS['K']


def vibration(path: Path) -> np.ndarray:
    key = path.stem
    record = loadmat(path, squeeze_me=True, struct_as_record=False)[key]
    entries = record.Y
    if not isinstance(entries, (list, tuple, np.ndarray)):
        entries = [entries]
    for entry in entries:
        if entry.Name == 'vibration_1':
            return np.asarray(entry.Data, dtype=np.float64).ravel()
    raise KeyError(f'vibration_1 not found in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default=str(default_raw()))
    ap.add_argument('--setting', default='N09_M07_F10')
    ap.add_argument('--window', type=int, default=4096)
    ap.add_argument('--per-file', type=int, default=24)
    ap.add_argument('--mode', choices=['3class', '4class'], default='3class')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--meta', default=str(DEFAULT_META))
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    files = sorted(raw.glob(f'*/*{args.setting}*.mat'))
    files = [p for p in files if p.stem.startswith(args.setting)]
    if not files:
        raise SystemExit(
            f'no files under {raw} for {args.setting}; expected '
            f'<bearing>/{args.setting}_<bearing>_<n>.mat')

    X, y, bearing_ids, file_ids = [], [], [], []
    skipped = []
    counts = {}
    for fid, path in enumerate(files):
        bearing = path.stem.split('_')[3]
        label = bearing_label(bearing)
        if args.mode == '3class' and label == LABELS['KB']:
            continue
        try:
            signal = vibration(path)
        except Exception as exc:
            skipped.append({'file': str(path), 'reason': str(exc)})
            print(f'skip {path}: {exc}', flush=True)
            continue
        if len(signal) < args.window:
            continue
        starts = np.unique(np.linspace(
            0, len(signal) - args.window, args.per_file).astype(int))
        for start in starts:
            X.append(signal[start:start + args.window].astype(np.float32))
            y.append(label)
            bearing_ids.append(bearing)
            file_ids.append(fid)
        counts[NAMES[label]] = counts.get(NAMES[label], 0) + len(starts)
        if (fid + 1) % 20 == 0:
            print(f'{fid + 1}/{len(files)} files, {len(X)} windows',
                  flush=True)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int8)
    bearing_ids = np.asarray(bearing_ids)
    file_ids = np.asarray(file_ids, dtype=np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, bearing=bearing_ids,
                        file_id=file_ids)
    per_bearing = {str(b): int((bearing_ids == b).sum())
                   for b in np.unique(bearing_ids)}
    meta = {
        'source': str(raw), 'setting': args.setting,
        'sampling_rate_hz': 64000, 'window': args.window,
        'windows_per_file': args.per_file, 'mode': args.mode,
        'classes': NAMES if args.mode == '4class' else NAMES[:3],
        'n_windows': int(len(X)), 'n_files': int(len(np.unique(file_ids))),
        'n_bearings': int(len(np.unique(bearing_ids))),
        'counts': counts, 'windows_per_bearing': per_bearing,
        'file_names': [p.name for p in files],
        'skipped': skipped,
        'split_unit': 'bearing (leave-one-bearing-out) and file_id (recording)',
    }
    Path(args.meta).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)[:1200])


if __name__ == '__main__':
    main()
