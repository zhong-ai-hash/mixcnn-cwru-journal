"""Dataset loading and split construction for the GPU experiments.

All splits are built once from the frozen npz files.  Standardisation is fit
on the training windows of each split only; the test recording never informs
the scaler.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from common import DATA

CWRU_NPZ = DATA / 'cwru_raw_windows.npz'
JNU_NPZ = DATA / 'jnu_windows.npz'
PADERBORN_NPZ = DATA / 'paderborn_windows.npz'
SEU_NPZ = DATA / 'seu_gear_windows.npz'
SPLIT_SEED = 2026
HALF = 25


def _stratified(y, idx, frac, seed):
    rng = np.random.default_rng(seed)
    picked = []
    for cls in np.unique(y[idx]):
        cls_idx = idx[y[idx] == cls]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        take = int(round(len(cls_idx) * frac))
        picked.append(cls_idx[:take])
    return np.sort(np.concatenate(picked))


@lru_cache(maxsize=1)
def load_cwru() -> dict:
    d = np.load(CWRU_NPZ)
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=1)
def load_jnu() -> dict:
    d = np.load(JNU_NPZ)
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=1)
def load_paderborn() -> dict:
    d = np.load(PADERBORN_NPZ)
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=1)
def load_seu() -> dict:
    d = np.load(SEU_NPZ)
    return {k: d[k] for k in d.files}


def _to_tensors(X, y, idx_tr, idx_va, idx_te, meta, n_classes=10):
    X = X.astype(np.float32, copy=False)
    mean = X[idx_tr].mean(axis=0, dtype=np.float64)
    std = X[idx_tr].std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    Xs = ((X - mean) / std).astype(np.float32)

    def pack(ii):
        return (torch.from_numpy(Xs[ii]).unsqueeze(1),
                torch.from_numpy(y[ii].astype(np.int64)))

    xtr, ytr = pack(idx_tr)
    xva, yva = pack(idx_va)
    xte, yte = pack(idx_te)
    meta = dict(meta)
    meta.update({
        'n_train': int(len(idx_tr)), 'n_val': int(len(idx_va)),
        'n_test': int(len(idx_te)),
        'class_counts_train': np.bincount(
            y[idx_tr], minlength=n_classes).tolist(),
        'class_counts_test': np.bincount(
            y[idx_te], minlength=n_classes).tolist(),
        'n_classes': int(n_classes),
        'scaler': 'per-position mean/std fit on train split only',
    })
    return {'xtr': xtr, 'ytr': ytr, 'xva': xva, 'yva': yva,
            'xte': xte, 'yte': yte, 'meta': meta}


@lru_cache(maxsize=64)
def cwru_split(name: str) -> dict:
    d = load_cwru()
    X, y = d['X'], d['y'].astype(np.int64)
    load, file_id = d['load'].astype(np.int64), d['file_id'].astype(np.int64)
    idx = np.arange(len(y))

    if name.startswith('lobo'):
        k = int(name[4:])
        idx_te = idx[load == k]
        idx_tr = idx[load != k]
        idx_va = np.array([], dtype=np.int64)
        meta = {'split': name, 'kind': 'lobo', 'test_loads': [k],
                'train_loads': sorted(set(load[idx_tr].tolist()))}

    elif name.startswith('group'):
        s = int(name[5:])
        rng = np.random.default_rng(SPLIT_SEED + s)
        tr, te = [], []
        for cls in np.unique(y):
            recs = np.unique(file_id[(y == cls)])
            recs = recs[rng.permutation(len(recs))]
            tr.append(recs[:3]), te.append(recs[3:4])
        rec_tr, rec_te = map(np.concatenate, (tr, te))
        idx_tr = idx[np.isin(file_id, rec_tr)]
        idx_va = np.array([], dtype=np.int64)
        idx_te = idx[np.isin(file_id, rec_te)]
        meta = {'split': name, 'kind': 'recording-random',
                'train_recordings': int(len(rec_tr)),
                'test_recordings': int(len(rec_te)),
                'overlap': int(len(set(rec_tr) & set(rec_te)))}

    elif name.startswith('window'):
        s = int(name[6:])
        idx_te = _stratified(y, idx, 0.20, SPLIT_SEED + s)
        idx_tr = np.setdiff1d(idx, idx_te)
        idx_va = np.array([], dtype=np.int64)
        meta = {'split': name, 'kind': 'window-random',
                'overlap_recordings': int(
                    len(set(file_id[idx_tr]) & set(file_id[idx_te])))}

    elif name in ('m1c_noov', 'm1c_leak'):
        def first_last(load_value):
            first, last = [], []
            for fid in np.unique(file_id[load == load_value]):
                ii = idx[(load == load_value) & (file_id == fid)]
                first.append(ii[:HALF])
                last.append(ii[-HALF:])
            return np.concatenate(first), np.concatenate(last)

        leak_train, idx_te = first_last(0)
        base = idx[load != 0]
        if name == 'm1c_noov':
            idx_tr = base
        else:
            donors, _ = first_last(3)
            idx_tr = np.concatenate([np.setdiff1d(base, donors), leak_train])
        idx_va = np.array([], dtype=np.int64)
        meta = {'split': name, 'kind': 'leakage-isolation',
                'test_loads': [0],
                'overlap_recordings': int(
                    len(set(file_id[idx_tr]) & set(file_id[idx_te])))}
    else:
        raise ValueError(f'unknown CWRU split: {name}')

    return _to_tensors(X, y, idx_tr, idx_va, idx_te, meta)


@lru_cache(maxsize=8)
def jnu_split(held_out: int) -> dict:
    d = load_jnu()
    X, y = d['X'], d['y'].astype(np.int64)
    speed = d['speed'].astype(np.int64)
    idx = np.arange(len(y))
    idx_te = idx[speed == held_out]
    idx_tr = idx[speed != held_out]
    idx_va = np.array([], dtype=np.int64)
    meta = {'split': f'speed{held_out}', 'kind': 'cross-speed',
            'test_speeds': [held_out],
            'train_speeds': sorted(set(speed[idx_tr].tolist()))}
    return _to_tensors(X, y, idx_tr, idx_va, idx_te, meta, n_classes=4)


def _fold_array(keys, y, n_folds=5, seed=0):
    """Assign whole keys (bearing, file or sample) to folds, stratified by class."""
    mapping = {}
    for ci, cls in enumerate(np.unique(y)):
        ks = np.unique(keys[y == cls])
        rng = np.random.default_rng(SPLIT_SEED + seed + 17 * ci)
        ks = ks[rng.permutation(len(ks))]
        for j, key in enumerate(ks):
            mapping[key] = j % n_folds
    return np.asarray([mapping[k] for k in keys], dtype=np.int64)


@lru_cache(maxsize=2)
def paderborn_split(name: str) -> dict:
    d = load_paderborn()
    X, y = d['X'], d['y'].astype(np.int64)
    bearing, file_id = d['bearing'], d['file_id'].astype(np.int64)
    idx = np.arange(len(y))
    n_classes = int(y.max()) + 1

    if name.startswith('pgroup'):
        k = int(name[6:])
        fold = _fold_array(bearing, y)
        idx_te = idx[fold == k]
        idx_tr = idx[fold != k]
        idx_va = np.array([], dtype=np.int64)
        meta = {
            'split': name, 'kind': 'bearing-grouped',
            'test_bearings': sorted({str(b) for b in bearing[idx_te]}),
            'n_bearings_train': int(len(np.unique(bearing[idx_tr]))),
            'n_bearings_test': int(len(np.unique(bearing[idx_te]))),
            'overlap_bearings': int(len(
                set(bearing[idx_tr].tolist()) & set(bearing[idx_te].tolist()))),
            'overlap_files': int(len(
                set(file_id[idx_tr].tolist()) & set(file_id[idx_te].tolist()))),
        }
    elif name.startswith('pfile'):
        k = int(name[5:])
        fold = _fold_array(file_id, y)
        idx_te = idx[fold == k]
        idx_tr = idx[fold != k]
        idx_va = np.array([], dtype=np.int64)
        meta = {
            'split': name, 'kind': 'file-random',
            'n_files_train': int(len(np.unique(file_id[idx_tr]))),
            'n_files_test': int(len(np.unique(file_id[idx_te]))),
            'overlap_bearings': int(len(
                set(bearing[idx_tr].tolist()) & set(bearing[idx_te].tolist()))),
            'overlap_files': int(len(
                set(file_id[idx_tr].tolist()) & set(file_id[idx_te].tolist()))),
        }
    else:
        raise ValueError(f'unknown Paderborn split: {name}')

    return _to_tensors(X, y, idx_tr, idx_va, idx_te, meta,
                       n_classes=n_classes)


@lru_cache(maxsize=2)
def seu_split(name: str) -> dict:
    d = load_seu()
    X, y = d['X'], d['y'].astype(np.int64)
    group = d['group_id'].astype(np.int64)
    idx = np.arange(len(y))
    n_classes = int(y.max()) + 1

    if name.startswith('sgroup'):
        k = int(name[6:])
        fold = _fold_array(group, y)
        idx_te = idx[fold == k]
        idx_tr = idx[fold != k]
        idx_va = np.array([], dtype=np.int64)
        meta = {
            'split': name, 'kind': 'sample-grouped',
            'n_samples_train': int(len(np.unique(group[idx_tr]))),
            'n_samples_test': int(len(np.unique(group[idx_te]))),
            'overlap_samples': int(len(
                set(group[idx_tr].tolist()) & set(group[idx_te].tolist()))),
        }
    elif name.startswith('swindow'):
        k = int(name[7:])
        idx_te = _stratified(y, idx, 0.20, SPLIT_SEED + k)
        idx_tr = np.setdiff1d(idx, idx_te)
        idx_va = np.array([], dtype=np.int64)
        meta = {
            'split': name, 'kind': 'window-random',
            'overlap_samples': int(len(
                set(group[idx_tr].tolist()) & set(group[idx_te].tolist()))),
        }
    else:
        raise ValueError(f'unknown SEU split: {name}')

    return _to_tensors(X, y, idx_tr, idx_va, idx_te, meta,
                       n_classes=n_classes)


def split_for(dataset: str, name: str) -> dict:
    if dataset == 'cwru':
        return cwru_split(name)
    if dataset == 'jnu':
        return jnu_split(int(name))
    if dataset == 'paderborn':
        return paderborn_split(name)
    if dataset == 'seu':
        return seu_split(name)
    raise ValueError(dataset)
