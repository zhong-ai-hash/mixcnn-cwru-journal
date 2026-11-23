"""Training and evaluation primitives (AMP, multi-draw AWGN, BN banks)."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from common import amp_enabled, seed_all


def auto_batch(base: int, length: int) -> int:
    """Keep activation memory bounded for 4096-sample inputs.

    The 12 GB 3080Ti cannot hold four workers at batch 128 x 4096; the
    effective batch is preserved by gradient accumulation instead.
    """
    if length <= 1024:
        return base
    return max(8, int(base * 1024 // length))


def _autocast(device, enabled):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except Exception:  # torch < 2.4
        return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:  # torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device):
    if device.type != 'cuda':
        return {}
    return {
        'peak_alloc_mb': round(
            torch.cuda.max_memory_allocated(device) / 2 ** 20, 1),
        'peak_reserved_mb': round(
            torch.cuda.max_memory_reserved(device) / 2 ** 20, 1),
    }


def add_noise(x: torch.Tensor, snr_db, generator=None) -> torch.Tensor:
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(snr_db, dtype=x.dtype, device=x.device)
    if snr_db.ndim == 0:
        snr_db = snr_db.expand(x.shape[0])
    snr_db = snr_db.reshape(-1, 1, 1)
    p = x.pow(2).mean(dim=2, keepdim=True).sqrt()
    sigma = p / (10 ** (snr_db / 20.0))
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype,
                        device=x.device)
    return x + noise * sigma


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    batch = auto_batch(batch, x.shape[-1])
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out)


def train_model(model, xtr, ytr, cfg, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg.get('weight_decay', 0.0))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.1))
    use_amp = amp_enabled(device, cfg.get('amp', True))
    scaler = _grad_scaler(use_amp)
    autocast = _autocast(device, use_amp)
    n = len(xtr)
    micro = auto_batch(cfg['batch'], xtr.shape[-1])
    accum = max(1, cfg['batch'] // micro)
    t0 = time.time()
    epoch_secs = []
    for _ in range(cfg['epochs']):
        t_epoch = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, micro):
            idx = perm[i:i + micro]
            if len(idx) < 2:
                continue
            if (i // micro) % accum == 0:
                opt.zero_grad(set_to_none=True)
            xb = xtr[idx]
            if cfg['cond'] == 'noise':
                snr = torch.empty(len(idx), dtype=xb.dtype, device=device).uniform_(
                    cfg['snr_train'][0], cfg['snr_train'][1])
                xb = add_noise(xb, snr)
            with autocast:
                loss = crit(model(xb), ytr[idx]) / accum
            scaler.scale(loss).backward()
            if (i // micro + 1) % accum == 0 or i + micro >= n:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        epoch_secs.append(round(time.time() - t_epoch, 2))
    total = round(time.time() - t0, 1)
    return {
        'secs': total,
        'epoch_secs': epoch_secs,
        'epoch_secs_mean': (round(float(np.mean(epoch_secs)), 2)
                            if epoch_secs else 0.0),
        'train_windows_per_s': round(
            n * len(epoch_secs) / max(total, 1e-9), 1),
        'n_train': int(n),
        'micro_batch': int(micro),
        'grad_accum': int(accum),
    }


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None


@torch.no_grad()
def recalibrate_bn(model, x, batch=256):
    batch = auto_batch(batch, x.shape[-1])
    reset_bn(model)
    model.train()
    for i in range(0, len(x), batch):
        model(x[i:i + batch])
    model.eval()


@torch.no_grad()
def evaluate(model, x, y, snrs, draws, seed_base=50000, batch=256):
    from sklearn.metrics import confusion_matrix

    model.eval()
    y_np = y.detach().cpu().numpy()
    pred = predict(model, x, batch).detach().cpu().numpy()
    labels = np.arange(int(y_np.max()) + 1)
    metrics = {
        'acc_clean': round(float(accuracy_score(y_np, pred)) * 100, 2),
        'f1_clean': round(float(f1_score(
            y_np, pred, average='macro', zero_division=0)) * 100, 2),
        'recall_clean': [round(float(v) * 100, 2) for v in
                         np.diag(confusion_matrix(
                             y_np, pred, labels=labels))
                         / np.maximum(np.bincount(
                             y_np, minlength=len(labels)), 1)],
        'confusion_clean': confusion_matrix(
            y_np, pred, labels=labels).tolist(),
    }
    correctness = {'clean': (pred == y_np).astype(np.int8)}
    for snr in snrs:
        accs, f1s, recalls, rows = [], [], [], []
        for d in range(draws):
            gen = torch.Generator(device=x.device).manual_seed(
                seed_base + int(round((snr + 20) * 100)) + d)
            preds = []
            for i in range(0, len(x), batch):
                xb = add_noise(x[i:i + batch], float(snr), generator=gen)
                preds.append(model(xb).argmax(1))
            p = torch.cat(preds).detach().cpu().numpy()
            accs.append(accuracy_score(y_np, p))
            f1s.append(f1_score(y_np, p, average='macro', zero_division=0))
            recalls.append(np.diag(confusion_matrix(
                y_np, p, labels=labels))
                / np.maximum(np.bincount(y_np, minlength=len(labels)), 1))
            rows.append((p == y_np).astype(np.int8))
        key = f'{snr:g}'
        metrics[f'acc_snr_{key}'] = round(float(np.mean(accs)) * 100, 2)
        metrics[f'std_snr_{key}'] = round(float(np.std(accs)) * 100, 2)
        metrics[f'f1_snr_{key}'] = round(float(np.mean(f1s)) * 100, 2)
        metrics[f'recall_snr_{key}'] = [
            round(float(v) * 100, 2) for v in np.mean(recalls, axis=0)]
        correctness[f'snr_{key}'] = np.stack(rows, axis=0)
    return metrics, correctness


def run_job(job: dict, cfg: dict, device) -> dict:
    """Train one model on one split and evaluate clean + AWGN + BN banks."""
    from data import split_for
    from models import build, n_params

    seed_all(job['seed'], cfg.get('deterministic', False))
    split = split_for(job['dataset'], job['split'])
    xtr, ytr = split['xtr'], split['ytr']
    xte, yte = split['xte'], split['yte']
    n_classes = int(split['meta'].get('n_classes', 10))

    model = build(job['model'], n_classes).to(device)
    _reset_peak_memory(device)
    train_stats = train_model(model, xtr, ytr, cfg, device)
    secs = train_stats['secs']

    t_eval = time.time()
    metrics, corr = evaluate(model, xte, yte, cfg['snrs'], cfg['draws'],
                             seed_base=cfg['eval_seed'])
    eval_secs = round(time.time() - t_eval, 1)
    eval_windows = (len(xte) * (1 + len(cfg['snrs']) * cfg['draws'])
                    * (1 + len(cfg['bn_snrs'])))
    payload = {
        **job,
        'n_train': int(len(xtr)), 'n_test': int(len(xte)),
        'n_classes': n_classes,
        'params': n_params(model), 'train_secs': secs,
        'epoch_secs': train_stats['epoch_secs'],
        'epoch_secs_mean': train_stats['epoch_secs_mean'],
        'train_windows_per_s': train_stats['train_windows_per_s'],
        'micro_batch': train_stats['micro_batch'],
        'grad_accum': train_stats['grad_accum'],
        'snrs': list(cfg['snrs']), 'draws': int(cfg['draws']),
        'eval_seed': int(cfg['eval_seed']),
        'bn_banks': ['clean' if v is None else f'{v:g}'
                     for v in cfg['bn_snrs']],
        'pre': metrics,
        'split_meta': split['meta'], 'complete': True,
    }
    arrays = {'y_test': yte.cpu().numpy().astype(np.int8)}
    for key, value in corr.items():
        arrays[f'pre_{key}'] = value

    for bank_snr in cfg['bn_snrs']:
        bank = 'clean' if bank_snr is None else f'{bank_snr:g}'
        if bank_snr is None:
            recalibrate_bn(model, xtr)
        else:
            gen = torch.Generator(device=device).manual_seed(
                cfg['eval_seed'] + int(round((bank_snr + 20) * 100)))
            noise = torch.randn(xtr.shape, generator=gen, dtype=xtr.dtype,
                                device=device)
            p = xtr.pow(2).mean(dim=2, keepdim=True).sqrt()
            sigma = p / (10 ** (bank_snr / 20.0))
            recalibrate_bn(model, xtr + noise * sigma)
        bank_metrics, bank_corr = evaluate(
            model, xte, yte, cfg['snrs'], cfg['draws'],
            seed_base=cfg['eval_seed'])
        payload[f'bn_{bank}'] = bank_metrics
        for key, value in bank_corr.items():
            arrays[f'bn_{bank}_{key}'] = value

    payload['eval_secs'] = round(time.time() - t_eval, 1)
    payload['eval_windows'] = int(eval_windows)
    payload['eval_windows_per_s'] = round(
        eval_windows / max(payload['eval_secs'], 1e-9), 1)
    payload['secs'] = round(payload['train_secs'] + payload['eval_secs'], 1)
    payload.update(_peak_memory(device))
    return payload, arrays
