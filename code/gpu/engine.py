"""Training and evaluation primitives (AMP, multi-draw AWGN, BN banks)."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from common import amp_enabled, config_stamp, seed_all


def auto_batch(base: int, length: int) -> int:
    """Keep activation memory bounded for 4096-sample inputs.

    The 12 GB 3080Ti cannot hold four workers at batch 128 x 4096; the
    effective batch is preserved by gradient accumulation instead.
    """
    if length <= 1024:
        return base
    return max(8, int(base * 1024 // length))


def noise_seed(seed_base: int, snr: float, draw: int) -> int:
    """Seed for one noise draw.

    The draw is generated on CPU so the same seed yields the same noise on any
    device; ``bn_switch.py`` replays these draws exactly.
    """
    return int(seed_base) + int(round((snr + 20) * 100)) + int(draw)


def add_noise_group(x, snr: float, d0: int, g: int, seed_base: int):
    """Return ``g`` independent noisy copies of ``x`` stacked on dim 0."""
    gen = torch.Generator().manual_seed(noise_seed(seed_base, snr, d0))
    xs = x.repeat(g, 1, 1) if g > 1 else x.clone()
    noise = torch.randn(xs.shape, generator=gen, dtype=torch.float32)
    p = xs.pow(2).mean(dim=2, keepdim=True).sqrt()
    sigma = (p / (10 ** (snr / 20.0))).float()
    if xs.device.type != 'cpu':
        noise = noise.to(xs.device)
        sigma = sigma.to(xs.device)
    return (xs + noise * sigma).to(x.dtype)


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
def predict(model, x, batch=256, micro_batch=None):
    model.eval()
    batch = int(micro_batch or auto_batch(batch, x.shape[-1]))
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out)


@torch.no_grad()
def predict_logits(model, x, batch=256, micro_batch=None):
    model.eval()
    batch = int(micro_batch or auto_batch(batch, x.shape[-1]))
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch]))
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
    micro = int(cfg.get('micro_batch') or auto_batch(
        cfg['batch'], xtr.shape[-1]))
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
def recalibrate_bn(model, x, batch=64, seed=50000, passes=2):
    """Re-estimate BN statistics from ``x``.

    Windows are shuffled with a fixed seed before batching.  In file order a
    batch holds only a few recordings, and averaging the per-batch unbiased
    variances then drops the between-recording variance; with 1500 windows
    that cost up to 25 accuracy points on a clean model.  Two shuffled passes
    at batch 64 mix recordings and shrink the sampling error of the variance.
    """
    reset_bn(model)
    was_training = model.training
    # Only BN layers may consume batch statistics here.  Keeping dropout and
    # other stochastic modules in eval mode matters for the TCN/Transformer
    # baselines even though MIXCNN itself has no dropout.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
    gen = torch.Generator().manual_seed(int(seed))
    try:
        for _ in range(int(passes)):
            perm = torch.randperm(len(x), generator=gen).to(x.device)
            for i in range(0, len(x), batch):
                idx = perm[i:i + batch]
                if len(idx) < 2:
                    continue
                model(x[idx])
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(model, x, y, snrs, draws, seed_base=50000, batch=256,
             micro_batch=None, want_probs=False):
    from sklearn.metrics import confusion_matrix

    model.eval()
    y_np = y.detach().cpu().numpy()
    logits = predict_logits(model, x, batch, micro_batch)
    pred = logits.argmax(1).cpu().numpy()
    labels = np.arange(int(y_np.max()) + 1)
    counts = np.maximum(np.bincount(y_np, minlength=len(labels)), 1)
    metrics = {
        'acc_clean': round(float(accuracy_score(y_np, pred)) * 100, 2),
        'f1_clean': round(float(f1_score(
            y_np, pred, average='macro', zero_division=0)) * 100, 2),
        'recall_clean': [round(float(v) * 100, 2) for v in
                         np.diag(confusion_matrix(
                             y_np, pred, labels=labels))
                         / counts],
        'confusion_clean': confusion_matrix(
            y_np, pred, labels=labels).tolist(),
    }
    correctness = {'clean': (pred == y_np).astype(np.int8)}
    probs_out = {}
    if want_probs:
        probs_out['clean'] = torch.softmax(
            logits.float(), 1).half().cpu().numpy()
    # Draws are batched together: one forward over g x N windows is far more
    # GPU-efficient than g small forwards, and the noise is still iid per
    # window because each group starts from its own generator seed.
    group = max(1, min(draws, 8192 // max(len(x), 1)))
    for snr in snrs:
        accs, f1s, recalls, rows = [], [], [], []
        prob_chunks = [] if want_probs else None
        for d0 in range(0, draws, group):
            g = min(group, draws - d0)
            xs = add_noise_group(x, float(snr), d0, g, seed_base)
            draw_logits = predict_logits(model, xs, batch, micro_batch)
            p = draw_logits.argmax(1).view(g, len(x)).cpu().numpy()
            if want_probs:
                prob_chunks.append(torch.softmax(
                    draw_logits.float(), 1).half().view(
                        g, len(x), -1).cpu().numpy())
            for i in range(g):
                pi = p[i]
                accs.append(accuracy_score(y_np, pi))
                f1s.append(f1_score(y_np, pi, average='macro',
                                    zero_division=0))
                recalls.append(np.diag(confusion_matrix(
                    y_np, pi, labels=labels)) / counts)
                rows.append((pi == y_np).astype(np.int8))
        if want_probs:
            probs_out[f'snr_{snr:g}'] = np.concatenate(prob_chunks, axis=0)
        key = f'{snr:g}'
        metrics[f'acc_snr_{key}'] = round(float(np.mean(accs)) * 100, 2)
        metrics[f'std_snr_{key}'] = round(float(np.std(accs)) * 100, 2)
        metrics[f'f1_snr_{key}'] = round(float(np.mean(f1s)) * 100, 2)
        metrics[f'recall_snr_{key}'] = [
            round(float(v) * 100, 2) for v in np.mean(recalls, axis=0)]
        correctness[f'snr_{key}'] = np.stack(rows, axis=0)
    if want_probs:
        return metrics, correctness, probs_out
    return metrics, correctness


def run_job(job: dict, cfg: dict, device) -> dict:
    """Train one model on one split and evaluate clean + AWGN + BN banks."""
    from data import split_for
    from models import build, n_params

    seed_all(job['seed'], cfg.get('deterministic', False))
    split = split_for(job['dataset'], job['split'])
    # Splits are built on CPU; keep every tensor on the compute device so the
    # permutation, noise draw and BN recalibration share one device.
    xtr = split['xtr'].to(device)
    ytr = split['ytr'].to(device)
    xte = split['xte'].to(device)
    yte = split['yte'].to(device)
    n_classes = int(split['meta'].get('n_classes', 10))

    model = build(job['model'], n_classes).to(device)
    _reset_peak_memory(device)
    train_stats = train_model(model, xtr, ytr, cfg, device)
    secs = train_stats['secs']
    if cfg.get('save_state'):
        from pathlib import Path
        stem = (f"{job['dataset']}_{job['split']}_{job['model']}_"
                f"{job['cond']}_seed{job['seed']}")
        state_dir = Path(cfg.get('state_dir') or '.')
        state_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_dir / f'{stem}.pt')

    want_probs = bool(cfg.get('save_probs'))
    t_eval = time.time()
    out = evaluate(model, xte, yte, cfg['snrs'], cfg['draws'],
                   seed_base=cfg['eval_seed'],
                   micro_batch=cfg.get('micro_batch'),
                   want_probs=want_probs)
    metrics, corr = out[0], out[1]
    probs = out[2] if want_probs else {}
    eval_secs = round(time.time() - t_eval, 1)
    eval_windows = (len(xte) * (1 + len(cfg['snrs']) * cfg['draws'])
                    * (1 + len(cfg['bn_snrs'])))
    payload = {
        **job,
        'cfg_stamp': config_stamp(cfg),
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
    for key, value in probs.items():
        arrays[f'prob_{key}'] = value

    cal_seed = int(cfg['eval_seed']) + 100000
    for bank_snr in cfg['bn_snrs']:
        bank = 'clean' if bank_snr is None else f'{bank_snr:g}'
        if bank_snr is None:
            recalibrate_bn(model, xtr, batch=64, seed=cal_seed)
        else:
            noisy = add_noise_group(xtr, float(bank_snr), 0, 1,
                                    cal_seed)
            recalibrate_bn(model, noisy, batch=64, seed=cal_seed)
        bank_metrics, bank_corr = evaluate(
            model, xte, yte, cfg['snrs'], cfg['draws'],
            seed_base=cfg['eval_seed'],
            micro_batch=cfg.get('micro_batch'))
        payload[f'bn_{bank}'] = bank_metrics
        for key, value in bank_corr.items():
            arrays[f'bn_{bank}_{key}'] = value

    # Self-calibration control: only clean-trained weights re-estimated on
    # clean data should stay within +-0.5.  Noise-trained weights are expected
    # to move and are recorded descriptively, not as a pass/fail assertion.
    if 'bn_clean' in payload:
        payload['bn_self_delta'] = round(
            payload['bn_clean']['acc_clean'] - payload['pre']['acc_clean'], 2)
        payload['bn_self_delta_control'] = (job['cond'] == 'clean')
        payload['bn_self_delta_note'] = (
            'valid clean-weights control; expected within +-0.5 points'
            if job['cond'] == 'clean' else
            'descriptive only; noise-trained weights need not match pre')
    else:
        payload['bn_self_delta_control'] = False
        payload['bn_self_delta_note'] = 'not available: BN banks skipped'

    payload['eval_secs'] = round(time.time() - t_eval, 1)
    payload['eval_windows'] = int(eval_windows)
    payload['eval_windows_per_s'] = round(
        eval_windows / max(payload['eval_secs'], 1e-9), 1)
    payload['secs'] = round(payload['train_secs'] + payload['eval_secs'], 1)
    payload.update(_peak_memory(device))
    return payload, arrays
