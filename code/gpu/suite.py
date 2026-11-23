"""Job matrix, resume-safe runner and aggregate summary."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import RESULTS, json_dump
from engine import run_job
from models import BASELINES, MIXCNN

SPLITS = ([f'lobo{k}' for k in range(4)]
          + [f'group{k}' for k in range(10)]
          + [f'window{k}' for k in range(10)])
PADERBORN_SPLITS = ([f'pgroup{k}' for k in range(5)]
                    + [f'pfile{k}' for k in range(5)])
SEU_SPLITS = [f'sgroup{k}' for k in range(5)]
DEFAULT_SEEDS = {'main': 10, 'baselines': 10, 'm1c': 20, 'jnu': 10,
                 'paderborn': 5, 'seu': 5}


def build_jobs(args) -> list[dict]:
    if args.suite == 'main':
        models = args.models or MIXCNN
        dataset, splits = 'cwru', args.splits or SPLITS
        seeds = args.seeds or list(range(10))
        conds = ['clean', 'noise']
        jobs = [dict(dataset=dataset, split=s, model=m, cond=c, seed=sd)
                for m in models for c in conds for s in splits for sd in seeds]
    elif args.suite == 'baselines':
        models = args.models or BASELINES
        dataset, splits = 'cwru', args.splits or SPLITS
        seeds = args.seeds or list(range(10))
        conds = ['clean', 'noise']
        jobs = [dict(dataset=dataset, split=s, model=m, cond=c, seed=sd)
                for m in models for c in conds for s in splits for sd in seeds]
    elif args.suite == 'm1c':
        models = args.models or MIXCNN
        seeds = args.seeds or list(range(20))
        jobs = [dict(dataset='cwru', split=s, model=m, cond=c, seed=sd)
                for m in models for c in ['clean', 'noise']
                for s in ['m1c_noov', 'm1c_leak'] for sd in seeds]
    elif args.suite == 'jnu':
        models = args.models or MIXCNN
        seeds = args.seeds or list(range(10))
        jobs = [dict(dataset='jnu', split=str(sp), model=m, cond=c, seed=sd)
                for m in models for c in ['clean', 'noise']
                for sp in (600, 800, 1000) for sd in seeds]
    elif args.suite == 'paderborn':
        models = args.models or ALL_MODELS
        seeds = args.seeds or list(range(5))
        splits = args.splits or PADERBORN_SPLITS
        jobs = [dict(dataset='paderborn', split=s, model=m, cond=c, seed=sd)
                for m in models for c in ['clean', 'noise']
                for s in splits for sd in seeds]
    elif args.suite == 'seu':
        models = args.models or MIXCNN
        seeds = args.seeds or list(range(5))
        splits = args.splits or SEU_SPLITS
        jobs = [dict(dataset='seu', split=s, model=m, cond=c, seed=sd)
                for m in models for c in ['clean', 'noise']
                for s in splits for sd in seeds]
    else:
        raise ValueError(args.suite)
    if args.limit:
        jobs = jobs[:args.limit]
    return jobs


def job_id(job: dict) -> str:
    return (f"{job['dataset']}_{job['split']}_{job['model']}_"
            f"{job['cond']}_seed{job['seed']}")


def _worker(job: dict, cfg: dict, out_dir: str) -> str:
    import torch
    torch.set_num_threads(cfg.get('cpu_threads', 1))
    from common import pick_device
    device = pick_device(cfg['device'])
    stem = job_id(job)
    out_path = Path(out_dir) / f'{stem}.json'
    pred_path = Path(out_dir) / 'preds' / f'{stem}.npz'
    if out_path.exists() and not cfg.get('force'):
        if json.loads(out_path.read_text()).get('complete'):
            return f'{stem} skip'
    payload, arrays = run_job(job, cfg, device)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **arrays)
    json_dump(out_path, payload)
    return f"{stem} done {payload['pre']['acc_clean']:.2f} {payload['secs']:.0f}s"


def _mean_sd(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return round(float(a.mean()), 2), round(sd, 2)


def aggregate(out_dir: Path):
    rows = []
    for path in sorted(out_dir.glob('*.json')):
        if path.name == 'summary.json':
            continue
        payload = json.loads(path.read_text())
        if payload.get('complete'):
            rows.append(payload)
    groups = {}
    for r in rows:
        kind = r.get('split_meta', {}).get('kind', 'unknown')
        key = f"{r['model']}|{r['cond']}|{kind}"
        groups.setdefault(key, {}).setdefault(r['split'], []).append(r)

    metrics = sorted({k for r in rows for k in r.get('pre', {})
                      if k.startswith('acc_') or k.startswith('f1_')})
    summary = {'runs': len(rows), 'groups': {}, 'contrasts': []}
    train_secs = [r.get('train_secs') or 0.0 for r in rows]
    eval_secs = [r.get('eval_secs') or 0.0 for r in rows]
    peaks = [r.get('peak_alloc_mb') for r in rows
             if r.get('peak_alloc_mb') is not None]
    tput = [r.get('train_windows_per_s') for r in rows
            if r.get('train_windows_per_s') is not None]
    epoch_means = [r['epoch_secs_mean'] for r in rows
                   if r.get('epoch_secs_mean')]
    flat = []
    for key, splits in sorted(groups.items()):
        entry = {'n_splits': len(splits), 'metrics': {}}
        for metric in metrics:
            per_split = {}
            for split, items in splits.items():
                per_split[split] = _mean_sd(
                    [it['pre'][metric] for it in items])[0]
            mean, sd = _mean_sd(list(per_split.values()))
            entry['metrics'][metric] = {
                'mean': mean, 'sd_across_splits': sd,
                'per_split': per_split,
            }
            flat.append({
                'group': key, 'metric': metric, 'mean': mean,
                'sd_across_splits': sd, 'n_splits': len(splits),
            })
        summary['groups'][key] = entry

    # Effect of the split unit, reported per model and condition.  The
    # "grouped" kind is the honest reference; the others quantify how much a
    # weaker split inflates the same metric.
    lookup = {(r['group'], r['metric']): r['mean'] for r in flat}
    pairs = [('lobo', 'recording-random', 'recording_vs_lobo'),
             ('lobo', 'window-random', 'window_vs_lobo'),
             ('recording-random', 'window-random', 'window_vs_recording'),
             ('bearing-grouped', 'file-random', 'file_vs_bearing'),
             ('sample-grouped', 'window-random', 'window_vs_sample')]
    models = sorted({k.split('|')[0] for k in summary['groups']})
    conds = sorted({k.split('|')[1] for k in summary['groups']})
    metrics = ('acc_clean', 'acc_snr_0')
    for model in models:
        for cond in conds:
            for a, b, label in pairs:
                for metric in metrics:
                    va = lookup.get((f'{model}|{cond}|{a}', metric))
                    vb = lookup.get((f'{model}|{cond}|{b}', metric))
                    if va is None or vb is None:
                        continue
                    summary['contrasts'].append({
                        'model': model, 'cond': cond, 'metric': metric,
                        'reference_kind': a, 'comparison_kind': b,
                        'contrast': label,
                        'reference': va, 'comparison': vb,
                        'delta': round(vb - va, 2),
                    })
    try:
        import pandas as pd
        pd.DataFrame(flat).to_csv(out_dir / 'summary.csv', index=False)
        run_rows = [{
            'model': r['model'], 'cond': r['cond'], 'split': r['split'],
            'kind': r.get('split_meta', {}).get('kind', 'unknown'),
            'seed': r['seed'], 'acc_clean': r['pre'].get('acc_clean'),
            'acc_snr_0': r['pre'].get('acc_snr_0'),
            'train_secs': r.get('train_secs'), 'eval_secs': r.get('eval_secs'),
            'secs': r.get('secs'),
            'epoch_secs_mean': r.get('epoch_secs_mean'),
            'train_windows_per_s': r.get('train_windows_per_s'),
            'eval_windows_per_s': r.get('eval_windows_per_s'),
            'peak_alloc_mb': r.get('peak_alloc_mb'),
            'peak_reserved_mb': r.get('peak_reserved_mb'),
            'n_train': r.get('n_train'), 'n_test': r.get('n_test'),
            'micro_batch': r.get('micro_batch'),
            'grad_accum': r.get('grad_accum'),
            'params': r.get('params'),
        } for r in rows]
        pd.DataFrame(run_rows).to_csv(out_dir / 'runs.csv', index=False)
    except Exception as exc:  # pandas is optional during smoke tests
        print(f'csv export skipped: {exc}')
    summary['overhead'] = {
        'n_runs': len(rows),
        'gpu_secs_total': round(sum(train_secs) + sum(eval_secs), 1),
        'train_secs_total': round(sum(train_secs), 1),
        'eval_secs_total': round(sum(eval_secs), 1),
        'train_secs_mean': round(float(np.mean(train_secs)), 2) if rows else 0,
        'eval_secs_mean': round(float(np.mean(eval_secs)), 2) if rows else 0,
        'epoch_secs_mean': (
            round(float(np.mean(epoch_means)), 3) if epoch_means else None),
        'train_windows_per_s_mean': (
            round(float(np.mean(tput)), 1) if tput else None),
        'peak_alloc_mb_max': (round(float(np.max(peaks)), 1)
                              if peaks else None),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def update_runtime(out: Path, record: dict) -> dict:
    """Accumulate wall clock across resumed sessions.

    ``manifest.elapsed_secs`` only covers the current process.  The ledger
    keeps every session so a matrix interrupted and restarted three times
    still reports one honest total for the paper's cost table.
    """
    path = out / 'runtime.json'
    ledger = {'sessions': []}
    if path.exists():
        try:
            ledger = json.loads(path.read_text())
        except Exception:
            ledger = {'sessions': []}
    ledger['sessions'].append(record)
    ledger['total_secs'] = round(
        sum(s.get('elapsed_secs', 0.0) for s in ledger['sessions']), 1)
    ledger['total_jobs_done'] = int(
        sum(s.get('runs_new', 0) for s in ledger['sessions']))
    ledger['sessions_count'] = len(ledger['sessions'])
    ledger['updated'] = datetime.now(timezone.utc).isoformat()
    json_dump(path, ledger)
    return ledger


def _count_complete(out: Path) -> int:
    n = 0
    for path in out.glob('*.json'):
        if path.name in ('summary.json', 'manifest.json', 'runtime.json',
                         'progress.json'):
            continue
        try:
            if json.loads(path.read_text()).get('complete'):
                n += 1
        except Exception:
            continue
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='main',
                    choices=['main', 'baselines', 'm1c', 'jnu',
                             'paderborn', 'seu'])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1,
                    help='torch CPU threads per worker (use 1 on GPU boxes)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--eval-seed', type=int, default=50000)
    ap.add_argument('--snrs', nargs='+', type=float,
                    default=[-8, -4, 0, 4, 8, 12, 16])
    ap.add_argument('--bn-snrs', nargs='+', type=float,
                    default=[-4, 0, 4, 8])
    ap.add_argument('--skip-bn', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--splits', nargs='+')
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps({'jobs': len(jobs), 'first': jobs[:3],
                          'last': jobs[-1]}, indent=1))
        return
    if args.aggregate_only:
        print(json.dumps(aggregate(RESULTS / args.suite)['runs']))
        return

    if args.jobs > 1:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

    cfg = {
        'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'cond': None, 'amp': True,
        'snr_train': (-4.0, 10.0), 'snrs': args.snrs,
        'draws': args.draws, 'eval_seed': args.eval_seed,
        'bn_snrs': [] if args.skip_bn else [None] + args.bn_snrs,
        'device': args.device, 'deterministic': args.deterministic,
        'force': args.force, 'cpu_threads': args.threads,
    }
    # cond is per job; run_job reads cfg['cond'], so wrap it per job.
    out = Path(args.out_dir) if args.out_dir else RESULTS / args.suite
    out.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.time()
    runs_before = _count_complete(out)
    fail_path = out / 'failures.jsonl'

    def record_failure(job, exc):
        with fail_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({
                'job': job, 'error': repr(exc),
                'time': datetime.now(timezone.utc).isoformat(),
            }) + '\n')

    if args.jobs == 1:
        for job in jobs:
            job_cfg = dict(cfg, cond=job['cond'])
            try:
                msg = _worker(job, job_cfg, str(out))
            except Exception as exc:
                record_failure(job, exc)
                msg = f'FAIL {job_id(job)} {exc!r}'
            done += 1
            print(f'[{done}/{len(jobs)}] {msg} ({time.time() - t0:.0f}s)',
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_worker, job, dict(cfg, cond=job['cond']),
                                   str(out)): job for job in jobs}
            for future in as_completed(futures):
                done += 1
                job = futures[future]
                try:
                    msg = future.result()
                except Exception as exc:
                    record_failure(job, exc)
                    msg = f'FAIL {job_id(job)} {exc!r}'
                print(f'[{done}/{len(jobs)}] {msg} '
                      f'({time.time() - t0:.0f}s)', flush=True)
    summary = aggregate(out)
    session = {
        'suite': args.suite,
        'started': datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        'finished': datetime.now(timezone.utc).isoformat(),
        'elapsed_secs': round(time.time() - t0, 1),
        'jobs_requested': len(jobs),
        'runs_new': max(0, summary['runs'] - runs_before),
        'runs_total': summary['runs'],
        'jobs_workers': args.jobs,
        'epochs': args.epochs, 'batch': args.batch, 'draws': args.draws,
        'gpu': (__import__('torch').cuda.get_device_name(0)
                if __import__('torch').cuda.is_available() else None),
    }
    ledger = update_runtime(out, session)
    manifest = {
        'suite': args.suite,
        'started': datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        'finished': datetime.now(timezone.utc).isoformat(),
        'elapsed_secs': round(time.time() - t0, 1),
        'total_secs_all_sessions': ledger['total_secs'],
        'sessions': ledger['sessions_count'],
        'runs_new_this_session': session['runs_new'],
        'jobs_requested': len(jobs), 'runs_completed': summary['runs'],
        'jobs_workers': args.jobs, 'device': args.device,
        'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
        'draws': args.draws, 'snrs': args.snrs,
        'seeds': args.seeds or list(range(DEFAULT_SEEDS[args.suite])),
        'splits': sorted({j['split'] for j in jobs}),
        'bn_banks': len(cfg['bn_snrs']),
        'torch': __import__('torch').__version__,
        'cuda': __import__('torch').version.cuda,
        'cudnn': __import__('torch').backends.cudnn.version(),
        'numpy': np.__version__,
        'host': __import__('socket').gethostname(),
        'gpu': (__import__('torch').cuda.get_device_name(0)
                if __import__('torch').cuda.is_available() else None),
    }
    json_dump(out / 'manifest.json', manifest)
    print(json.dumps({'runs': summary['runs'],
                      'groups': len(summary['groups'])}, indent=1))


if __name__ == '__main__':
    main()
