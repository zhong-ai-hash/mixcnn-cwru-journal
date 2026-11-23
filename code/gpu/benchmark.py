"""One full job per model, then extrapolate the whole suite's wall time."""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from common import RESULTS, json_dump, pick_device
from engine import run_job
from models import ALL_MODELS, BASELINES, MIXCNN


def bench_one(model, dataset, split, cond, args, device, **overrides):
    cfg = {
        'epochs': args.epochs, 'batch': args.batch, 'lr': 1e-4,
        'weight_decay': 0.0, 'cond': cond, 'amp': True,
        'snr_train': (-4.0, 10.0),
        'snrs': args.snrs, 'draws': args.draws,
        'eval_seed': 50000,
        'bn_snrs': [] if args.skip_bn else [None, -4.0, 0.0, 4.0, 8.0],
        'device': str(device), 'deterministic': False,
    }
    cfg.update(overrides)
    job = dict(dataset=dataset, split=split, model=model, cond=cond, seed=0)
    t0 = time.time()
    payload, _ = run_job(job, cfg, device)
    total = time.time() - t0
    return {
        'model': model, 'dataset': dataset, 'split': split, 'cond': cond,
        'params': payload['params'], 'train_secs': payload['train_secs'],
        'eval_secs': payload.get('eval_secs'), 'total_secs': round(total, 1),
        'acc_clean': payload['pre']['acc_clean'],
        'acc_snr_0': payload['pre'].get('acc_snr_0'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--snrs', nargs='+', type=float,
                    default=[-8, -4, 0, 4, 8, 12, 16])
    ap.add_argument('--skip-bn', action='store_true')
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--pb-models', nargs='+',
                    default=['base'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.dry_run:
        models = args.models or ALL_MODELS
        plan = [{'model': m, 'dataset': 'cwru', 'split': 'lobo0'} for m in models]
        plan += [{'model': m, 'dataset': 'paderborn', 'split': 'pgroup0'}
                 for m in args.pb_models if m in models]
        plan += [{'model': 'base', 'dataset': 'jnu', 'split': '1000'},
                 {'model': 'base', 'dataset': 'seu', 'split': 'sgroup0'}]
        print(json.dumps({'device': str(device), 'jobs': plan}, indent=1))
        return
    props = (torch.cuda.get_device_properties(0)
             if device.type == 'cuda' else None)
    models = args.models or ALL_MODELS
    rows = []
    for model in models:
        print(f'benchmark cwru {model} ...', flush=True)
        rows.append(bench_one(model, 'cwru', 'lobo0', 'noise', args, device))

    print('benchmark jnu/base ...', flush=True)
    jnu = bench_one('base', 'jnu', '1000', 'noise', args, device,
                    bn_snrs=[])

    pb_rows = []
    for model in [m for m in args.pb_models if m in models]:
        print(f'benchmark paderborn {model} ...', flush=True)
        pb_rows.append(bench_one(model, 'paderborn', 'pgroup0', 'noise',
                                 args, device))

    print('benchmark seu/base ...', flush=True)
    seu = bench_one('base', 'seu', 'sgroup0', 'noise', args, device,
                    bn_snrs=[])

    per = {r['model']: r for r in rows}
    seen_mix = [m for m in MIXCNN if m in per]
    seen_base = [m for m in BASELINES if m in per]
    n_main = 24 * 2 * 10          # splits x conditions x seeds
    n_m1c = 2 * 2 * 20            # arms x conditions x seeds
    n_jnu = 3 * 2 * 10            # speeds x conditions x seeds
    n_pb = 10 * 2 * 5             # splits x conditions x seeds
    n_seu = 5 * 2 * 5

    main_h = sum(per[m]['total_secs'] for m in seen_mix) * n_main / 3600
    base_h = sum(per[m]['total_secs'] for m in seen_base) * n_main / 3600
    m1c_h = sum(per[m]['train_secs'] + 0.15 * (per[m]['eval_secs'] or 0)
                for m in seen_mix) * n_m1c / 3600
    ref = per.get('base', rows[0])
    jnu_factor = jnu['total_secs'] / max(ref['total_secs'], 1e-9)
    jnu_h = (sum(per[m]['total_secs'] for m in seen_mix)
             * jnu_factor * n_jnu / 3600)

    pb_per = {r['model']: r for r in pb_rows}
    if pb_per:
        factors = [pb_per[m]['total_secs'] / per[m]['total_secs']
                   for m in pb_per if m in per]
        pb_factor = sorted(factors)[len(factors) // 2]
    else:
        pb_factor = 4.0
    pb_h = sum(per[m]['total_secs'] for m in models) * pb_factor * n_pb / 3600
    seu_factor = (seu['total_secs'] / per['base']['total_secs']
                  if 'base' in per else 1.0)
    seu_h = (sum(per[m]['total_secs'] for m in seen_mix)
             * seu_factor * n_seu / 3600)
    total_h = main_h + base_h + m1c_h + jnu_h + pb_h + seu_h

    payload = {
        'host': platform.node(), 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': str(device),
        'gpu': props.name if props else None,
        'vram_gb': round(props.total_memory / 1e9, 1) if props else None,
        'epochs': args.epochs, 'draws': args.draws,
        'snrs': args.snrs, 'bn_banks': 0 if args.skip_bn else 5,
        'per_model': rows, 'jnu_base': jnu,
        'paderborn_models': pb_rows, 'seu_base': seu,
        'runs_per_suite': {
            'main': len(seen_mix) * n_main,
            'baselines': len(seen_base) * n_main,
            'm1c': len(seen_mix) * n_m1c,
            'jnu': len(seen_mix) * n_jnu,
            'paderborn': len(models) * n_pb,
            'seu': len(seen_mix) * n_seu,
        },
        'assumptions': {
            'paderborn_scale_factor': round(pb_factor, 2),
            'seu_scale_factor': round(seu_factor, 2),
            'bn_banks': 0 if args.skip_bn else 5,
        },
        'eta_gpu_hours': {
            'main': round(main_h, 2), 'baselines': round(base_h, 2),
            'm1c': round(m1c_h, 2), 'jnu': round(jnu_h, 2),
            'paderborn': round(pb_h, 2), 'seu': round(seu_h, 2),
            'total': round(total_h, 2),
        },
        'eta_wall_days': {
            'workers_4': round(total_h / 4 / 24, 2),
            'workers_8': round(total_h / 8 / 24, 2),
        },
    }
    json_dump(RESULTS / 'benchmark.json', payload)
    try:
        import pandas as pd
        pd.DataFrame(rows + [jnu] + pb_rows + [seu]).to_csv(
            RESULTS / 'benchmark.csv', index=False)
    except Exception as exc:
        print(f'csv skipped: {exc}')
    print(json.dumps(payload, indent=1))


if __name__ == '__main__':
    main()
