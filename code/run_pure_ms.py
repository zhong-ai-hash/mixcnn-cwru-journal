"""Three-seed control for the isolated multi-scale effect.

The original `multi` variant also removes the stem padding. This variant keeps
the padding, so the only change relative to Ref is the multi-scale branch.
"""
import json
from pathlib import Path

from run_final import train, measure, recalibrate_bn

SEEDS = [0, 1, 2]
OUT = Path(__file__).resolve().parent.parent / 'results' / 'results_multi_pad.json'

if __name__ == '__main__':
    rows = []
    for seed in SEEDS:
        for aug in (False, True):
            cond = 'aug' if aug else 'clean'
            print(f'>>> seed{seed} multi_pad [{cond}]', flush=True)
            model, xtr, xte, yte, secs = train('multi_pad', seed, aug)
            row = dict(seed=seed, variant='multi_pad', cond=cond, secs=secs,
                       params=sum(p.numel() for p in model.parameters()))
            row.update({f'pre_{k}': val for k, val in measure(model, xtr, xte, yte).items()})
            if aug:
                recalibrate_bn(model, xtr)
                row.update({f'post_{k}': val for k, val in measure(model, xtr, xte, yte).items()})
            print(f"    {cond}: pre acc={row['pre_acc']} snr0={row['pre_snr0']}"
                  + (f" | post acc={row['post_acc']} snr0={row['post_snr0']}" if aug else ''),
                  flush=True)
            rows.append(row)
            OUT.write_text(json.dumps(rows, indent=1))
    print('done', len(rows))
