"""Three-seed repetition of the three variants that carry the argument:
reference, cheapest (no pad), and the proposed (MS). Both training
distributions, with BN recalibration reported for the noise-trained models.
"""
import json
from pathlib import Path
from run_final import train, measure, recalibrate_bn

SEEDS = [0, 1, 2]
VARIANTS = ['base', 'noPad', 'multi']
OUT = Path(__file__).resolve().parent.parent / 'results' / 'results_seed3.json'

if __name__ == '__main__':
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r['seed'], r['variant'], r['cond']) for r in rows}
    for seed in SEEDS:
        for v in VARIANTS:
            for aug in (False, True):
                cond = 'aug' if aug else 'clean'
                if (seed, v, cond) in done:
                    print(f'>>> seed{seed} {v} [{cond}] already done, skipping', flush=True)
                    continue
                print(f'>>> seed{seed} {v} [{cond}]', flush=True)
                model, xtr, xte, yte, secs = train(v, seed, aug)
                row = dict(seed=seed, variant=v, cond=cond, secs=secs,
                           params=sum(p.numel() for p in model.parameters()))
                row.update({f'pre_{k}': val for k, val in measure(model, xtr, xte, yte).items()})
                if aug:
                    recalibrate_bn(model, xtr)
                    row.update({f'post_{k}': val for k, val in measure(model, xtr, xte, yte).items()})
                print(f"    {cond} {v}: pre acc={row['pre_acc']} snr0={row['pre_snr0']}"
                      + (f" | post acc={row['post_acc']} snr0={row['post_snr0']}" if aug else ''), flush=True)
                rows.append(row)
                OUT.write_text(json.dumps(rows, indent=1))
    print('done', len(rows))
