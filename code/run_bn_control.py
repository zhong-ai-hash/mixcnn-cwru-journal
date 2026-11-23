"""Target-normalization control for clean-trained models.

For each seed, train on clean windows, then separately re-estimate BN
statistics on clean, 4 dB, and 0 dB training windows.  The weights never
change, so the three rows isolate the normalization statistics.
"""
import json
from pathlib import Path

import torch

from run_final import (add_noise, acc_noisy, acc_of, recalibrate_bn, train)

SEEDS = [0, 1, 2]
VARIANTS = ['noPad']
OUT = Path(__file__).resolve().parent.parent / 'results' / 'results_bn_control.json'


def noisy_calibration_set(xtr, snr, draws=5):
    parts = []
    for d in range(draws):
        torch.manual_seed(5000 + d)
        parts.append(add_noise(xtr, snr))
    return torch.cat(parts)


def measure(model, xte, yte):
    return {
        'acc': round(acc_of(model, xte, yte)[0] * 100, 2),
        'snr4': round(acc_noisy(model, xte, yte, 4.0)[0] * 100, 2),
        'snr0': round(acc_noisy(model, xte, yte, 0.0)[0] * 100, 2),
    }


if __name__ == '__main__':
    rows = []
    for seed in SEEDS:
        for v in VARIANTS:
            print(f'>>> seed{seed} {v} clean-trained', flush=True)
            model, xtr, xte, yte, secs = train(v, seed, False)
            base = measure(model, xte, yte)
            rows.append(dict(seed=seed, variant=v, calib='clean',
                             secs=secs, **base))
            for snr in (4.0, 0.0):
                recalibrate_bn(model, noisy_calibration_set(xtr, snr))
                rows.append(dict(seed=seed, variant=v, calib=f'{snr:g}dB',
                                 secs=secs, **measure(model, xte, yte)))
            print(f'    clean={base}', flush=True)
            OUT.write_text(json.dumps(rows, indent=1))
    print('done', len(rows))
