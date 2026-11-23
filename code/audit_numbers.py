"""Recompute parameter counts, per-seed spread, and TOST for the manuscript."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from variants import Net

RUNS = Path(__file__).resolve().parent.parent / 'results_v2' / 'runs'
VARIANTS = ['base', 'nores', 'noPad', 'soft', 'multi_pad', 'multi']


def load(variant, cond):
    files = sorted(RUNS.glob(f'{variant}_{cond}_seed*.json'))
    rows = [json.loads(p.read_text()) for p in files]
    return sorted(rows, key=lambda r: r['seed'])


def tost(d, delta):
    n = len(d)
    m, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    if se == 0:
        p_low = 0.0 if m > -delta else 1.0
        p_high = 0.0 if m < delta else 1.0
        return max(p_low, p_high)
    p_low = stats.t.sf((m + delta) / se, n - 1)
    p_high = stats.t.sf((delta - m) / se, n - 1)
    return max(p_low, p_high)


def main():
    print('--- parameters ---')
    groups = {'stem': 0, 'bn1': 0, 'blocks': 0, 'bn2': 0, 'fc': 0}
    model = Net(n_cls=10, residual=True, soft=False, multi=False, pad=True)
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith('conv_1'):
            groups['stem'] += n
        elif name.startswith('bn_1'):
            groups['bn1'] += n
        elif name.startswith('mix.'):
            groups['blocks'] += n
        elif name.startswith('bn_2'):
            groups['bn2'] += n
        else:
            groups['fc'] += n
        print(f'{name:24s} {n:6d}')
    print(groups,
          'stem_total', groups['stem'] + groups['bn1'],
          'head_total', groups['bn2'] + groups['fc'],
          'total', sum(groups.values()))
    print('per_block', groups['blocks'] / 3)

    print('--- clean accuracy per seed ---')
    clean = {}
    for v in VARIANTS:
        rows = load(v, 'clean')
        acc = np.array([r['pre']['acc_clean'] for r in rows])
        clean[v] = acc
        print(f'{v:10s}', np.round(acc, 4), 'mean', round(acc.mean(), 3),
              'sd', round(acc.std(ddof=1), 3),
              'range', f'{acc.min():.2f}-{acc.max():.2f}')

    print('--- paired differences vs Ref (clean) ---')
    crit = stats.t.ppf(0.975, 4)
    for v in VARIANTS[1:]:
        d = clean[v] - clean['base']
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - crit * se, d.mean() + crit * se
        print(f'{v:10s} mean {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]'
              f'  TOST p(0.5)={tost(d, 0.5):.3f}'
              f'  TOST p(1.0)={tost(d, 1.0):.3f}')


if __name__ == '__main__':
    main()
