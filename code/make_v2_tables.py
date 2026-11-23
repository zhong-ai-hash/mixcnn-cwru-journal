"""Build paper tables and paired effects from results_v2/summary.json."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results_v2'
RUNS = OUT / 'runs'

VARIANTS = ['base', 'nores', 'noPad', 'soft', 'multi_pad', 'multi']
LABEL = {
    'base': 'Ref',
    'nores': 'NoRes',
    'noPad': 'NoPad',
    'soft': 'Soft',
    'multi_pad': 'MS (pad)',
    'multi': 'MS (no pad)',
}
SNRS = ['clean', '4', '0', '8']


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob('*.json'))]


def val(run, section, snr):
    key = 'acc_clean' if snr == 'clean' else f'acc_snr_{snr}'
    return run[section][key]


def msd(runs, section, snr):
    a = np.array([val(r, section, snr) for r in runs], dtype=float)
    return a.mean(), a.std(ddof=1), len(a)


def ci95(a):
    a = np.asarray(a, dtype=float)
    h = t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))
    return float(a.mean()), float(h)


def table2(runs):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Protocol v2, five initialization seeds on one fixed split. '
        r'Mean $\pm$ sample std. The test set is never used for fitting; '
        r'noise is injected per window and averaged over ten draws.}',
        r'\label{tab:main_v2}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{2.5pt}',
        r'\begin{tabular}{@{}llcccc@{}}',
        r'\toprule',
        r'Variant & Training & Clean & 4\,dB & 0\,dB & 8\,dB \\',
        r'\midrule',
    ]
    for v in VARIANTS:
        for cond in ('clean', 'noise'):
            q = [r for r in runs if r['variant'] == v and r['cond'] == cond]
            cells = []
            for s in SNRS:
                m, sd, _ = msd(q, 'pre', s)
                cells.append(f'{m:.2f}$\\pm${sd:.2f}')
            lines.append(
                f'{LABEL[v]} & {cond} & ' + ' & '.join(cells) + r' \\')
        if v != VARIANTS[-1]:
            lines.append(r'\midrule')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\par\vspace{2pt}\footnotesize Training labels: clean or AWGN '
        r'with per-window SNR $\sim U[-4,10]$\,dB. '
        r'MS (pad) keeps the reference padding; MS (no pad) removes it.',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def table3(runs):
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{BatchNorm statistic crossing. Weights are trained clean '
        r'or with noise; BN statistics are estimated on clean training '
        r'windows or on noisy training windows at the indicated SNR. '
        r'Five-seed means; the test set is not used.}',
        r'\label{tab:bn_v2}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}lllccc@{}}',
        r'\toprule',
        r'Variant & Weights & BN stats & Clean & 4\,dB & 0\,dB \\',
        r'\midrule',
    ]
    for v in ['base', 'noPad']:
        q = {c: [r for r in runs if r['variant'] == v and r['cond'] == c]
             for c in ('clean', 'noise')}
        rows = [
            ('clean', 'pre', 'clean'),
            ('clean', 'bn_noise_4', r'4\,dB'),
            ('clean', 'bn_noise_0', r'0\,dB'),
            ('noise', 'bn_clean', 'clean'),
            ('noise', 'pre', 'training'),
            ('noise', 'bn_noise_4', r'4\,dB'),
            ('noise', 'bn_noise_0', r'0\,dB'),
        ]
        for i, (cond, section, label) in enumerate(rows):
            if i == 0:
                prefix = f'\\multirow{{7}}{{*}}{{{LABEL[v]}}}'
            else:
                prefix = ''
            cells = []
            for s in ['clean', '4', '0']:
                m, _, _ = msd(q[cond], section, s)
                cells.append(f'{m:.2f}')
            lines.append(f'{prefix} & {cond} & {label} & ' + ' & '.join(cells) + r' \\')
        if v == 'base':
            lines.append(r'\midrule')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\par\vspace{2pt}\footnotesize Training denotes the statistics '
        r'produced during noise augmentation; clean/4/0 means statistics '
        r're-estimated on the corresponding unlabeled training windows.',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def effects(runs):
    out = {}

    def group(v, c):
        return {r['seed']: r for r in runs if r['variant'] == v and r['cond'] == c}

    for c in ('clean', 'noise'):
        for v in VARIANTS[1:]:
            a, b = group(v, c), group('base', c)
            for s in ['clean', '4', '0', '-4', '8', '12']:
                d = [val(a[k], 'pre', s) - val(b[k], 'pre', s) for k in sorted(a)]
                m, h = ci95(d)
                out[f'{v}-base|{c}|{s}'] = {
                    'mean': round(m, 2), 'ci95_low': round(m - h, 2),
                    'ci95_high': round(m + h, 2), 'n': len(d),
                }

    for v in VARIANTS:
        a, b = group(v, 'noise'), group(v, 'clean')
        for s in ['clean', '4', '0', '-4', '8', '12']:
            d = [val(a[k], 'pre', s) - val(b[k], 'pre', s) for k in sorted(a)]
            m, h = ci95(d)
            out[f'{v}|noise-clean|{s}'] = {
                'mean': round(m, 2), 'ci95_low': round(m - h, 2),
                'ci95_high': round(m + h, 2), 'n': len(d),
            }

    for v in ['base', 'noPad']:
        g = group(v, 'clean')
        for s, sec in [('0', 'bn_noise_0'), ('4', 'bn_noise_4')]:
            d = [val(g[k], sec, s) - val(g[k], 'pre', s) for k in sorted(g)]
            m, h = ci95(d)
            out[f'{v}|clean-target-bn|{s}'] = {
                'mean': round(m, 2), 'ci95_low': round(m - h, 2),
                'ci95_high': round(m + h, 2), 'n': len(d),
            }
        g = group(v, 'noise')
        d0 = [val(g[k], 'bn_clean', '0') - val(g[k], 'pre', '0') for k in sorted(g)]
        dc = [val(g[k], 'bn_clean', 'clean') - val(g[k], 'pre', 'clean') for k in sorted(g)]
        for name, d in [('noise-cleanBN|0', d0), ('noise-cleanBN|clean', dc)]:
            m, h = ci95(d)
            out[f'{v}|{name}'] = {
                'mean': round(m, 2), 'ci95_low': round(m - h, 2),
                'ci95_high': round(m + h, 2), 'n': len(d),
            }
    return out


def curves(runs):
    path = OUT / 'noise_curves.csv'
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'training', 'section', 'snr', 'mean', 'sd', 'n'])
        for v in VARIANTS:
            for c in ('clean', 'noise'):
                q = [r for r in runs if r['variant'] == v and r['cond'] == c]
                for sec in ['pre', 'bn_clean', 'bn_noise_0', 'bn_noise_4']:
                    if not q or q[0].get(sec) is None:
                        continue
                    for s in ['-4', '0', '4', '8', '12']:
                        m, sd, n = msd(q, sec, s)
                        w.writerow([v, c, sec, s, f'{m:.2f}', f'{sd:.2f}', n])


if __name__ == '__main__':
    runs = load_runs()
    (OUT / 'table2.tex').write_text(table2(runs))
    (OUT / 'table3.tex').write_text(table3(runs))
    (OUT / 'effects.json').write_text(json.dumps(effects(runs), indent=1))
    curves(runs)
    print(table2(runs))
    print(table3(runs))
    print('wrote', OUT / 'table2.tex', OUT / 'table3.tex',
          OUT / 'effects.json', OUT / 'noise_curves.csv')
