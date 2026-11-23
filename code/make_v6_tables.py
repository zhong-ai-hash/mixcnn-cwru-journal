"""Per-class recall table for the v6 manuscript."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRED = ROOT.parent / 'results_v2' / 'preds'
OUT = ROOT.parent / 'results_v2' / 'table4_perclass.tex'


def per_class(variant, cond, key):
    vals = []
    for seed in range(5):
        z = np.load(PRED / f'{variant}_{cond}_seed{seed}.npz')
        y = z['y_test']
        mask = z[key]
        if mask.ndim == 2:
            mask = mask.mean(axis=0)
        for c in range(10):
            vals.append((c, float(mask[y == c].mean() * 100)))
    out = {c: [] for c in range(10)}
    for c, v in vals:
        out[c].append(v)
    return {c: float(np.mean(v)) for c, v in out.items()}


def main():
    cells = {
        'rc': per_class('base', 'clean', 'pre_clean'),
        'r0': per_class('base', 'noise', 'pre_snr_0'),
        'nc': per_class('noPad', 'clean', 'pre_clean'),
        'n0': per_class('noPad', 'noise', 'pre_snr_0'),
    }
    for k, v in cells.items():
        print(k, [round(v[c], 1) for c in range(10)])

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Per-class recall (\%). Clean and 0\,dB, five seeds and ten noise draws. C0--C9 are indices of the packaged label array; class names are not documented in the artifact.}',
        r'\label{tab:perclass}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}lcccc@{}}',
        r'\toprule',
        r'Class & Ref clean & Ref 0\,dB & NoPad clean & NoPad 0\,dB \\',
        r'\midrule',
    ]
    for c in range(10):
        lines.append(
            f'C{c} & {cells["rc"][c]:.1f} & {cells["r0"][c]:.1f} & '
            f'{cells["nc"][c]:.1f} & {cells["n0"][c]:.1f} \\\\')
    lines += [
        r'\midrule',
        r'Overall & 98.41 & 97.38 & 98.51 & 97.70 \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
