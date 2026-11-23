"""Plot the v2 noise curves for Ref and NoPad."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results_v2'
RUNS = OUT / 'runs'
SNRS = [-4, 0, 4, 8, 12]


def collect(variant, cond, section):
    rows = [json.loads(p.read_text()) for p in RUNS.glob(f'{variant}_{cond}_*.json')]
    means, sds = [], []
    for snr in SNRS:
        vals = [r[section][f'acc_snr_{snr}'] for r in rows]
        means.append(np.mean(vals))
        sds.append(np.std(vals, ddof=1))
    return np.array(means), np.array(sds)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, variant, title in zip(axes, ['base', 'noPad'], ['Ref', 'NoPad']):
        specs = [
            ('clean', 'pre', 'clean-trained, clean BN', 'o', '-'),
            ('noise', 'pre', 'noise-trained, training BN', 's', '-'),
            ('noise', 'bn_clean', 'noise-trained, clean BN', '^', '--'),
        ]
        for cond, sec, label, marker, ls in specs:
            m, sd = collect(variant, cond, sec)
            ax.errorbar(SNRS, m, yerr=sd, marker=marker, linestyle=ls,
                        capsize=2, label=label, linewidth=1.2, markersize=4)
        ax.axvline(0, color='0.7', lw=0.8, zorder=0)
        ax.axvline(4, color='0.7', lw=0.8, zorder=0)
        ax.set_title(title)
        ax.set_xlabel('Test SNR (dB)')
        ax.set_ylim(20, 101)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_noise.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig_noise.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
