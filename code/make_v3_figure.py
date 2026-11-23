"""Column-width noise curves for the v3 conference manuscript."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
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
    plt.rcParams.update({
        'font.size': 7.0,
        'axes.titlesize': 7.5,
        'axes.labelsize': 7.0,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 5.6,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05), sharey=True)
    specs = [
        ('clean', 'pre', 'clean-trained', 'o', '-'),
        ('noise', 'pre', 'noise-trained', 's', '-'),
        ('noise', 'bn_clean', 'noise-trained, clean BN', '^', '--'),
    ]
    for ax, variant, title in zip(axes, ['base', 'noPad'], ['(a) Ref', '(b) NoPad']):
        for cond, sec, label, marker, ls in specs:
            m, sd = collect(variant, cond, sec)
            ax.errorbar(SNRS, m, yerr=sd, marker=marker, linestyle=ls,
                        capsize=1.6, label=label, linewidth=0.9,
                        markersize=3.2, elinewidth=0.6)
        ax.axvline(0, color='0.75', lw=0.6, zorder=0)
        ax.set_title(title, pad=2)
        ax.set_xlabel('Test SNR (dB)')
        ax.set_xticks(SNRS)
        ax.set_xticklabels(['-4', '0', '4', '8', '12'])
        ax.set_ylim(20, 103)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.tick_params(pad=1.5)
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='lower right', frameon=True, framealpha=0.9,
                   handlelength=1.8, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    fig.savefig(OUT / 'fig_noise_v3.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig_noise_v3.png', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    main()
