"""Simple spectral baseline on the same fixed split as protocol_v2."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
OUT = ROOT.parent / 'results_v2' / 'baseline_fft.json'
SPLIT_SEED = 2026
SNRS = [-4.0, 0.0, 4.0, 8.0, 12.0]


def add_noise(x, snr, rng):
    p = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
    return x + rng.normal(size=x.shape) * (p / (10 ** (snr / 20.0)))


def features(x):
    return np.log1p(np.abs(np.fft.rfft(x, axis=1)))[:, :128]


def main():
    X = np.load(DATA / 'cwru_data.npy')
    y = np.load(DATA / 'cwru_label.npy')
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=.30, random_state=SPLIT_SEED,
                               stratify=y)
    va, te = train_test_split(tmp, test_size=2 / 3, random_state=SPLIT_SEED,
                              stratify=y[tmp])

    sc = StandardScaler().fit(features(X[tr]))
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(
        sc.transform(features(X[tr])), y[tr])
    out = {'clean': {'acc': float(clf.score(sc.transform(features(X[te])), y[te])) * 100},
           'noise_trained': {}}
    for snr in SNRS:
        rng = np.random.default_rng(1000 + int(round((snr + 20) * 100)))
        z = add_noise(X[te], snr, rng)
        out['clean'][f'snr_{snr:g}'] = float(
            clf.score(sc.transform(features(z)), y[te])) * 100

    rng = np.random.default_rng(0)
    xx, yy = [X[tr]], [y[tr]]
    for _ in range(10):
        snr = rng.uniform(-4, 10, size=(len(tr), 1))
        xx.append(add_noise(X[tr], snr, rng))
        yy.append(y[tr])
    xx = np.vstack(xx)
    yy = np.concatenate(yy)
    sc2 = StandardScaler().fit(features(xx))
    clf2 = LogisticRegression(max_iter=3000, C=1.0).fit(
        sc2.transform(features(xx)), yy)
    out['noise_trained']['acc'] = float(
        clf2.score(sc2.transform(features(X[te])), y[te])) * 100
    for snr in SNRS:
        vals = []
        for d in range(10):
            rng = np.random.default_rng(2000 + int(round((snr + 20) * 100)) + d)
            z = add_noise(X[te], snr, rng)
            vals.append(clf2.score(sc2.transform(features(z)), y[te]))
        out['noise_trained'][f'snr_{snr:g}'] = float(np.mean(vals)) * 100
        out['noise_trained'][f'snr_{snr:g}_std'] = float(np.std(vals)) * 100
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
