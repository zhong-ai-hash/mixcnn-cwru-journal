"""Decisive experiment: does training-time noise augmentation rescue the
variants that failed under test-time noise?

For every variant we train twice -- clean and with on-the-fly AWGN -- keep the
weights, and then evaluate noise accuracy averaged over 5 draws (the previous
run used a single draw, which was too noisy to rank variants).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from variants import build

torch.manual_seed(0); np.random.seed(0)
SNR_TRAIN = (-4.0, 10.0)
DRAWS = 5

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def load(seed=0):
    X = np.load(DATA_DIR / 'cwru_data.npy')
    y = np.load(DATA_DIR / 'cwru_label.npy')
    X = StandardScaler().fit_transform(X.T).T
    X = np.expand_dims(X, axis=1).astype(np.float32)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    xtr, xva, ytr, yva = train_test_split(xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    return (torch.from_numpy(xtr), torch.from_numpy(ytr).long(),
            torch.from_numpy(xva), torch.from_numpy(yva).long(),
            torch.from_numpy(xte), torch.from_numpy(yte).long())


def add_noise(x, snr_db):
    p = x.pow(2).mean().sqrt()
    sigma = p / (10 ** (snr_db / 20.0))
    return x + torch.randn_like(x) * sigma


@torch.no_grad()
def evaluate(model, x, y, snr=None, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        preds.append(model(xb).argmax(1))
    preds = torch.cat(preds).numpy()
    return accuracy_score(y.numpy(), preds), f1_score(y.numpy(), preds, average='macro')


@torch.no_grad()
def evaluate_noisy(model, x, y, snr, bs=64):
    """Average over DRAWS independent noise realisations."""
    model.eval()
    accs = []
    for d in range(DRAWS):
        torch.manual_seed(1000 + d)
        preds = []
        for i in range(0, len(x), bs):
            xb = add_noise(x[i:i + bs], snr)
            preds.append(model(xb).argmax(1))
        preds = torch.cat(preds).numpy()
        accs.append(accuracy_score(y.numpy(), preds))
    return float(np.mean(accs)), float(np.std(accs))


def train(variant, aug=False, epochs=20, lr=1e-4, bs=32):
    xtr, ytr, xva, yva, xte, yte = load()
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = len(xtr); t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = xtr[idx]
            if aug:
                # one SNR per batch, drawn uniformly per step
                snr = float(np.random.uniform(*SNR_TRAIN))
                xb = add_noise(xb, snr)
            opt.zero_grad()
            loss = crit(model(xb), ytr[idx])
            loss.backward(); opt.step()
    return model, xtr, ytr, xva, yva, xte, yte, round(time.time() - t0, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['base', 'nores', 'noPad', 'soft', 'multi', 'soft_multi'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', default='results_noise.json')
    a = ap.parse_args()
    rows = []
    for cond in ('clean', 'aug'):
        for v in a.variants:
            print(f'>>> {v} [{cond}]', flush=True)
            model, xtr, ytr, xva, yva, xte, yte, secs = train(v, aug=(cond == 'aug'), epochs=a.epochs)
            acc, f1 = evaluate(model, xte, yte)
            n4, s4 = evaluate_noisy(model, xte, yte, 4.0)
            n0, s0 = evaluate_noisy(model, xte, yte, 0.0)
            row = dict(variant=v, cond=cond, acc=round(acc * 100, 2), f1=round(f1 * 100, 2),
                       snr4=round(n4 * 100, 2), snr4_std=round(s4 * 100, 2),
                       snr0=round(n0 * 100, 2), snr0_std=round(s0 * 100, 2),
                       params=sum(p.numel() for p in model.parameters()), secs=secs)
            print(json.dumps(row), flush=True)
            rows.append(row)
            json.dump(rows, open(a.out, 'w'), indent=1)
    print('done', len(rows))
