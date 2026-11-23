"""Clean trainer: fixes the three defects in the released model_2_view.py.

Defect 1: hard-coded Windows path H:\\MIXCNN\\cwru_data.npy
Defect 2: test() called once per mini-batch inside the training loop, and never
          switching back to train(), so BatchNorm statistics freeze after step 1
Defect 3: torch.squeeze(out) without a dim collapses a batch of size 1
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from variants import build, Net  # noqa

torch.manual_seed(0); np.random.seed(0)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def load(seed=0):
    X = np.load(DATA_DIR / 'cwru_data.npy')
    y = np.load(DATA_DIR / 'cwru_label.npy')
    X = StandardScaler().fit_transform(X.T).T
    X = np.expand_dims(X, axis=1).astype(np.float32)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    xtr, xva, ytr, yva = train_test_split(xtr, ytr, test_size=0.2,
                                          random_state=seed, stratify=ytr)
    return (torch.from_numpy(xtr), torch.from_numpy(ytr).long(),
            torch.from_numpy(xva), torch.from_numpy(yva).long(),
            torch.from_numpy(xte), torch.from_numpy(yte).long())


def add_noise(x, snr_db, gen):
    """AWGN at a given SNR, on the already normalised input."""
    p = x.pow(2).mean().sqrt()
    sigma = p / (10 ** (snr_db / 20.0))
    return x + torch.randn(x.shape, generator=gen) * sigma


@torch.no_grad()
def evaluate(model, x, y, snr=None, gen=None, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        if snr is not None:
            xb = add_noise(xb, snr, gen)
        preds.append(model(xb).argmax(1))
    preds = torch.cat(preds).numpy()
    return accuracy_score(y.numpy(), preds), f1_score(y.numpy(), preds, average='macro')


def run(variant, epochs=25, lr=1e-4, bs=32, log_every=0):
    xtr, ytr, xva, yva, xte, yte = load()
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = len(xtr); t0 = time.time(); hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) == 1:
                continue
            opt.zero_grad()
            loss = crit(model(xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        va, _ = evaluate(model, xva, yva)
        hist.append(round(va, 4))
        if log_every and ep % log_every == 0:
            print(f'  {variant} ep{ep:02d} val={va:.4f} loss={tot/n:.3f}', flush=True)
    acc, f1 = evaluate(model, xte, yte)
    gen = torch.Generator().manual_seed(123)
    n4, _ = evaluate(model, xte, yte, snr=4, gen=gen)
    n0, _ = evaluate(model, xte, yte, snr=0, gen=torch.Generator().manual_seed(123))
    return dict(variant=variant, acc=round(acc * 100, 2), f1=round(f1 * 100, 2),
                snr4=round(n4 * 100, 2), snr0=round(n0 * 100, 2),
                params=sum(p.numel() for p in model.parameters()),
                secs=round(time.time() - t0, 1), hist=hist)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['base'])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default='results.json')
    a = ap.parse_args()
    res = []
    for v in a.variants:
        print(f'>>> {v}', flush=True)
        r = run(v, epochs=a.epochs, lr=a.lr, log_every=5)
        print(json.dumps({k: r[k] for k in ('variant', 'acc', 'f1', 'snr4', 'snr0',
                                            'params', 'secs')}), flush=True)
        res.append(r)
        json.dump(res, open(a.out, 'w'), indent=1)
