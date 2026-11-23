"""Final experiment: 3 seeds x (clean|noise) x 6 variants, plus BN recalibration.

For the noise-trained models we report accuracy both before and after
re-estimating the batch-norm statistics on clean windows -- the AdaBN step that
Table III suggests is missing.
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from variants import build

SEEDS = [0, 1, 2]
VARIANTS = ['base', 'nores', 'noPad', 'soft', 'multi', 'soft_multi']
DRAWS = 5
EPOCHS = 20
SNR_TRAIN = (-4.0, 10.0)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def load(seed):
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
    return x + torch.randn_like(x) * (p / (10 ** (snr_db / 20.0)))


@torch.no_grad()
def acc_of(model, x, y, bs=64):
    model.eval()
    p = torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)]).numpy()
    return accuracy_score(y.numpy(), p), f1_score(y.numpy(), p, average='macro')


@torch.no_grad()
def acc_noisy(model, x, y, snr, bs=64):
    a = []
    for d in range(DRAWS):
        torch.manual_seed(1000 + d)
        p = torch.cat([model(add_noise(x[i:i + bs], snr)).argmax(1)
                       for i in range(0, len(x), bs)]).numpy()
        a.append(accuracy_score(y.numpy(), p))
    return float(np.mean(a)), float(np.std(a))


def recalibrate_bn(model, xtr, bs=64):
    """AdaBN: re-estimate running statistics on clean windows."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d,)):
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average
    model.train()
    with torch.no_grad():
        for i in range(0, len(xtr), bs):
            model(xtr[i:i + bs])
    model.eval()


def train(variant, seed, aug):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr, xva, yva, xte, yte = load(seed)
    model = build(variant)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = len(xtr); t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            if len(idx) < 2:
                continue
            xb = xtr[idx]
            if aug:
                xb = add_noise(xb, float(np.random.uniform(*SNR_TRAIN)))
            opt.zero_grad()
            loss = crit(model(xb), ytr[idx])
            loss.backward(); opt.step()
    return model, xtr, xte, yte, round(time.time() - t0, 1)


def measure(model, xtr, xte, yte, recal=False, tag=''):
    out = {}
    acc, f1 = acc_of(model, xte, yte)
    out['acc'] = round(acc * 100, 2); out['f1'] = round(f1 * 100, 2)
    out['snr4'], out['snr4_std'] = [round(v * 100, 2) for v in acc_noisy(model, xte, yte, 4.0)]
    out['snr0'], out['snr0_std'] = [round(v * 100, 2) for v in acc_noisy(model, xte, yte, 0.0)]
    return out


if __name__ == '__main__':
    rows = []
    for seed in SEEDS:
        for v in VARIANTS:
            for aug in (False, True):
                cond = 'aug' if aug else 'clean'
                print(f'>>> seed{seed} {v} [{cond}]', flush=True)
                model, xtr, xte, yte, secs = train(v, seed, aug)
                row = dict(seed=seed, variant=v, cond=cond, secs=secs,
                           params=sum(p.numel() for p in model.parameters()))
                row.update({f'pre_{k}': val for k, val in measure(model, xtr, xte, yte).items()})
                if aug:
                    recalibrate_bn(model, xtr)
                    row.update({f'post_{k}': val for k, val in measure(model, xtr, xte, yte, recal=True).items()})
                print(json.dumps({k: row[k] for k in ('seed', 'variant', 'cond', 'secs')}
                                 | {k: row[k] for k in row if k.startswith(('pre_acc', 'pre_snr0', 'post_acc', 'post_snr0'))}),
                      flush=True)
                rows.append(row)
                json.dump(rows, open('results_final3.json', 'w'), indent=1)
    print('done', len(rows))
