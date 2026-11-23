"""
train.py —— 管线连通性验证（smoke test），不是论文级实验

只证明四件事：
  1) 数据生成 -> 特征 -> 模型 -> 指标 全链路可跑；
  2) 按 instance 切分（避免同一仿真样本泄漏到训练集）；
  3) 记录“极端容差健康样本上的误报率” TFA，作为协议的一部分；
  4) 四个模型在同一协议下可比。

注意：这里的数值来自单电路、小样本、CPU 短期训练，只能用于验证管线，
不能写进论文结果表。
"""
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import (TinySSMClassifier, CNN1D, GRUClassifier, TransformerClassifier,
                   count_params, quantize_ptq)


def load(path):
    df = pd.read_csv(path)
    L = 61
    mag = df[[f'mag_{i}' for i in range(L)]].values
    ph = df[[f'ph_{i}' for i in range(L)]].values
    fr = df[[f'f_{i}' for i in range(L)]].values
    X = np.stack([mag, ph, fr], axis=-1).astype(np.float32)
    y = pd.factorize(df['label'])[0].astype(np.int64)
    return X, y, df['instance_id'].values, df['group'].values, list(pd.factorize(df['label'])[1])


def macro_f1(pred, true, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return float(np.mean(f1s))


def run(model, Xtr, ytr, Xte, yte, Xc, epochs=12, bs=64, lr=1e-3, qat=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    if qat:
        enable_qat(model)
    n = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]); yb = torch.from_numpy(ytr[idx])
            opt.zero_grad(); out = model(xb)
            loss = lossf(out, yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pte = model(torch.from_numpy(Xte)).argmax(1).numpy()
        pc = model(torch.from_numpy(Xc)).argmax(1).numpy() if len(Xc) else np.array([])
    n_cls = int(ytr.max()) + 1
    res = dict(acc=float((pte == yte).mean()), macro_f1=macro_f1(pte, yte, n_cls))
    if len(pc):
        res['tfa'] = float((pc != 0).mean())      # 健康样本被判成故障的比例
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data.csv')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y, inst, grp, labels = load(args.data)
    # 按 instance 切分：同一仿真样本只出现在一侧
    uid = np.unique(inst)
    rs = np.random.default_rng(0)
    te_ids = set(rs.choice(uid, size=int(0.3 * len(uid)), replace=False))
    te = np.isin(inst, list(te_ids))
    tr = (~te) & (grp == 'main')
    corner = grp == 'corner'
    print(f"train={tr.sum()} test={te.sum()} corner={corner.sum()} L={X.shape[1]} classes={labels}")

    out = {}
    for name, ctor in [('CNN1D', CNN1D), ('GRU', GRUClassifier),
                       ('Transformer', TransformerClassifier), ('TinySSM', TinySSMClassifier)]:
        m = ctor(n_classes=len(labels))
        t0 = time.time()
        r = run(m, X[tr], y[tr], X[te], y[te], X[corner], epochs=args.epochs)
        r['params'] = count_params(m); r['time_s'] = round(time.time() - t0, 1)
        out[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # SSM + 训练后量化（PTQ）
    m = TinySSMClassifier(n_classes=len(labels))
    run(m, X[tr], y[tr], X[te], y[te], X[corner], epochs=args.epochs)
    quantize_ptq(m)
    m.eval()
    with torch.no_grad():
        pte = m(torch.from_numpy(X[te])).argmax(1).numpy()
        pc = m(torch.from_numpy(X[corner])).argmax(1).numpy()
    out['TinySSM+PTQ'] = dict(acc=float((pte == y[te]).mean()),
                              macro_f1=macro_f1(pte, y[te], len(labels)),
                              tfa=float((pc != 0).mean()),
                              params=count_params(m))
    print('TinySSM+PTQ', out['TinySSM+PTQ'])

    json.dump(out, open('smoke_results.json', 'w'), indent=2, ensure_ascii=False)
    print('已写出 smoke_results.json')
