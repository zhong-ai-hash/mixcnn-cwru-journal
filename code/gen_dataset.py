"""
gen_dataset.py —— 模拟电路软故障基准数据生成器（复数 MNA 解析求解）

为什么不用 ngspice：
  线性模拟电路（R/C/理想运放）的频响可以由节点导纳矩阵直接精确解出，
  无需 SPICE 迭代，单条样本 < 1 ms，方便做大规模蒙特卡洛扫描与可复现实验。

电路：Sallen-Key 二阶低通（K=2），标称 fc = 1/(2*pi*10k*10nF) = 1.59 kHz
故障：单个无源元件参数漂移到容差之外（软故障），标签 6 类（1 健康 + 5 故障）

用法：
  python gen_dataset.py --check            # 自检：打印 DC 增益与 -3dB 频率
  python gen_dataset.py --out data.csv     # 生成训练/测试集
"""
import argparse
import numpy as np
import pandas as pd

# ---------------- 复数 MNA 求解器 ----------------
def solve_ac(elements, n_nodes, source, freqs):
    """elements: list of ('R'|'C', node_i, node_j, value)
       source: ('V', node_p, node_n, amplitude) 接在 node_p 与地之间
       opamp: 单独处理，见 sallen_key_response
       返回 node 电压随频率的复数组 {node: ndarray}
    """
    n = n_nodes
    n_extra = 1                      # 电压源支路电流
    A = np.zeros((n + n_extra, n + n_extra), dtype=complex)
    for kind, i, j, val in elements:
        if i < 0 or j < 0:
            continue
        y = 1.0 / val if kind == 'R' else None
        if kind == 'C':
            continue                 # 电容随频率变化，在循环内 stamp
        A[i, i] += y
        A[j, j] += y
        A[i, j] -= y
        A[j, i] -= y
    # 电压源 stamp：节点 source[1] 接地为参考
    k = n
    p, q, amp = source
    A[p, k] += 1.0
    A[k, p] += 1.0
    A[k, k] = 0.0
    A[k, q] -= 1.0
    b_base = np.zeros(n + n_extra, dtype=complex)
    b_base[k] = amp

    out = np.zeros((len(freqs), n), dtype=complex)
    for fi, f in enumerate(freqs):
        s = 2j * np.pi * f
        Af = A.copy()
        for kind, i, j, val in elements:
            if kind != 'C':
                continue
            y = s * val
            Af[i, i] += y
            Af[j, j] += y
            Af[i, j] -= y
            Af[j, i] -= y
        x = np.linalg.solve(Af, b_base)
        out[fi] = x[:n]
    return out


def sallen_key_response(R1, R2, C1, C2, R3, R4, freqs):
    """理想运放（nullor）Sallen-Key 二阶低通，返回 out 节点电压（复数组）。

    拓扑（node 0 为地）：
      1=in, 2=a, 3=b(运放 +), 4=out, 5=minus(运放 -, 经 R3 接 out、R4 接地)
      元件：R1(in,a)  R2(a,b)  C1(a,out)  C2(b,gnd)  R3(out,minus)  R4(minus,gnd)
    """
    IN, A_, B, OUT, MINUS = 0, 1, 2, 3, 4
    N = 5
    n_extra = 2                      # 电压源支路电流 + 运放输出电流

    def adm(M, i, j, y):
        M[i, i] += y; M[j, j] += y; M[i, j] -= y; M[j, i] -= y

    def gnd(M, i, y):
        M[i, i] += y

    A = np.zeros((N + n_extra, N + n_extra), dtype=complex)
    b = np.zeros(N + n_extra, dtype=complex)

    adm(A, IN, A_, 1.0 / R1)
    adm(A, A_, B, 1.0 / R2)
    adm(A, OUT, MINUS, 1.0 / R3)
    gnd(A, MINUS, 1.0 / R4)

    k_src, k_op = N, N + 1
    A[IN, k_src] += 1.0; A[k_src, IN] += 1.0; b[k_src] = 1.0
    # 运放：V(B) - V(MINUS) = 0，输出电流注入 OUT 节点
    A[k_op, B] += 1.0; A[k_op, MINUS] -= 1.0; A[OUT, k_op] += 1.0

    out = np.zeros(len(freqs), dtype=complex)
    for fi, f in enumerate(freqs):
        Af = A.copy()                # 关键：每个频点从"只有电阻/源"的基础矩阵重新开始
        adm(Af, A_, OUT, 2j * np.pi * f * C1)
        gnd(Af, B, 2j * np.pi * f * C2)
        x = np.linalg.solve(Af, b)
        out[fi] = x[OUT]
    return out


# ---------------- 数据生成 ----------------
NOMINAL = dict(R1=10e3, R2=10e3, C1=10e-9, C2=10e-9, R3=10e3, R4=10e3)
FAULTS = {
    'healthy': None,
    'R1_up':   ('R1', +0.20),
    'R2_up':   ('R2', +0.20),
    'C1_down': ('C1', -0.20),
    'C2_down': ('C2', -0.20),
    'R3_up':   ('R3', +0.30),   # 增益漂移
}
TOL = 0.05        # 元件容差 ±5%（软故障判据的分界线）


def make_instance(label, rng, drift=None, tol=TOL):
    p = dict(NOMINAL)
    for k in ('R1', 'R2', 'C1', 'C2', 'R3', 'R4'):
        p[k] = p[k] * (1.0 + rng.uniform(-tol, tol))
    if label != 'healthy':
        comp, d = FAULTS[label]
        p[comp] = NOMINAL[comp] * (1.0 + d)
    return p


def make_corner_healthy(rng):
    """极端容差但完全健康的样本：用于度量“容差误报率” TFA。
    所有元件同时偏向 ±5% 的角点，是健康样本里最容易误报的一类。"""
    p = dict(NOMINAL)
    sign = rng.choice([-1.0, 1.0])
    for k in p:
        p[k] = p[k] * (1.0 + sign * TOL * rng.uniform(0.8, 1.0))
    return p


def to_features(p, freqs, noise_db=0.30, rng=None):
    H = sallen_key_response(p['R1'], p['R2'], p['C1'], p['C2'], p['R3'], p['R4'], freqs)
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    ph = np.angle(H) / np.pi
    if rng is not None and noise_db > 0:
        # 测量噪声：幅度加高斯 dB 噪声，相位加小噪声
        mag = mag + rng.normal(0, noise_db, size=mag.shape)
        ph = ph + rng.normal(0, 0.02, size=ph.shape)
    return np.stack([mag / 20.0, ph, np.log10(freqs) / 3.0], axis=1)  # L x 3


def build(n_per_class=400, seed=0, split='instance'):
    rng = np.random.default_rng(seed)
    freqs = np.logspace(2, 5, 61)          # 100 Hz - 100 kHz, 61 点
    X, y, inst, grp = [], [], [], []
    labels = list(FAULTS)
    iid = 0
    for ci, lab in enumerate(labels):
        for _ in range(n_per_class):
            p = make_instance(lab, rng)
            X.append(to_features(p, freqs, rng=rng))
            y.append(ci)
            inst.append(iid); grp.append('main'); iid += 1
    # 极端容差健康集（只在测试侧使用）
    for _ in range(n_per_class // 2):
        p = make_corner_healthy(rng)
        X.append(to_features(p, freqs, rng=rng))
        y.append(0)
        inst.append(iid); grp.append('corner'); iid += 1
    return np.stack(X), np.array(y), np.array(inst), np.array(grp), labels, freqs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()

    if args.check:
        freqs = np.logspace(1, 5, 200)
        H = sallen_key_response(NOMINAL['R1'], NOMINAL['R2'], NOMINAL['C1'],
                                NOMINAL['C2'], NOMINAL['R3'], NOMINAL['R4'], freqs)
        gain_dc = 20 * np.log10(np.abs(H[0]))
        half = np.abs(H) / np.abs(H[0])
        idx = np.argmin(np.abs(half - 1 / np.sqrt(2)))
        print(f"DC 增益 = {gain_dc:.2f} dB (期望 6.02 dB, 即 K=2)")
        print(f"-3dB 频率 = {freqs[idx]:.0f} Hz (Q=1 时理论值 ≈ 1.272*fn ≈ 2025 Hz)")
        print(f"自然频率 fn = {1/(2*np.pi*10e3*10e-9):.0f} Hz")
        # 故障可分性自检
        f2 = np.logspace(2, 5, 61)
        Hh = to_features(NOMINAL, f2)[:, 0]
        Hf = to_features(dict(NOMINAL, R1=NOMINAL['R1'] * 1.2), f2)[:, 0]
        print(f"R1 +20% 与健康样本的幅度谱最大差异 = {np.abs(Hh-Hf).max()*20:.2f} dB")

    if args.out:
        X, y, inst, grp, labels, freqs = build(n_per_class=args.n)
        flat = X.reshape(len(X), -1)
        cols = [f'mag_{i}' for i in range(X.shape[1])] + \
               [f'ph_{i}' for i in range(X.shape[1])] + \
               [f'f_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(np.hstack([X[:, :, 0], X[:, :, 1], X[:, :, 2]]), columns=cols)
        df['label'] = [labels[i] for i in y]
        df['instance_id'] = inst
        df['group'] = grp
        df.to_csv(args.out, index=False)
        print(f"已写出 {args.out}: {df.shape}, 类别 {labels}")
