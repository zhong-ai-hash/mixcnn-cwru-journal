# 期刊版方案（v7，Measurement 主线）

日期：2026-11-22
前置：ICPHM/MLCIPR 会议版（v6）已投出；本文件定义期刊版的科学定位、实验矩阵、统计方案与三周排期。

---

## 0. 决策与边界

已确认的条件：

- 第一目标：**Measurement**（Elsevier，Q1）。
- 备选顺序：IEEE Sensors Journal（快、传感+边缘）→ MST（保底）→ ISA Transactions（需把方法做重）。
- 数据：原始 CWRU `.mat` 位于
  `D:\paper\paper_A\data\CWRU\cwru_raw`；第二数据集选用 JNU，
  已纳入 `data/journal/jnu/`（4 类 × 3 转速，50 kHz）。
- 硬件：STM32/ESP32、普通 PC，可采购。
- 算力：CPU，周期 2–3 周。

由此确定三条硬边界：

1. **不做大网络、不做新 SOTA**。Measurement 要的是测量有效性与工程指南，不是又一个 99%。
2. **主实验控制在 60 CPU 小时以内**（4 核并行约 15 小时墙钟），保证 2–3 周内可完成并留出重跑余量。
3. **会议版必须被引用**，期刊版新增内容 ≥30%，且新增的是"录音级泛化 + 方差分解 + 可部署切换"，不是把正文拉长。

---

## 1. 一句话定位与标题

会议版回答："在窗口级固定切分上，99% 意味着什么？"

期刊版要回答："**这个 99% 里，有多少是测量假象？**"——把同一批结论放到非泄漏切分、跨负载、多种子、可部署条件下重新测量，并给出可操作的测量协议。

拟定标题（二选一）：

- **How Much of a 99% Bearing-Fault Benchmark Is Measurement Artifact? A Recording-Level, Five-Seed Re-evaluation**
- From Window to Recording: Quantifying Leakage, Noise Sensitivity, and Normalization Effects in a Bearing-Fault Benchmark

推荐第一个：问句直接点出"测量假象"，与 Measurement 的期刊气质一致；副标题交代方法学。

---

## 2. 与会议版的差异清单（审稿人可见的 delta）

| 维度 | 会议版 v6 | 期刊版 v7 | 性质 |
|---|---|---|---|
| 切分 | 随机窗口 70/10/20 | 留一负载（LOBO）× 4 折 + 同数据随机切分对照 | 新实验 |
| 泄漏量化 | 未做 | 随机切分与录音级切分的准确率差，逐变体报告 | 新结论 |
| 种子 | 5 种子、单一 split | 4 折 LOBO + 随机折 × 5 种子，方差分解（split vs seed） | 新统计 |
| 基线 | 仅 FFT+logreg | WDCNN、DRSN、ResNet1D、FFT+logreg | 新对比 |
| 噪声 | 逐窗口 AWGN，5 档 | 同协议 + 噪声水平估计器 | 延续+扩展 |
| BN | oracle 双库（已知 SNR） | 可部署的噪声感知切换（估计→选库） | 新方法 |
| 校准 | 无 | ECE、NLL、risk–coverage | 新测量指标 |
| 部署 | 理论 FLOPs | ONNX/INT8 + MCU 实测延迟/内存 | 新测量 |
| 第二数据 | 无 | 独立数据集验证或跨域 | 新外部效度 |
| 篇幅 | 5 页 conference | 10–12 页期刊，单栏 | — |

---

## 3. 实验矩阵

单位 run = 一次训练（20 epoch，MIXCNN 规模）。本机实测：4 线程训练 110 s，整条命令约 5 min；1 线程约 8 min。

### M1 录音级切分与泄漏量化（核心）

数据：原始 12k drive-end，10 类（N + IR/B/OR × 007/014/021），每类 4 个负载文件，1024 点窗，stride 1024；**每文件均匀取 50 个窗**（`--max-per-file 50`），保留整段录音覆盖，训练集约 1500 窗，是会议版 707 的约两倍。全窗（约 118/文件）只在最终确认时抽一个 run 复核。

本机镜像的 `X098/X099` 两个 normal 文件没有 RPM 字段，负载按文件表回退；
meta JSON 会把这两条标为 `load_source=file_table`，其余 38 条标为 `rpm`。
这是数据源缺失，不是用估计值伪装成实测转速。

切分：

- **LOBO**：留一负载，4 折；每折训练集内部再切 90/10 做验证；测试集为被留出负载的全部窗口。
- **随机窗口对照**：在同一批原始窗口上复刻会议版的随机 70/10/20，用于量化"同录音窗口互渗"带来的乐观偏差。
- 可选 **M1b 故障尺寸外推**：训练 007+014，测试 021（3 个尺寸中留一个），检验严重度外推。

运行矩阵：

- 变体：Ref、NoPad（只保留这两个，避免矩阵爆炸）。
- 条件：clean、noise。
- 种子：0/1/2/3/4（与会议版一致，且方差分解的种子自由度更足）。
- 折：LOBO 4 + 随机 1 = 5。

计算：5 × 2 × 2 × 5 = **100 runs**；按本机实测（1 线程约 8 min/run），
4 进程并行约 3–5 小时墙钟。

产出：Table「随机切分 vs LOBO」、逐折准确率、泄漏差值及配对 CI。

### M2 外部基线（固定切分，可加一折 LOBO）

基线：WDCNN、DRSN-1D、ResNet-1D、FFT+logreg（已有）。

- 条件：clean、noise；种子 0/1/2；切分：固定随机 + LOBO 折 0。
- 运行数：4 × 2 × 3 × 2 = 48 runs；DRSN 较重，估 **12–18 CPU 小时**。
- 若时间紧，先做固定切分 24 runs（约 6 小时），LOBO 只补 WDCNN。

作用：证明"不是我们没调好一个模型"，同时给 Measurement 审稿人一个熟悉的参照系。

### M3 第二数据集（外部效度）

选用 **JNU 独立轴承数据集**，源目录
`D:\paper\paper_D\ZJL_PE_IEEE_Sensor_J\data\extra_datasets\JNU-Bearing-Dataset-main`，
已复制到 `data/journal/jnu/`。采样率 50 kHz；4 类为 N、IB（内圈）、
OB（外圈）、TB（滚动体）；3 个转速为 600/800/1000 rpm；每文件单一
0.3 mm × 0.05 mm 故障尺寸。

主验证采用“协议跨数据集复现”：600/800 rpm 训练，1000 rpm 测试，
每文件均匀取 50 窗（共 600 窗，每类 150 窗）。运行
Ref/NoPad × clean/noise × 3 种子 = **12 runs ≈ 5–6 小时**。若时间允许，
再补 3 折留一转速作为稳健性检查，但不把 12-run 主表扩成 36-run。

窗口长度用 **4096 点**，不是 1024。50 kHz 下 4096 点为 82 ms，与 CWRU
的 1024@12 kHz（85 ms）物理时长对齐，同时保留 JNU 原生 0–25 kHz 频带。
代价是同一卷积核覆盖的物理时间比 CWRU 短 4 倍，感受野不严格一致，需在
限制中写明。若审稿人坚持严格感受野匹配，备选方案是抗混叠重采样到
12 kHz + 1024 点，但会削掉高频共振带，默认不做。

不使用 Paderborn（本机目录为空）、IMS（缺少该 4 类监督标签）或 SEU
（齿轮箱跨部件）来冒充“独立轴承验证”。

### M4 噪声感知 BN 切换（把 oracle 变成方法）

不改网络，只改推理路径：

1. 噪声水平估计器（轻量、无需训练或极轻训练）：谱平坦度、高频能量比、RMS/峭度组合；在验证集上标定 SNR 分档阈值。
2. 按估计档位选择 clean-BN 或 noise-BN 统计量。
3. 报告：切换准确率、误切换的准确率代价、相对 oracle 的差距、推理额外开销（µs、字节）。
4. 对照：单一 BN、oracle、AdaBN、测试时自适应（若实现成本可控）。

实现前提：三个 BN 库必须在**同一组噪声抽样**上评估，否则窗口级配对无效。
`protocol_v3.py` 已把 `bn_clean`、`bn_noise_0`、`bn_noise_4` 的 `seed_base`
统一为 40000，并用 `--save-state` 保存训练后、重校准前的权重（含训练 BN
统计量），供 `bn_switch.py` 复算。

计算：**<1 CPU 小时**（复用已训练权重与校准集）。

这是期刊版相对会议版最重要的"方法增量"，也是投 ISA Transactions 时的主线。

### M5 校准与不确定性

- ECE（15 bin）、NLL、Brier、risk–coverage。
- 准确率的 Wilson 区间；配对差的层级 bootstrap（先重采样折，再重采样种子）。
- 报告 1 个测试窗口 = 多少个百分点，避免"差 0.2 点"的过度解读。

计算：**<1 小时**。

### M6 边缘部署测量

- 导出：ONNX → INT8 动态/静态量化；同时导出 C 头文件（权重数组）。
- PC 侧：ONNX Runtime 延迟（单线程、批量 1、1000 次取中位数）、模型大小、峰值内存。
- MCU 侧：STM32Cube.AI 或 TFLite Micro 部署，报告 flash/RAM/cycles；至少给出一次实测。
- 对照：Ref 与 NoPad 的实测延迟比，与理论 FLOPs 比（38.8M vs 153.0M）对照。

计算：**<1 小时**（不含烧录与串口采集）。

作用：Measurement 作为一节"deployment cost"；若改投 Sensors Journal，则升级为核心贡献。

### 汇总

| 模块 | runs | 估计 CPU 小时 |
|---|---:|---:|
| M1 LOBO + 随机对照 | 100 | 13–15 |
| M2 外部基线 | 48 | 8–12 |
| M3 第二数据集 | 12 | 2–3 |
| M1b 尺寸外推（可选） | 12 | 2 |
| M4–M6 | — | 3 |
| 重跑余量 | — | 6 |
| **合计** | **172** | **34–41** |

按本机实测（单 run 约 8 min，1 线程）与 4 进程并行，墙钟约 10–12 小时；
分散到 2–3 周、每天跑 4–6 小时即可完成，不影响白天使用机器。

---

## 4. 统计方案

1. **方差分解**：以折和种子为随机效应拟合 accuracy ~ 1 + (1|fold) + (1|seed)，报告各方差分量与 ICC。回答"文献里报的差异，有多少只是切分运气"。
2. **配对比较**：同一折、同一种子下配对；跨折汇总用层级 bootstrap（折→种子→窗口三层，窗口层只用于 McNemar，不用于独立样本 CI）。
3. **等价检验**：沿用 TOST，但边际预注册为 ±1.0 点（会议版已证明 ±0.5 不可达）；在 LOBO 下复核是否仍成立。
4. **泄漏差**：随机切分 − LOBO，逐变体、逐条件给点估计与 95% CI。
5. **多重比较**：主表 4 个变体 × 2 条件，Holm 校正；探索性分析明确标注。

---

## 5. 图表清单（v7）

1. Fig. 1 协议示意：原始录音 → 窗口 → LOBO 折 → 训练/测试（替换会议版架构图的位置，架构图移附录）。
2. Fig. 2 泄漏图：随机切分 vs LOBO 的准确率（clean 与 0 dB 两组柱/点）。
3. Fig. 3 噪声曲线：Ref/NoPad × clean/noise 训练，5 档 SNR，LOBO 均值 ± 折间 SD。
4. Fig. 4 方差分解：折间 vs 种子间 vs 残差。
5. Fig. 5 噪声感知切换：估计 SNR vs 实际 SNR，误切换代价。
6. Fig. 6 部署：延迟/内存/FLOPs 对照（Ref vs NoPad）。
7. Table 1 协议对照（会议版 vs 期刊版）。
8. Table 2 主结果（LOBO，4 变体 × 2 条件 × 3 种子）。
9. Table 3 外部基线。
10. Table 4 BN 2×2 + 切换策略。
11. Table 5 校准指标。
12. Table 6 部署测量。
13. 附录：逐类召回、逐折明细、超参数、复现命令。

---

## 6. 三周排期

**第 1 周（数据与主实验）**

- D1：原始 CWRU 重建（`code/journal/raw_cwru.py`），核对每类/每负载窗口数、RPM、缺失文件。
- D2：实现 `protocol_v3.py`（LOBO + 随机对照），先跑 1 折 × 1 变体 × 1 种子做冒烟测试。
- D3–D5：跑满 M1 的 60 runs；同时准备第二数据集窗口化脚本。
- 周末：M1 汇总表与泄漏图初版。

**第 2 周（基线与方法）**

- D6–D8：M2 外部基线（先固定切分，再补 LOBO 折 0）。
- D9：M3 第二数据集 12 runs。
- D10：M4 噪声感知切换 + M5 校准。
- 周末：主表、BN 表、校准表定稿。

**第 3 周（部署与写作）**

- D11–D12：M6 导出、INT8、MCU 实测；PC 侧计时。
- D13–D14：统计（方差分解、层级 bootstrap、TOST 复核）、图 1–6 定稿。
- D15–D17：写 v7 正文（Measurement 格式，单栏，10–12 页），补参考文献（目标 35–45 条）。
- D18–D19：数字审计（复用 `audit_numbers.py` 思路）、复现包、内部审读。
- D20：定稿。

---

## 7. 目录与产物

期刊版所有新增代码、数据产物、结果、图表和论文稿统一放在 `journal/`
命名空间下，不覆盖会议版 v6 的任何文件。

```
code/journal/
  raw_cwru.py          原始 .mat → 带元数据的窗口集
  protocol_v3.py       LOBO / 随机对照 / 训练评估
  baselines.py         WDCNN / DRSN / ResNet1D
  bn_switch.py         噪声估计 + BN 选择 + 校准
  edge_export.py       ONNX / INT8 / C 头文件 / 计时
  make_v7_tables.py    统计与表格
  make_v7_figures.py   图
data/journal/
  cwru_raw_windows.npz
  cwru_raw_meta.json
  jnu/                 第二数据集 JNU-Bearing-Dataset-main
results/journal/
  runs/  preds/  summary.json  effects.json
  tables/  figures/
paper/journal/
  main_B_v7.tex / .pdf
```

默认路径已写入脚本：`raw_cwru.py` 输出到 `data/journal/`，`protocol_v3.py`
从 `data/journal/cwru_raw_windows.npz` 读取并输出到 `results/journal/`。
如需临时测试，使用 `--out-dir` 指向 `results/journal/smoke/`，不要污染主结果。

---

## 8. 风险与止损

1. **原始文件命名/变量名不标准**：`raw_cwru.py` 先做扫描与试运行，不直接跑全量；元数据核对通过后再训练。
2. **LOBO 下准确率大幅下降**：这不是坏消息，是论文的核心发现；但要在摘要里提前框定"测量有效性"而非"模型失效"。
3. **第二数据集与 CWRU 类别体系不一致**：只做"协议可迁移"的结论，不强行对齐类别语义。
4. **MCU 工具链不可用**：退化为 ONNX Runtime + 理论 MACs，并在文中明确"PC 模拟，非嵌入式实测"；此时 Sensors Journal 路线暂缓。
5. **时间超支**：优先保 M1 + M4 + M5（这三块已足以支撑 Measurement）；M2 可减到 WDCNN 一个基线；M3 可缩为单数据集。

---

## 9. 需要你提供

1. 已确认：原始 CWRU 路径为 `D:\paper\paper_A\data\CWRU\cwru_raw`。
2. 已确认：第二数据集为 JNU，采样率 50 kHz，4 类（N/IB/OB/TB），
   3 转速（600/800/1000 rpm），落盘到 `data/journal/jnu/`。
3. MCU 具体型号与可用工具链（STM32Cube.AI / TFLite Micro / ESP-DL），决定导出格式。
4. 目标投稿月份（决定是否先出 Sensors Journal 短版）。

---

## 10. M1 结果与叙事修正（2026-11-22）

M1 已完成 100/100（5 折 × 2 变体 × 2 条件 × 5 种子），逐 run 结果见
`results/journal/M1_results_summary.md`。三个结论改变了期刊版的主线：

1. **协议差是真实的，但含义要拆开。** 随机窗口切分相对 LOBO 的 clean
   精度高估 3.10–4.23 点；clean 训练的 NoPad 在 0 dB 上高估 8.74 点；
   一旦加噪训练，差距降到 1.35–1.91 点。注意 random 与 LOBO 的测试集
   构成不同（前者混合四个负载），所以这是"常用协议相对录音级协议的
   乐观偏差"，不能全部归因于同录音窗口泄漏。需要 M1c 对照把两者拆开。
2. **架构效应是条件性的。** clean 训练下 NoPad 的 0 dB 优势为
   +29.64 ± 8.34 点（t=7.95），但 clean 精度只差 +1.28 点（不显著）；
   加噪训练后两个条件都等价（−0.20 与 −0.31）。所以 NoPad 不是"更准的
   架构"，而是"训练/测试噪声失配时更稳的架构"。
3. **BN 是补偿机制，不是独立第二杠杆。** clean 训练模型在 0 dB 下，
   BN 重校准把 base 从 37.73 拉到 86.21、NoPad 从 67.37 拉到 95.64；
   加噪训练后三套 BN 库基本无增益（97.53/97.31/97.14）。会议版
   "second lever" 的表述必须改为"失配条件下的补偿"。

新的中心句可以写成：**噪声训练是主效应，它同时消除了切分泄漏差、
架构差与 BN 差；架构和归一化的可见效应只在训练/测试噪声失配时才出现。**
这比会议版"架构不可排序"更精确，也更符合 Measurement 的测量条件视角。

两处必须先补的方法学缺口：

- **BN 行未对齐。** `protocol_v3.py` 的 as-trained 评估用
  `seed_base=10000`，三个 BN 库用 40000，均值可比但窗口级不配对。
  `reeval_aligned.py` 用保存的 `.pt` 在统一 `seed_base=50000` 下重评
  四套配置，不重训，约 1 小时。
- **泄漏与跨负载混淆。** 建议 M1c：固定测试集为 load0 每条录音的后
  25 窗（250 窗），Arm A 用 load0 以外的 1500 窗训练；Arm B 把其中
  250 窗换成 load0 每条录音的前 25 窗。两臂训练集大小相同、测试集
  完全相同，差值即同录音泄漏的净效应。40 runs，约 2–3 小时。

---

## 11. 执行补充（2026-11-23）

- `reeval_aligned.py` 已完成 100/100 run 的统一噪声抽样重评，输出到
  `results/journal/aligned/`；评估参数未改，只把 as-trained 与三套 BN
  统一到 `eval_seed=50000`。为控制计算量，对齐重评取 clean + 0 dB，
  完整五档噪声曲线仍由 M1 的 `pre` 提供。
- `m1c_leak.py` 已完成并通过 dry-run：两臂均为 1500 训练窗 / 250 测试窗，
  `no_overlap` 与测试录音重叠 0 条，`recording_leak` 覆盖 10 条测试录音、
  250 个泄漏窗。替换窗取 load3 每条录音前 25 窗，按类别和录音内位置匹配。
- `baselines.py` 的 WDCNN/ResNet-1D/DRSN-1D 已通过 dry-run；建议先跑
  WDCNN+ResNet-1D 的 100 run，DRSN 单独排队。
- `protocol_jnu.py` 已完成并通过 dry-run：默认 600/800 rpm 训练、
  1000 rpm 测试，train/val/test = 360/40/200，Ref/NoPad × clean/noise ×
  3 种子 = 12 run。
- 当前环境使用 `code/journal/.venv`（Python 3.12），旧的
  `D:\paper\paper_B\venv` 已因底层 Python 3.13 被移除而失效。

### 11.1 M1c 结果：泄漏是天花板，不是被“吸收”的差（2026-11-23）

`m1c_leak.py` 已完成 40/40 run。两臂测试集完全相同（load0 每条录音后
25 窗，250 窗），训练集同为 1500 窗；`recording_leak` 把 load3 的前 25 窗
换成 load0 每条录音的前 25 窗，训练/测试窗口索引无重叠。

| 训练条件 | 变体 | clean：无泄漏 → 泄漏 | 效应（95% CI） | p |
|---|---|---:|---:|---:|
| clean | base | 92.00 → 98.56 | +6.56 (+0.62, +12.50) | 0.037 |
| clean | NoPad | 93.12 → 98.96 | +5.84 (+0.77, +10.91) | 0.033 |
| noise | base | 97.44 → 98.48 | +1.04 (−0.26, +2.34) | 0.090 |
| noise | NoPad | 97.12 → 98.64 | +1.52 (+0.98, +2.06) | 0.0015 |

关键读法：四个“泄漏”格都停在 98.5–99.0，跨训练条件和 padding 变体只差
0.5 点；变的是无泄漏基线（clean→noise 抬高 4–5 点）。因此“噪声训练吸收
泄漏差”是错误归因，正确表述是“泄漏给出一个稳定的记录内插值天花板，噪声
训练把诚实基线抬近天花板”。0 dB 下 clean-trained base 两端都在 32% 地板，
该格不可解释；noise-trained 两变体仍有 +2.07 / +2.19 的显著效应。

正文相应改写：random−LOBO 的 3–4 点 = 跨负载泛化差（主体）+
记录内泄漏（噪声训练后 1–2 点）；不能把整段差值称为泄漏。

### 11.2 运行状态与算力取舍

- `protocol_jnu.py` 的 12 run 已完成。4096 点输入使 Mixconv 主体在
  1024 长度上运行（CWRU 为 256），单 run 约 75 分钟，墙钟约 2.5 小时。
  跨转速 clean 精度 ~82%；噪声训练在 0 dB 只提高 +5.5/+7.8 点
  （p=0.38/0.26），4 dB 提高 +8.5/+9.1（p=0.16/0.33），均未显著；
  变体差 ≤3.7 点且 p>0.24。结论按“外部效度边界”写，不按第二个主结果写。
- DRSN-1D 实测 203.3 MFLOPs，约为 MIXCNN 的 5.2 倍；单 run 预计
  2–2.5 小时，50 run 需 100–125 CPU 小时。默认不跑，只在审稿要求时做
  `random + load0 × 2 条件 × 3 种子` 的 12-run 子集。
- M1c 的绝对精度（92.0/93.1）不能与 M1 LOBO（96.05/95.52）直接比较：
  测试子集不同（录音后段 250 窗 vs 全段随机 500 窗），只用于臂间相减。

### 11.3 环境一致性（2026-11-23 发现，必须在投稿前解决）

M1 主矩阵跑于旧 Python 3.13 环境；BN 对齐、M1c、基线与 JNU 跑于项目内
固定的 Python 3.12 + PyTorch 2.9.1 环境。用同一个 `.pt` 重评，clean 精度
从 86.6 变为 88.6（seed0），逐种子最大差 3.0 点；测试标签、切分、权重均
一致，差异来自运行时数值路径。组均值差约 0.5–1.2 点。

因此论文不能把 M1 的 `pre` 与 aligned/基线的绝对精度混排。处理方案：
`reeval_curve.py` 在新环境重评 100 个 `.pt` 的 clean + 5 档 SNR（只评估、
不重训），输出 `results/journal/curve/summary.json`；表 II/III、噪声曲线与
基线对比统一改用 `curve/` + `aligned/`。M1 的 `pre` 仅作为旧环境记录保留。

详细状态、命令和产物路径见
`results/journal/期刊版项目进度_2026-11-23.md`。
