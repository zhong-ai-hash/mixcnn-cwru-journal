# GPU 实验代码

设备无关：`--device cpu` 可在本机小规模冒烟，`--device cuda` 在 AutoDL
3080Ti 上跑全矩阵。

## 快速检查

```bash
python suite.py --suite main --models base --splits lobo0 --seeds 0 \
  --limit 2 --dry-run
python suite.py --suite main --models base --splits lobo0 --seeds 0 \
  --epochs 1 --draws 1 --snrs 0 --bn-snrs 0 --limit 1 --device cpu
```

## 主矩阵

```bash
python benchmark.py --device cuda --epochs 20 --draws 20
python suite.py --suite main --jobs 4 --device cuda \
  --epochs 20 --draws 20
```

- `main`：MIXCNN 六变体 × clean/noise × 24 切分 × 10 种子
- `baselines`：WDCNN/ResNet/DRSN/TCN/Transformer，同矩阵
- `m1c`：泄漏两臂 × 六变体 × 2 条件 × 20 种子
- `jnu`：三折留一转速 × 六变体 × 2 条件 × 10 种子
- `paderborn`：11 个模型 × 2 条件 ×（5 轴承分组 + 5 录音随机）× 5 种子
- `seu`：六变体 × 2 条件 × 5 样本分组折 × 5 种子（跨部件 sanity check）

`benchmark.py` 先跑每个模型一个完整 job（20 epoch、完整噪声网格、5 套 BN），
把 `eta_gpu_hours` 写进 `results/gpu/benchmark.json`，再决定开几个 worker。

断点续跑：直接重跑同一命令，已完成的 JSON 会跳过；`--force` 才重算。

## 输出

```text
results/gpu/<suite>/<stem>.json      # 指标、切分元数据、耗时
results/gpu/<suite>/preds/<stem>.npz # 逐窗口正确性（配对统计用）
results/gpu/<suite>/summary.json     # 跨种子/跨切分汇总
results/gpu/<suite>/summary.csv      # 同上，扁平表
results/gpu/<suite>/runs.csv         # 每个 run 一行
results/gpu/<suite>/manifest.json    # 环境、seeds/splits、耗时
results/gpu/<suite>/runtime.json     # 跨断点续跑的累计墙钟（sessions 列表）
results/gpu/benchmark.json|csv       # 单 job 实测与外推 ETA
```

## 运行开销

每个 run 的 JSON 都记录，可直接进论文的 cost 表：

- `train_secs` / `eval_secs` / `secs`：训练、评估、合计秒数
- `epoch_secs` / `epoch_secs_mean`：逐 epoch 秒数与均值
- `train_windows_per_s` / `eval_windows_per_s`：训练与推理吞吐（窗/秒）
- `peak_alloc_mb` / `peak_reserved_mb`：本 run 的 CUDA 峰值显存
- `micro_batch` / `grad_accum`：4096 点输入下的实际 micro-batch 与累积步数

`summary.json` 的 `overhead` 段给出整个 suite 的合计 GPU 秒、均值与峰值；
`runtime.json` 把每次启动（含中断后的续跑）的墙钟累加，
`manifest.total_secs_all_sessions` 是真实总开销，不是最后一次进程的耗时。
`runs.csv` 含上述全部字段，可直接做图或做表。

## 统计

```bash
python stats.py --suite main --metric acc_snr_0 \
  --cond-a noise --cond-b clean
python progress.py                     # 进度表，随时可跑
python bn_switch.py --suites main baselines paderborn seu
```

## 远端

```powershell
# 检查远端 GPU / torch / 磁盘
.\remote\sync.ps1 -HostName region-xx.autodl.com -Port 12345 -Direction check
# 上传 code/ 与 data/（排除 .venv、results、*.pt）
.\remote\sync.ps1 -HostName region-xx.autodl.com -Port 12345 -Direction up
# 回传汇总（默认不含 preds）
.\remote\sync.ps1 -HostName region-xx.autodl.com -Port 12345 -Direction down
# 远端
bash code/gpu/remote/setup.sh /root/autodl-tmp/mixcnn
tmux new -s mixcnn
bash code/gpu/remote/run_all.sh /root/autodl-tmp/mixcnn
bash code/gpu/remote/status.sh /root/autodl-tmp/mixcnn
```

## PyTorch 版本

- 3080Ti（sm_86）：AutoDL 自带 `torch 2.3 + cu121` 即可，AMP API 有回退。
- RTX 5090（Blackwell，sm_120）：必须 `cu128`；`setup.sh` 会自动检测
  `compute_cap >= 12` 并安装 cu128，也可显式指定：

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash code/gpu/remote/setup.sh
```
