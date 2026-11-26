# 期刊版窗口数据

本目录存放四个数据集窗口化后的 `.npz` 与可核对的 `meta.json`。
`SHA256SUMS` 覆盖四个二进制 `.npz` 文件（`sha256sum -c` 可直接校验）；
`meta.json` 是文本文件，Git 会按 LF 规范化，因此不放进校验和。

| 文件 | 形状 | 说明 |
|---|---|---|
| `cwru_raw_windows.npz` | X (2000, 1024) | CWRU 10 类，40 条录音 × 50 窗，12 kHz |
| `jnu_windows.npz` | X (1464, 4096) | JNU 4 类，600/800/1000 rpm，50 kHz，每文件 122 窗 |
| `paderborn_windows.npz` | X (13896, 4096) | Paderborn 3 类，29 轴承 × 20 记录 × 24 窗，64 kHz |
| `seu_gear_windows.npz` | X (5616, 1024) | SEU 齿轮箱 3 类，936 样本 × 6 窗 |

## 字段

- CWRU：`X, y, load, size, file_id, rec_id, rpm`
- JNU：`X, y, speed, file_id, rec_id`
- Paderborn：`X, y, bearing, file_id`
- SEU：`X, y, group_id, sample_id`

`rec_id` / `file_id` / `group_id` 是切分单位；训练/测试按这些 id 分组，
避免同一录音或同一原始样本的窗口互渗。

## 重新生成

```bash
# CWRU：需要原始 .mat 目录
python code/journal/raw_cwru.py --raw-dir <CWRU_RAW>

# JNU：原始 CSV 放在 data/journal/jnu，再按每文件 122 窗平衡生成
python code/journal/raw_jnu.py --max-per-file 122

# Paderborn：默认自动找 Paper_X1/data/paderborn_raw
python code/journal/raw_paderborn.py --per-file 24 --mode 3class

# SEU：默认找 D:\SEU_gear_db
python code/journal/raw_seu_gear.py
```

## 来源与许可

- **CWRU**：Case Western Reserve University Bearing Data Center，公开数据。
  引用：W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics
  using the Case Western Reserve University data: A benchmark study,"
  *Mechanical Systems and Signal Processing*, 2015.
- **Paderborn**：Paderborn University Bearing DataCenter，公开数据。
  引用：C. Lessmeier et al., "Condition monitoring of bearing damage in
  electromechanical drive systems by using motor current signals of electric
  motors: A benchmark data set for data-driven classification," PHM Society
  European Conference, 2016.
- **JNU**：Jiangnan University bearing dataset，公开数据；本地副本按其
  原始文件命名与转速划分。
- **SEU 齿轮箱**：山东大学齿轮箱数据集的本地标注副本；本地副本未注明
  采样率，因此只作跨部件 sanity check。

使用这些数据时请引用各自原始文献；本仓库只提供窗口化与切分代码。
