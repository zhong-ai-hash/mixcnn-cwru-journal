# MIXCNN-CWRU Journal

Journal extension of the MIXCNN/CWRU re-evaluation.  The conference version is
kept separately at
[`zhong-ai-hash/mixcnn-cwru-reevaluation`](https://github.com/zhong-ai-hash/mixcnn-cwru-reevaluation).

The journal study asks a measurement question rather than proposing a new
network: **how much of the reported 99% accuracy is attributable to the split
unit, the training noise distribution, and the BatchNorm statistics?**

Status: code, windowed data, and the full GPU experiment suite are complete;
the GPU matrix is pending.  Manuscript in preparation.

## What is here

```text
code/gpu/          GPU suite: splits, 11 models, AMP, AWGN, BN banks,
                   benchmark, statistics, BN switching, progress
code/journal/      data builders (CWRU/JNU/Paderborn/SEU) and CPU reference
code/*.py          the audited MIXCNN, variants, and earlier CPU scripts
data/journal/      windowed .npz + meta + SHA256SUMS
results/cpu_reference/  pre-GPU reference summaries and per-run JSON
docs/              journal plan, GPU runbook, dataset notes, review plan
paper/             manuscript placeholder
```

## Data

| File | Windows | Classes | Sampling | Split unit |
|---|---:|---:|---|---|
| `cwru_raw_windows.npz` | 2,000 | 10 | 12 kHz, 1024 pt | recording (`rec_id`) |
| `jnu_windows.npz` | 2,196 | 4 | 50 kHz, 4096 pt | speed / recording |
| `paderborn_windows.npz` | 13,896 | 3 | 64 kHz, 4096 pt | bearing / file |
| `seu_gear_windows.npz` | 5,616 | 3 | not documented | original sample |

`data/journal/SHA256SUMS` verifies the upload.  Raw datasets are not included;
`data/journal/README.md` documents provenance, licences, and regeneration
commands.

## Experiment matrix

| Suite | Runs | Purpose |
|---|---:|---|
| `main` | 2,880 | 6 MIXCNN variants x clean/noise x 24 splits x 10 seeds |
| `baselines` | 2,400 | WDCNN, ResNet-1D, DRSN-1D, TCN, Transformer |
| `m1c` | 480 | same-recording leakage isolation, 2 arms x 20 seeds |
| `jnu` | 360 | leave-one-speed-out external validity |
| `paderborn` | 1,100 | leave-bearing-out + file-random external validity |
| `seu` | 300 | cross-component sanity check (optional) |
| **total** | **7,520** | |

Protocol notes:

- the scaler is fit on the training split only;
- no validation split is reserved; every non-test recording is used for
  training;
- 20 epochs, fixed budget, no early stopping and no per-baseline tuning;
- 1024-point inputs use batch 128; 4096-point inputs use micro-batch 32 with
  4-step gradient accumulation (effective batch 128) to fit a 12 GB GPU;
- every run stores clean accuracy, 7 SNR points x 20 noise draws, 5 BN banks,
  per-class recall, macro-F1, and per-window correctness.

## Quick start

```bash
# CPU smoke
cd code/gpu
python suite.py --suite main --models base --splits lobo0 --seeds 0 \
  --epochs 1 --draws 1 --snrs 0 --bn-snrs 0 --device cpu

# GPU: setup, timing benchmark, full matrix
bash remote/setup.sh /root/autodl-tmp/mixcnn
python benchmark.py --device cuda --epochs 20 --draws 20
JOBS=4 bash remote/run_all.sh /root/autodl-tmp/mixcnn
bash remote/status.sh /root/autodl-tmp/mixcnn
```

See `docs/期刊版GPU运行手册_2026-11-24.md` for the full runbook.

## Key references

- W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using
  the Case Western Reserve University data: A benchmark study," *MSSP*, 2015.
- C. Lessmeier et al., "Condition monitoring of bearing damage in
  electromechanical drive systems by using motor current signals," *PHM
  Society European Conference*, 2016.

## Licence

Code: MIT (`LICENSE`).  Data and text: CC-BY-4.0
(`LICENSE-CC-BY-4.0.txt`).  Original datasets remain under their own terms;
please cite the source papers listed in `data/journal/README.md`.
