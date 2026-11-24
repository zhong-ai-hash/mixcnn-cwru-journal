#!/usr/bin/env bash
# Journal matrix driver for the AutoDL instance.
# Benchmark has already been run; this only runs the suites, in cost order.
set -uo pipefail

ROOT=${1:-/root/autodl-tmp/mixcnn}
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
LOG="$ROOT/results/gpu/logs"
LEDGER="$ROOT/results/gpu/phase_ledger.csv"
mkdir -p "$LOG"
cd "$ROOT/code/gpu"

if [ ! -f "$LEDGER" ]; then
  echo "phase,start_utc,elapsed_secs,rc,runs_total" > "$LEDGER"
fi

phase() {
  local name=$1; shift
  local t0 runs_before runs_after
  t0=$(date +%s)
  runs_before=$(ls "$ROOT/results/gpu/$name"/*.json 2>/dev/null | wc -l)
  echo "=== $name start $(date -Is) ===" | tee -a "$LOG/run_all.log"
  "$@" >> "$LOG/$name.log" 2>&1
  local rc=$?
  runs_after=$(ls "$ROOT/results/gpu/$name"/*.json 2>/dev/null | wc -l)
  echo "=== $name end rc=$rc $(date -Is) ===" | tee -a "$LOG/run_all.log"
  echo "$name,$(date -u -d @"$t0" +%Y-%m-%dT%H:%M:%SZ),$(( $(date +%s) - t0 )),$rc,$(( runs_after - runs_before ))" >> "$LEDGER"
}

phase main python suite.py --suite main --jobs 12 --threads 1 \
  --device cuda --epochs 20 --draws 10 --bn-snrs 0 4

phase baselines python suite.py --suite baselines --jobs 12 --threads 1 \
  --device cuda --epochs 20 --draws 10 --bn-snrs 0 4

phase m1c python suite.py --suite m1c --jobs 12 --threads 1 \
  --device cuda --epochs 20 --draws 10 --snrs 0 --skip-bn

phase jnu python suite.py --suite jnu --jobs 6 --threads 1 \
  --device cuda --epochs 20 --draws 10 --skip-bn

phase paderborn python suite.py --suite paderborn --jobs 6 --threads 1 \
  --device cuda --epochs 20 --draws 10 --bn-snrs 0 4 \
  --models base noPad wdcnn --seeds 0 1 2

phase seu python suite.py --suite seu --jobs 6 --threads 1 \
  --device cuda --epochs 20 --draws 10 --skip-bn

echo "=== stats $(date -Is) ===" | tee -a "$LOG/run_all.log"
for s in main baselines paderborn; do
  python stats.py --suite "$s" --metric acc_snr_0 \
    --cond-a noise --cond-b clean >> "$LOG/stats_$s.log" 2>&1 || true
done
python progress.py >> "$LOG/progress.log" 2>&1 || true
python bn_switch.py --suites main baselines paderborn seu \
  >> "$LOG/bn_switch.log" 2>&1 || true
echo "ALL DONE $(date -Is)" | tee -a "$LOG/run_all.log"
