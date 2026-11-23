#!/usr/bin/env bash
# Run the full journal suite on the GPU box.  Designed for tmux/nohup.
set -uo pipefail

ROOT=${1:-/root/autodl-tmp/mixcnn}
JOBS=${JOBS:-4}
EPOCHS=${EPOCHS:-20}
DRAWS=${DRAWS:-20}
PADERBORN=${PADERBORN:-1}
SEU=${SEU:-0}
LOG="$ROOT/results/gpu/logs"
mkdir -p "$LOG"
cd "$ROOT/code/gpu"

run() {
  local suite=$1
  echo "=== $suite $(date -Is) ==="
  python suite.py --suite "$suite" --jobs "$JOBS" --device cuda \
    --epochs "$EPOCHS" --draws "$DRAWS" \
    > "$LOG/$suite.log" 2>&1
  local rc=$?
  echo "=== $suite rc=$rc $(date -Is) ==="
  return $rc
}

echo "=== benchmark $(date -Is) ==="
python benchmark.py --device cuda --epochs "$EPOCHS" --draws "$DRAWS" \
  > "$LOG/benchmark.log" 2>&1
echo "=== benchmark rc=$? $(date -Is) ==="

run main

echo "=== m1c (clean+0dB, no BN banks) ==="
python suite.py --suite m1c --jobs "$JOBS" --device cuda \
  --epochs "$EPOCHS" --draws "$DRAWS" --snrs 0 --skip-bn \
  > "$LOG/m1c.log" 2>&1

run baselines

echo "=== jnu (no BN banks) ==="
python suite.py --suite jnu --jobs "$JOBS" --device cuda \
  --epochs "$EPOCHS" --draws "$DRAWS" --skip-bn \
  > "$LOG/jnu.log" 2>&1

if [ "$PADERBORN" = "1" ]; then
  run paderborn
fi

if [ "$SEU" = "1" ]; then
  run seu
fi

python stats.py --suite main --metric acc_snr_0 \
  --cond-a noise --cond-b clean > "$LOG/stats_main.json" 2>&1 || true
python stats.py --suite baselines --metric acc_snr_0 \
  --cond-a noise --cond-b clean > "$LOG/stats_baselines.json" 2>&1 || true
python stats.py --suite paderborn --metric acc_snr_0 \
  --cond-a noise --cond-b clean > "$LOG/stats_paderborn.json" 2>&1 || true

python progress.py > "$LOG/progress.log" 2>&1 || true
python bn_switch.py --suites main baselines paderborn seu \
  > "$LOG/bn_switch.log" 2>&1 || true

python - <<'PY'
import json
from pathlib import Path
root = Path('../../results/gpu')
out = {}
for suite in ('main', 'baselines', 'm1c', 'jnu', 'paderborn', 'seu'):
    p = root / suite / 'summary.json'
    if p.exists():
        out[suite] = json.loads(p.read_text()).get('runs')
(root / 'run_status.json').write_text(json.dumps(out, indent=1))
print(out)
PY
echo "ALL DONE $(date -Is)"
