#!/usr/bin/env bash
# Print per-suite progress, failures and the benchmark ETA.
set -uo pipefail
ROOT=${1:-/root/autodl-tmp/mixcnn}
cd "$ROOT/results/gpu" || exit 1

echo "time: $(date -Is)"
if [ -f "$ROOT/code/gpu/progress.py" ]; then
  (cd "$ROOT/code/gpu" && python progress.py 2>/dev/null)
fi
for suite in main baselines m1c jnu paderborn seu; do
  [ -d "$suite" ] || continue
  done_n=$(find "$suite" -maxdepth 1 -name '*.json' ! -name 'summary.json' \
    -exec grep -l '"complete": true' {} + 2>/dev/null | wc -l)
  pred_n=$(find "$suite/preds" -name '*.npz' 2>/dev/null | wc -l)
  printf '%-10s runs=%-5s preds=%-5s\n' "$suite" "$done_n" "$pred_n"
done

if [ -f benchmark.json ]; then
  python - <<'PY'
import json
b = json.load(open('benchmark.json'))
print('gpu', b.get('gpu'), 'vram', b.get('vram_gb'))
print('eta_gpu_hours', b.get('eta_gpu_hours'))
print('eta_wall_days', b.get('eta_wall_days'))
PY
fi

echo "--- recent errors ---"
grep -h -iE 'error|traceback|killed|out of memory' logs/*.log 2>/dev/null | tail -20
echo "--- disk ---"
df -h /root/autodl-tmp | tail -1
