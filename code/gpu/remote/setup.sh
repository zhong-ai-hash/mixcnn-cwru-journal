#!/usr/bin/env bash
# Prepare the AutoDL instance.  Usage: bash setup.sh [remote_root]
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/mixcnn}
cd "$ROOT"
mkdir -p results/gpu/{main,baselines,m1c,jnu,paderborn,seu,logs}

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

if "$PY" - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Using existing CUDA-enabled PyTorch"
else
  # RTX 50xx (Blackwell, sm_120) needs cu128; RTX 30xx works with cu121.
  if [ -z "${TORCH_INDEX:-}" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "$CC" ] && [ "${CC%%.*}" -ge 12 ]; then
      TORCH_INDEX=https://download.pytorch.org/whl/cu128
    else
      TORCH_INDEX=https://download.pytorch.org/whl/cu121
    fi
  fi
  echo "Installing PyTorch from $TORCH_INDEX"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
fi

"$PY" -m pip install numpy scipy scikit-learn pandas

"$PY" - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('vram_gb', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print('capability', torch.cuda.get_device_capability(0))

# One 1024-sample and one 4096-sample forward pass catches the two input
# regimes used by CWRU and Paderborn/JNU before the matrix starts.
x1 = torch.randn(4, 1, 1024)
x2 = torch.randn(2, 1, 4096)
for name in ('base', 'noPad'):
    import sys
    sys.path.insert(0, 'code/gpu')
    from models import build
    m = build(name)
    with torch.no_grad():
        print(name, '1024 ->', tuple(m(x1).shape), '4096 ->', tuple(m(x2).shape))
PY
