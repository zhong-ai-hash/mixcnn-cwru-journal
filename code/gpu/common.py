"""Shared paths, device selection and seeding for the GPU suite."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[2]
CODE = PKG / 'code'
JOURNAL = CODE / 'journal'
DATA = PKG / 'data' / 'journal'
RESULTS = PKG / 'results' / 'gpu'

for path in (str(CODE), str(JOURNAL)):
    if path not in os.sys.path:
        os.sys.path.insert(0, path)


def pick_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.secs = time.time() - self.t0


def amp_enabled(device: torch.device, flag: bool) -> bool:
    return bool(flag and device.type == 'cuda')
