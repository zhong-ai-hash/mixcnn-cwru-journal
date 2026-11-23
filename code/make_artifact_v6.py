"""Build a self-contained local release artifact for manuscript v6."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "artifact_v6"


README = """# MIXCNN five-seed re-evaluation artifact

This archive accompanies the manuscript:

> Architecture, Noise Training, or Batch Normalization? A Five-Seed
> Re-evaluation of a Lightweight Bearing-Fault Classifier

Version: v6 ({today})

Repository: https://github.com/zhong-ai-hash/mixcnn-cwru-reevaluation

## Contents

- `code/`: training, evaluation, table, figure, and audit scripts.
- `data/cwru_data.npy` (1010 x 1024) and `data/cwru_label.npy`
  (10 classes, 101 examples each).
- `results_v2/runs/`: one JSON record for each of the 60 runs.
- `results_v2/preds/`: per-window correctness vectors as NPZ files.
- `results_v2/summary.json` and `effects.json`: aggregated estimates,
  paired confidence intervals, and equivalence tests.
- `results_v2/table*.tex` and `results_v2/fig_*.pdf`: manuscript tables
  and figures.
- `paper/`: the v6 TeX source, compiled PDF, bibliography, and IEEEtran
  style files.

## Protocol

The frozen split uses seed 2026 and contains 707 training, 101 validation,
and 202 test windows. The validation set is reserved. A per-position
StandardScaler is fitted on the 707 training windows only. Five seeds
change initialization and batch order, not the split. Evaluation uses
per-window additive white Gaussian noise, ten draws at each SNR in
{-4, 0, 4, 8, 12} dB, with the same noise seeds across variants.

The packaged arrays contain no recording identifiers. All claims are
therefore window-level claims from one fixed split, not recording-level or
cross-machine claims.

## Reproduction

From the artifact root:

```bash
./run_all.sh
```

This regenerates tables, figures, and the numeric audit from the supplied
run records. To rerun the full training matrix first:

```bash
RUN_TRAINING=1 ./run_all.sh
```

Equivalent manual commands:

```bash
cd code
python protocol_v2.py --jobs 2 --threads 1
python make_v2_tables.py
python make_v3_figure.py
python make_v6_tables.py
python make_v6_figure.py
python audit_numbers.py
```

The full 60-run matrix takes about 11 CPU-hours on two cores and does not
require a GPU. Existing completed runs are skipped unless `--force` is
passed to `protocol_v2.py`.

## Integrity

`SHA256SUMS.txt` is in `sha256sum` format and covers every file except
itself. Verify on Linux with:

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -Path (Get-Content SHA256SUMS.txt |
  ForEach-Object { ($_ -split '  ', 2)[1] })
```

The data and label hashes are also recorded in
`results_v2/split_meta.json` and in every run record.

## Provenance and scope

`MIXCNN.py` and `model_2_view.py` are retained unmodified as the released
implementation under study. The corrected variants, protocol, and analysis
scripts are part of this work. The benchmark data derive from the CWRU
bearing-fault dataset; see Smith and Randall, MSSP 64-65 (2015), 100-131,
doi:10.1016/j.ymssp.2015.04.021.

The artifact supports the paper's reported numbers but does not establish
cross-recording generalization, performance under impulsive or
non-stationary noise, or deployment latency on embedded hardware.

## License

Original artifact code is MIT-licensed. `code/MIXCNN.py` and
`code/model_2_view.py` are unmodified upstream files and remain subject to
their upstream terms. Manuscript text, tables, figures, and packaged arrays
are CC BY 4.0, to the extent permitted by the CWRU source-data terms. See
`LICENSE` and `LICENSE-CC-BY-4.0.txt`.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.aux",
                                      "*.log", "*.out", "*.blg"),
    )


def prune_caches(dest: Path) -> None:
    """Remove interpreter and OS caches without touching source files."""
    root = dest.resolve()
    for cache in sorted(dest.rglob("__pycache__"), reverse=True):
        resolved = cache.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"Refusing to remove path outside artifact: {resolved}")
        shutil.rmtree(resolved)
    for pattern in ("*.pyc", "*.pyo", ".DS_Store"):
        for path in dest.rglob(pattern):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise RuntimeError(
                    f"Refusing to remove path outside artifact: {resolved}")
            resolved.unlink()


def artifact_files(dest: Path) -> list[Path]:
    """List release files, excluding Git metadata and the manifest itself."""
    files = []
    for path in dest.rglob("*"):
        rel = path.relative_to(dest)
        if not path.is_file():
            continue
        if ".git" in rel.parts or path.name == "SHA256SUMS.txt":
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(dest).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="overwrite an existing artifact in place")
    args = parser.parse_args()
    if DEST.exists() and (DEST / "SHA256SUMS.txt").exists() and not args.refresh:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {DEST}")
    if not DEST.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Artifact path is outside project root: {DEST}")

    DEST.mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "code", DEST / "code")
    copy_tree(ROOT / "data", DEST / "data")
    copy_tree(ROOT / "results_v2", DEST / "results_v2")
    for name in ("LICENSE", "LICENSE-CC-BY-4.0.txt", "CITATION.cff",
                 "requirements.txt", "run_all.sh", ".gitignore",
                 ".gitattributes"):
        shutil.copy2(ROOT / name, DEST / name)

    paper = DEST / "paper"
    paper.mkdir(exist_ok=True)
    for name in ("main_A_v6.tex", "main_A_v6.pdf", "refsA.bib",
                 "IEEEtran.cls", "IEEEtran.bst"):
        shutil.copy2(ROOT / "paper" / name, paper / name)

    (DEST / "README.md").write_text(
        README.replace("{today}", date.today().isoformat()), encoding="utf-8")
    prune_caches(DEST)

    files = artifact_files(DEST)
    lines = []
    for path in files:
        rel = path.relative_to(DEST).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    (DEST / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Built {DEST} with {len(files)} files.")


if __name__ == "__main__":
    main()
