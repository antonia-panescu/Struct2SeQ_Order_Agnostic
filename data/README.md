# Data

This directory holds *small* benchmark targets and metadata that travel
with the repository. Large training dumps and per-machine raw data live
elsewhere and are excluded from version control via `.gitignore`.

## Layout

```
data/
├── README.md                           (this file)
├── eterna100/
│   ├── eterna100_puzzles.tsv           Eterna100 puzzle targets (committed)
│   └── eterna100_targets_v2.csv        Cleaned puzzle-target CSV (committed)
└── eterna_training_dump/
    ├── README_eterna_puzzles_20250227_analysis.md
    ├── eterna100_overlap_report.csv    Overlap of training pool with
    │                                   Eterna100 (committed)
    └── eterna_puzzles_20250227.csv     **44 MB; NOT committed**
```

## What's committed vs not

| Path                                                        | Tracked? | Why |
|-------------------------------------------------------------|----------|-----|
| `eterna100/eterna100_puzzles.tsv`                           | yes      | Tiny, canonical benchmark targets |
| `eterna100/eterna100_targets_v2.csv`                        | yes      | Cleaned variant used by `evaluation/run_eterna100_*.sh` |
| `eterna_training_dump/README_eterna_puzzles_20250227_analysis.md` | yes | Provenance notes |
| `eterna_training_dump/eterna100_overlap_report.csv`         | yes      | Small analysis output (23 KB) |
| `eterna_training_dump/eterna_puzzles_20250227.csv`          | **no**   | 44 MB raw dump |
| `../top2M.csv`                                              | **no**   | 515 MB training set |

## How to obtain the un-tracked files

`top2M.csv` (training data):

```bash
# From RDAV cluster:
scp rdav:/gpfs/radev/home/afp38/scratch/Struct2SeQ_training_data/top2M.csv ~/
# or place at the repository root; launch_brev.sh searches both.
```

`eterna_puzzles_20250227.csv` (reference Eterna dump):
This is a one-shot dump of the public Eterna puzzle database as of
2025-02-27. The original source is the Eterna2 puzzle export. The CSV
in `eterna_training_dump/` is the unmodified dump.

If the file is missing locally, regenerate by re-running the export
from Eterna with `state.status = 'OK'` filtering, or copy it from a
known-good machine (e.g. `rdav:~/Struct2SeQ_training_data/`).

## Expected schemas

`eterna100/eterna100_puzzles.tsv`:

| Column        | Description                              |
|---------------|------------------------------------------|
| `id`          | Eterna puzzle ID                         |
| `name`        | Human-readable puzzle name               |
| `secstruct`   | Target dot-bracket structure             |
| `length`      | Length (nucleotides)                     |

`top2M.csv` (training set; columns of interest):

| Column            | Description                                          |
|-------------------|------------------------------------------------------|
| `sequence`        | Reference RNA sequence (240 nt window)               |
| `structure`       | Target dot-bracket structure                         |
| `ok_pred`         | OK-filter prediction passed (RibonanzaNet oracle)    |

Other columns are tolerated by `Dataset.py` but unused.
