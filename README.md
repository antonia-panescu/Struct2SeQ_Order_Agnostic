# Struct2SeQ-bidir (RL)

Bidirectional / order-agnostic Q-learning extension of the Struct2SeQ
RNA inverse-folding model. Given a target dot-bracket secondary
structure, the policy network learns to fill nucleotides in any order
(not only left-to-right) using a Q-learning objective, with rewards
computed from a RibonanzaNet-SS oracle.

This repository contains the training code, evaluation scripts for the
Eterna100 + OpenKnot 7 benchmarks, paper figures for the 100-mer and
240-mer experiments, and the ICML 2026 paper sources.

> **Status (2026-05-26):** active research code. The training and
> evaluation paths used in the paper run end-to-end on the Brev / Azure
> DGX setup. Some evaluation scripts still contain machine-specific
> paths — see *Hardcoded paths* below.

## Repository structure

```
.
├── run.py                       # Main training entry (Accelerate, bf16)
├── Encoder_Decoder.py           # Transformer policy network (RPE arch)
├── Functions.py                 # Reward / scheduler / play-phase utils
├── Dataset.py                   # Structure-to-sequence dataset loader
├── Env.py                       # RibonanzaNet-SS oracle wrapper
├── Network_test10.py            # RibonanzaNet model definition
├── dropout.py                   # Shared-mask dropout (OpenFold-style)
├── default_config.yaml          # Default training hyper-parameters
├── config_brev_8gpu.yaml        # Brev / Azure DGX (8x A100) override
├── test10_configs/              # RibonanzaNet configs (pairwise.yaml used)
├── launch_brev.sh               # Portable 4-GPU launcher
├── make_arnie_dummy.py          # Writes a placeholder arnie config
│
├── evaluation/                  # Eterna100 + OpenKnot 7 eval pipeline
│   ├── run_eterna100_benchmark.sh
│   ├── run_eterna100_remaining_parallel.sh
│   ├── watch_and_parallelize_eterna100.sh
│   ├── prepare_eterna100_targets.py
│   ├── write_eterna100_report.py
│   ├── run_eval.py              # bpRNA-1m eval
│   ├── run_ok7_eval.py          # OpenKnot 7 Round 3
│   ├── run_bidir_rescue.py
│   ├── motif_extraction.py
│   ├── compute_metrics.py
│   └── eterna100_logs/          # 240-mer Eterna100 run logs (text)
│
├── paper_figs/                  # Figure generation for the paper
│   ├── make_paper_figures.py    # Current figure builder
│   ├── analyze_results_240mer.py
│   ├── build_master_tables.py
│   └── *.png/svg/pdf            # Pre-rendered figures
│
├── data/
│   ├── eterna100/               # Eterna100 benchmark targets
│   └── eterna_training_dump/    # Reference Eterna puzzle dump
│                                # (large CSV not tracked; see data/README.md)
│
├── icml2026/                    # ICML 2026 paper sources (.tex + figures)
├── paper/                       # Writing prompts and idea notes
├── lab_notebook/                # Dated lab notes (kept for provenance)
├── ideas/                       # Reward design / contrastive notes
├── archive/                     # Superseded scripts (see docs/history/)
├── docs/                        # Long-form docs and history
├── eterna100_results.md         # Auto-generated Eterna100 benchmark report
└── tests/                       # CPU smoke tests (no GPU required)
```

## Installation

Tested with Python 3.10–3.11 and CUDA 12.x on Linux (Brev A100, Azure
DGX, plus assorted local boxes).

```bash
git clone git@github.com:antonia-panescu/Struct2SeQ_bidir.git
cd Struct2SeQ_bidir

pip install -r requirements.txt
pip install git+https://github.com/DasLab/arnie.git

# arnie needs a config pointing at LinearPartition + a TMP directory.
# The training launcher writes one for you; for one-off scripts:
python make_arnie_dummy.py        # writes ./arnie_file.txt
export ARNIEFILE="$(pwd)/arnie_file.txt"
```

## Pretrained weights

None of the model checkpoints or oracle weights are checked into git
(each is 100–120 MB). Download them from Google Drive and place them
as below.

### RibonanzaNet oracle (required for training and eval)

`Env.py` loads two RibonanzaNet checkpoints from a **sibling**
`weights/` directory at `../weights/` relative to this repo:

```
<parent>/
├── struct2seq_bidir_rl/        ← this repo
└── weights/
    ├── RibonanzaNet-SS.pt      ← used by finetuned_RibonanzaNet (oracle reward)
    └── RibonanzaNet.pt         ← used for reactivity scoring
```

The original `RibonanzaNet*.pt` files are published by the DasLab
RibonanzaNet release. If they live elsewhere on your machine, symlink:

```bash
mkdir -p ../weights
ln -s /abs/path/to/RibonanzaNet-SS.pt ../weights/RibonanzaNet-SS.pt
ln -s /abs/path/to/RibonanzaNet.pt    ../weights/RibonanzaNet.pt
```

### Struct2SeQ-bidir model checkpoints

Place these at the **repo root** (next to `run.py`):

| File                       | Size   | Purpose                                          |
|----------------------------|--------|--------------------------------------------------|
| `Struct2SeQ.pt`            | 119 MB | Original L→R baseline (paper comparison + init)  |
| `best_policy_network.pt`   | 112 MB | Best test-reward checkpoint from our RL training |
| `policy_network_{N}.pt`    | 112 MB | Per-episode checkpoints (optional, for resume)   |

Google Drive (TODO — paste shared link here once uploaded):

```
RibonanzaNet weights:        <Google Drive link>
Struct2SeQ.pt:               <Google Drive link>
best_policy_network.pt:      <Google Drive link>
policy_network_{N}.pt set:   <Google Drive link>
```

Quick fetch with `gdown` once links are available:

```bash
pip install gdown
gdown --id <FILE_ID> -O ./Struct2SeQ.pt
gdown --id <FILE_ID> -O ./best_policy_network.pt
mkdir -p ../weights
gdown --id <FILE_ID> -O ../weights/RibonanzaNet-SS.pt
gdown --id <FILE_ID> -O ../weights/RibonanzaNet.pt
```

## Data

Two data resources are referenced:

- `top2M.csv` (~515 MB): top-2 M 240-mer windows from Shujun's genome
  scan with OK-filtering. **This is the training set used to produce
  the paper checkpoints.** It is not committed (see `.gitignore`).
  Copy it to the project root or `$HOME`; `launch_brev.sh` looks in
  both.
- `data/eterna100/`: Eterna100 puzzle targets (committed, small).
- `data/eterna_training_dump/eterna_puzzles_20250227.csv`: 44 MB
  reference dump of Eterna puzzles (not committed). See
  `data/README.md` for provenance and how to regenerate.

The 100-mer and 240-mer master result tables live in
`paper_figs/MASTER_100mer.csv` / `MASTER_240mer.csv`.

## Training

```bash
# 4-GPU portable launcher (Brev / Azure DGX; auto-detects top2M.csv).
bash launch_brev.sh --from-scratch
# or, fine-tune the published Struct2SeQ checkpoint:
bash launch_brev.sh --from-pretrained ./Struct2SeQ.pt

# Manual single-machine invocation:
accelerate launch --mixed_precision bf16 --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file ./top2M.csv \
    --order-agnostic
```

Useful flags (full list in `run.py`):

| Flag                       | Meaning                                       |
|----------------------------|-----------------------------------------------|
| `--order-agnostic`         | Train the bidirectional / random-order policy |
| `--checkpoint PATH`        | Initialise from existing weights              |
| `--skip-play-episode0`     | Skip exploratory play in episode 0            |
| `--save-every-steps N`     | Mid-epoch checkpointing cadence               |
| `--config PATH`            | YAML hyper-parameter override                 |

Outputs land in the project root:

- `policy_network.pt` / `policy_network_{episode}.pt` — checkpoints
- `best_policy_network.pt` — best test reward so far
- `stats/episode*.csv` — per-episode reward curves
- `rewards_log.csv` — train/test reward summary
- `logs/` and `wandb/` — Accelerate / W&B logs

## Evaluation

The paper benchmarks live under `evaluation/`. The Eterna100 driver:

```bash
cd evaluation
bash run_eterna100_benchmark.sh ./eterna100_logs/run.log
# or for partial / parallel reruns:
bash run_eterna100_remaining_parallel.sh
```

OpenKnot 7 (240-mer in-painting + best-of-N) lives in
`evaluation/run_ok7_eval.py` and the `launch_ok7_*.sh` wrappers.

### Hardcoded paths

A number of evaluation scripts (`evaluation/run_eterna100_*.sh`,
`evaluation/launch_ok7_*.sh`, `evaluation/run_ok7_eval.py`,
`paper_figs/analyze_results_*.py`, `evaluation/motif_extraction.py`)
still contain absolute paths to this machine — `/home/nvidia/...`,
`/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/`,
`/home/nvidia/miniconda3/envs/struct2seq/`. Before running them on a
new machine, set the relevant variables at the top of each script or
point the equivalent env vars (`OPENKNOT_DATA`, `OPENKNOT_TARGETS_R3`,
`OPENKNOT_TARGETS_R4`, `OPENKNOTSCORE_SRC`, `STRUCT2SEQ_ROOT`).

## Tests

Minimal CPU smoke tests (no GPU, no data required):

```bash
pip install pytest
pytest tests/ -q
```

## Known limitations / TODOs

- Evaluation scripts still need fully env-driven path resolution.
- No formal package layout (modules live at the top level).
  Refactoring into a `struct2seq/` package would be useful but is out
  of scope for this cleanup pass.
- `arnie_file.txt` is generated per machine and is not committed.
- The `Struct2SeQ.pt` baseline checkpoint, our trained
  `*_policy_network.pt` files, and `RibonanzaNet*.pt` weights are not
  in this repository — see *Pretrained weights*.

## Reproducing the paper

The 100-mer / 240-mer numbers in the ICML 2026 submission come from:

- Headline run: `MASTER_240mer.csv` row `bidir + best-of-N (K=1000)`
- Apples-to-apples: `PAPER_TABLE_240mer.csv`
- Sample efficiency: `paper_figs/best_of_n_240mer.csv`

See `lab_notebook/` for the run-by-run trail and `icml2026/paper.tex`
for the final manuscript.

## Citing / contact

This repository is research code under active development. Until a
release is tagged, cite the ICML 2026 GenBio submission and contact
the authors before reusing in downstream work.
