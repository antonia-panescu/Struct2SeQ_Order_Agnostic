# Struct2SeQ Order-Agnostic

Order-agnostic reinforcement learning for RNA inverse folding. This code extends
Struct2SeQ-style sequence design so the decoder can fill RNA positions in a
random order instead of only left to right. The model is trained with a
Q-learning objective and scored with a RibonanzaNet-SS structure oracle.

## Repository Contents

```text
.
├── run.py                  Training entry point
├── Encoder_Decoder.py      Encoder and order-aware decoder architecture
├── Functions.py            Decoding, sampling, and sequence utilities
├── Dataset.py              Dataset and dot-bracket tokenization utilities
├── Env.py                  RibonanzaNet-based reward environment
├── Network_test10.py       RibonanzaNet architecture used by the oracle
├── default_config.yaml     Small default training configuration
├── config_brev_8gpu.yaml   Paper-scale 240 nt configuration
├── data/eterna100/         Small Eterna100 benchmark target files
├── evaluation/             Public evaluation utilities
└── tests/                  Lightweight import/config/dropout tests
```

Large checkpoints, generated results, run logs, paper drafts, and local research
notes are intentionally excluded from Git. They can remain in your local working
tree without being pushed.

## Installation

The code has been tested on Linux with Python 3.10-3.11, PyTorch 2.x, and CUDA.

```bash
git clone git@github.com:antonia-panescu/Struct2SeQ_Order_Agnostic.git
cd Struct2SeQ_Order_Agnostic

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/DasLab/arnie.git
```

`arnie` expects an `ARNIEFILE`. For the RibonanzaNet-only training/evaluation
paths, a minimal local config is sufficient:

```bash
python make_arnie_dummy.py
export ARNIEFILE="$(pwd)/arnie_file.txt"
```

## Required Weights

The repository does not track model checkpoints or oracle weights. Place the
RibonanzaNet weights in a directory and point `RIBONANZA_WEIGHTS_DIR` at it:

```text
weights/
├── RibonanzaNet-SS.pt
└── RibonanzaNet.pt
```

```bash
export RIBONANZA_WEIGHTS_DIR=/path/to/weights
```

If `RIBONANZA_WEIGHTS_DIR` is not set, `Env.py` looks for these files in
`../weights` relative to the repository root.

For checkpointed Struct2SeQ models, place files such as `Struct2SeQ.pt` or
`best_policy_network.pt` at the repository root, or pass their paths explicitly
with `--checkpoint`.

## Data

Training expects a CSV with a `structure` column containing dot-bracket targets.
The paper-scale configuration used 240 nt structures. Large training CSVs are
not tracked by Git.

The committed Eterna100 files under `data/eterna100/` are small benchmark target
files used by the evaluation scripts.

## Training

Single-node multi-GPU training:

```bash
accelerate launch --mixed_precision bf16 --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file /path/to/train_targets.csv \
    --order-agnostic
```

Fine-tune from an existing checkpoint:

```bash
accelerate launch --mixed_precision bf16 --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file /path/to/train_targets.csv \
    --order-agnostic \
    --checkpoint /path/to/Struct2SeQ.pt
```

Useful flags:

| Flag | Description |
| --- | --- |
| `--order-agnostic` | Train with random RNA-position decoding order. |
| `--checkpoint PATH` | Initialize from an existing policy checkpoint. |
| `--config PATH` | Load training hyperparameters from YAML. |
| `--start-episode N` | Resume training from episode `N`. |
| `--skip-train` | Skip training for the starting episode and run test play. |
| `--wandb` | Enable Weights & Biases logging. |
| `--wandb-project NAME` | Set the W&B project name. |
| `--wandb-entity NAME` | Set the W&B entity or team. |

Install `wandb` separately before using the W&B flags.

Training writes checkpoints and metrics to local files such as:

```text
policy_network.pt
policy_network_{episode}.pt
best_policy_network.pt
final_policy_network.pt
logs/
stats/
rewards_log.csv
runtime.txt
```

These outputs are ignored by Git.

## Evaluation

Generic checkpoint evaluation on a CSV of target structures:

```bash
python evaluation/run_eval.py \
    --checkpoint best_policy_network.pt \
    --config config_brev_8gpu.yaml \
    --test-csv data/eterna100/eterna100_targets_v2.csv \
    --experiment order_agnostic_eval \
    --label best \
    --decoding-order permuted
```

`evaluation/run_eval.py` supports `--decoding-order permuted` for random
per-sample decoding orders and `--decoding-order l2r` for a left-to-right
ablation.

OpenKnot-style evaluation is available through `evaluation/run_ok7_eval.py`.
That path also requires the OpenKnotScorePipeline source and target CSVs:

```bash
export OPENKNOTSCORE_SRC=/path/to/OpenKnotScorePipeline/src

accelerate launch --mixed_precision bf16 --num_processes 4 \
    evaluation/run_ok7_eval.py \
    --targets-csv /path/to/openknot_targets.csv \
    --out-dir results/ok7_eval/bidir_random \
    --model bidir \
    --inference-mode random \
    --k-samples 1000
```

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

Some import tests are skipped automatically when optional external packages such
as `arnie` are not installed.

## Citation

If you use this repository, please cite the accompanying ICML 2026 paper. The
full citation will be added here once the camera-ready metadata is finalized.
