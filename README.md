# Struct2SeQ Order-Agnostic

Order-agnostic reinforcement learning for RNA inverse folding. This code extends
Struct2SeQ-style sequence design so the decoder can fill RNA positions in a
random order instead of only left to right. The model is trained with a
Q-learning objective and scored with a RibonanzaNet-SS structure oracle.

## Features

- Order-agnostic training with `run.py --order-agnostic`, which samples random
  RNA-position decoding orders during Q-learning.
- A decoder that uses relative-position attention biases in RNA-position space,
  rather than step-order-dependent decoder positional layers.
- Public generation with random-order or left-to-right decoding, Struct2SeQ-style
  wild-type biasing, and rescue mutation search.
- Evaluation entry points for left-to-right ablations, random-order decoding,
  paired-first decoding, and inpainting through `evaluation/run_eval.py` and
  `evaluation/run_ok7_eval.py`.

## Repository Contents

```text
.
├── generate.py             Public sequence generation entry point
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

## Installation and Run Instructions

1. Clone the repository.

```bash
git clone https://github.com/antonia-panescu/Struct2SeQ_Order_Agnostic
cd Struct2SeQ_Order_Agnostic
```

2. Create and activate a conda environment.

```bash
conda create -n struct2seq-order-agnostic python=3.11
conda activate struct2seq-order-agnostic
conda install pip
```

3. Install required Python dependencies.

The recommended path is to install from `requirements.txt`:

```bash
pip install -r requirements.txt
pip install git+https://github.com/DasLab/arnie.git
```

This mirrors the original Struct2SeQ recipe, with additional packages for
multi-GPU training, order-agnostic evaluation, and the public test suite.

4. Download model weights.

Place the policy checkpoint and RibonanzaNet oracle weights under the repository
`weights/` directory before running sequence generation or evaluation. If using
the Kaggle command-line client, install it inside the active conda environment
first:

```bash
pip install kaggle
mkdir -p weights
kaggle datasets download -d shujun717/ribonanzanet-weights -p weights
unzip weights/ribonanzanet-weights.zip -d weights
```

The Kaggle client requires local Kaggle credentials. You can also download the
zip manually from Kaggle and place the files yourself.

For the RibonanzaNet oracle, `Env.py` looks for the following files:

```text
weights/
├── RibonanzaNet-SS.pt
└── RibonanzaNet.pt
```

These files must be in `weights/` relative to the repository root. If they are
missing, the code raises an error with the expected path.

For the order-agnostic policy, place the checkpoint at:

```text
weights/order_agnostic_policy.pt
```

or pass a different path explicitly with `--weights_path`.

5. Initialize a dummy Arnie configuration.

Struct2SeQ relies on Arnie for RNA folding backends. For the RibonanzaNet-only
training/evaluation paths, a minimal local config is sufficient:

```bash
python make_arnie_dummy.py
export ARNIEFILE="$(pwd)/arnie_file.txt"
```

## Input Format

`generate.py` follows the original Struct2SeQ input format. The target CSV
should contain `Title`, `Dot-bracket`, and optionally `wild_type_sequence`.
The script also accepts `name`/`structure`/`sequence` aliases.

```csv
Title,Dot-bracket,wild_type_sequence
example_1,(((...))),GGGAAACCC
example_2,((..[[..))..]],GGAAUUCCGGAAUU
```

The committed Eterna100 files under `data/eterna100/` are small benchmark target
files that can be used to check the generation and benchmarking scripts.

## Run Sequence Generation

This command generates RNA sequences conditioned on the provided target
structures. It mirrors the original Struct2SeQ `generate.py` interface while
using order-agnostic decoding by default.

```bash
python generate.py \
    --target_df data/eterna100/eterna100_targets_v2.csv \
    --output_csv results.csv \
    --out_folder output_results \
    --gpu_id 0 \
    --n_structures 100 \
    --up_bias 0.0 \
    --weights_path weights/order_agnostic_policy.pt \
    --config config_brev_8gpu.yaml \
    --decoding_order permuted \
    --rescue_max_diff 4
```

`--decoding_order permuted` uses random RNA-position decoding orders, matching
the order-agnostic training objective. Use `--decoding_order l2r` for a
left-to-right ablation with the same checkpoint. `--up_bias` matches the
original Struct2SeQ wild-type bias when the target CSV includes a
`wild_type_sequence` column. The rescue mutation pass is enabled by default for
the best design when at most four pairing positions differ from the target; use
`--rescue_max_diff -1` to disable it.

The output CSV is written to `--out_folder/--output_csv` and contains:

| Column | Description |
| --- | --- |
| `sequence` | Designed RNA sequence. |
| `predicted_structure` | RibonanzaNet-SS predicted dot-bracket structure. |
| `source` | Target identifier from `Title` or `name`. |
| `shape_profile` | RibonanzaNet predicted SHAPE profile. |
| `target_structure` | Input target structure. |
| `jaccard` | Jaccard similarity between target and predicted base pairs. |
| `structure_match` | Whether predicted and target structures match exactly. |
| `strategy` | Sampling strategy used for this sequence. |
| `sample_idx` | Index within the sampling strategy. |
| `decoding_order` | `permuted` or `l2r`. |
| `hamming_distance` | Distance from `wild_type_sequence`, when available. |

`--n_structures` controls the number of samples per strategy per target. The
script runs the three Struct2SeQ-style sampling strategies: epsilon-greedy with
`p=0.05`, epsilon-greedy with `p=0.10`, and stochastic sampling. Rescue
candidates are written with `strategy=rescue`.

## Benchmarking

For benchmark-style runs with aggregate metrics and checkpoint snapshots, use
`evaluation/run_eval.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_eval.py \
    --checkpoint weights/order_agnostic_policy.pt \
    --config config_brev_8gpu.yaml \
    --test-csv data/eterna100/eterna100_targets_v2.csv \
    --experiment order_agnostic_eval \
    --label checkpoint_name \
    --out-dir results/eval \
    --decoding-order permuted
```

OpenKnot-style benchmarking is available through `evaluation/run_ok7_eval.py`.
This path also requires the OpenKnotScorePipeline source and target CSVs:

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

## Training and Fine-Tuning

To train an order-agnostic policy, run `run.py` from the repository root:

```bash
accelerate launch --mixed_precision bf16 --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file /path/to/train_targets.csv \
    --order-agnostic
```

To fine-tune from an existing Struct2SeQ checkpoint:

```bash
accelerate launch --mixed_precision bf16 --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file /path/to/train_targets.csv \
    --order-agnostic \
    --checkpoint weights/Struct2SeQ.pt
```

For the 5-episode order-agnostic fine-tune from the original Struct2SeQ
checkpoint on a Slurm cluster, the recommended launch path is a dependency
chain. The chain submits all chunks up front, so it keeps running if the login
session disconnects.

```bash
bash scripts/submit_order_agnostic_5ep_regular_chain.sh
```

For shorter 2-hour `ghx4-interactive` chunks, submit the interactive chain:

```bash
bash scripts/submit_order_agnostic_5ep_interactive_loop.sh
```

Both chain submitters call `sbatch` internally and pass the same `RUN_ID` to
each linked job with `--dependency=afterany:<previous_job_id>`. Each chunk
resumes from the run directory and exits immediately once
`final_policy_network.pt` exists. The worker script uses
`data/training_data_240/top2M.csv` by default and starts from
`weights/Struct2SeQ.pt`. It writes checkpoints and logs under:

```text
/projects/bdxs/apanescu/struct2seq_order_agnostic/weights/$RUN_ID
```

and generated play data under:

```text
/projects/bdxs/apanescu/struct2seq_order_agnostic/training_generated_data/$RUN_ID
```

Each `RUN_ID` begins with a timestamp, followed by a short run description. To
extend or resume a preempted run, submit another chain with the same run ID:

```bash
RUN_ID=YYYYMMDD_HHMMSS_oa5ep_from_s2s_original_4gpu \
    MAX_SUBMISSIONS=2 \
    bash scripts/submit_order_agnostic_5ep_regular_chain.sh
```

For a single Slurm job without chaining, submit the worker directly:

```bash
sbatch scripts/train_order_agnostic_5ep_from_s2s.sbatch
```

Useful training flags:

| Flag                   | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `--order-agnostic`     | Train with random RNA-position decoding order.            |
| `--checkpoint PATH`    | Initialize from an existing policy checkpoint.            |
| `--config PATH`        | Load training hyperparameters from YAML.                  |
| `--start-episode N`    | Resume training from episode `N`.                         |
| `--skip-train`         | Skip training for the starting episode and run test play. |
| `--play-save-interval N` | Save generated training data every `N` play batches.    |
| `--mid-epoch-save-interval N` | Save mid-epoch checkpoints every `N` optimizer steps. |
| `--wandb`              | Enable Weights & Biases logging.                          |
| `--wandb-project NAME` | Set the W&B project name.                                 |
| `--wandb-entity NAME`  | Set the W&B entity or team.                               |
| `--wandb-run-id ID`    | Resume or continue a specific W&B run.                    |
| `--wandb-resume MODE`  | Set W&B resume mode, such as `allow` or `must`.           |

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

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

Some import tests are skipped automatically when optional external packages such
as `arnie` are not installed.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{rna_inverse_order_agnostic_decoding_2026,
    title={Order-Agnostic Decoding for Sample-Efficient RNA Inverse Folding},
    author={Antonia Panescu and Shujun He and Yixuan He and Rex Ying},
    booktitle={ICML 2026 Workshop on Generative and Agentic AI for Biology},
    year={2026},
    url={https://openreview.net/forum?id=3tWscvIMJG}
}
```
