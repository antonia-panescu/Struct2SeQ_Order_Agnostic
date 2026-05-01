#!/usr/bin/env bash
# Launch order-agnostic Struct2SeQ RL training on Brev (8x A100).
#
# Prerequisites:
#   1. SSH into Brev: brev shell a100he
#   2. knitnet conda env activated
#   3. top2M.csv available (scp from RDAV if needed):
#        scp rdav:/gpfs/radev/home/afp38/scratch/Struct2SeQ_training_data/top2M.csv ~/
#   4. This directory synced to Brev:
#        scp -r scripts/struct2seq_bidir_rl/ brev:~/struct2seq_bidir_rl/
#
# Usage (on Brev):
#   cd ~/struct2seq_bidir_rl
#   bash launch_brev.sh [--from-scratch | --from-pretrained /path/to/Struct2SeQ.pt]
#
# Training modes:
#   --from-scratch:    Train order-agnostic DQN from random init (full 10 episodes)
#   --from-pretrained: Load pretrained L->R weights, fine-tune with random order

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Default: from scratch
CHECKPOINT_ARG=""
MODE="from-scratch"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-pretrained)
            CHECKPOINT_ARG="--checkpoint $2"
            MODE="from-pretrained"
            shift 2
            ;;
        --from-scratch)
            MODE="from-scratch"
            shift
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Check data — look in project dir first, then home
if [[ -f "${SCRIPT_DIR}/top2M.csv" ]]; then
    DATA_FILE="${SCRIPT_DIR}/top2M.csv"
elif [[ -f "${HOME}/top2M.csv" ]]; then
    DATA_FILE="${HOME}/top2M.csv"
else
    echo "ERROR: top2M.csv not found in ${SCRIPT_DIR} or ${HOME}."
    echo "  scp rdav:/gpfs/radev/home/afp38/scratch/Struct2SeQ_training_data/top2M.csv ~/"
    exit 1
fi

# Create arnie dummy config (required by Functions.py)
cat > arnie_file.txt << 'EOF'
linearpartition: .
TMP: /tmp
EOF
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

# Restrict to 4 GPUs (0,1,2,5) — shared lab server, leave others for labmates
export CUDA_VISIBLE_DEVICES=0,1,2,5

# Extend NCCL watchdog timeout to 2 hours (default 600s is too short when one
# rank is slower than others during test play, causing spurious SIGABRT crashes)
export NCCL_TIMEOUT=7200

echo "=== Struct2SeQ Order-Agnostic RL Training ==="
echo "Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Mode:   ${MODE}"
echo "Data:   ${DATA_FILE}"
echo "GPUs:   $(nvidia-smi -L 2>/dev/null | wc -l) available, using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

mkdir -p tmp stats

# Launch with HuggingFace Accelerate across 4 GPUs
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file "${DATA_FILE}" \
    --order-agnostic \
    ${CHECKPOINT_ARG} \
    ${EXTRA_ARGS}

echo ""
echo "=== Training Complete ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs:"
echo "  best_policy_network.pt  (best test reward)"
echo "  final_policy_network.pt (final episode)"
echo "  stats/                  (per-episode rewards)"
echo "  rewards_log.csv         (train/test reward curve)"
