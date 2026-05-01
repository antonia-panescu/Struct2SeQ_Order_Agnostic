#!/usr/bin/env bash
# Launch script for Brev instance jvkq0dvug (8x A100 Azure DGX).
# Hardcodes paths for this session to avoid conda env ARNIEFILE override.

set -euo pipefail

SCRIPT_DIR="/home/nvidia/haiwen/antonia/struct2seq_bidir_rl"
cd "${SCRIPT_DIR}"

# Override ARNIEFILE unconditionally — conda env sets it to ../arnie_file.txt which is wrong
cat > "${SCRIPT_DIR}/arnie_file.txt" << 'EOF'
linearpartition: .
TMP: /tmp
EOF
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

DATA_FILE="${SCRIPT_DIR}/top2M.csv"

# Default: from scratch
CHECKPOINT_ARG=""
MODE="from-scratch"
SKIP_PLAY_ARG=""
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5000}"

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
        --skip-play-episode0)
            SKIP_PLAY_ARG="--skip-play-episode0"
            shift
            ;;
        --save-every-steps)
            SAVE_EVERY_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: ${DATA_FILE} not found."
    exit 1
fi

echo "=== Struct2SeQ Order-Agnostic RL Training ==="
echo "Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Mode:   ${MODE}"
echo "Data:   ${DATA_FILE}"
echo "ARNIEFILE: ${ARNIEFILE}"
echo "GPUs:   $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

mkdir -p tmp stats

# --- Distributed launch controls (important on shared Brev VMs) ---
# Default: 8 processes on all visible GPUs. For 4 GPUs only:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch_brev_jvkq0dvug.sh --from-pretrained /path/Struct2SeQ.pt
# If CUDA_VISIBLE_DEVICES lists N GPUs, set NUM_PROCESSES=N (Accelerate uses local rank 0..N-1).
NUM_PROCESSES="${NUM_PROCESSES:-8}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# Safer NCCL behavior when multiple jobs share the node (does not fix all hangs).
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "NUM_PROCESSES=${NUM_PROCESSES}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo ""

LOGFILE="${SCRIPT_DIR}/run_$(date '+%Y%m%d_%H%M%S').log"
echo "Logging stdout/stderr to: ${LOGFILE}"
echo "SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"
echo ""

# Force unbuffered Python output so tqdm + print lines reach the log file
# in real time (otherwise they block-buffer when piped through tee).
export PYTHONUNBUFFERED=1

# Write a tiny header to the log so you can grep it for run metadata.
{
    echo "=== run.log ==="
    echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "mode: ${MODE}"
    echo "checkpoint_arg: ${CHECKPOINT_ARG}"
    echo "skip_play_arg: ${SKIP_PLAY_ARG}"
    echo "save_every_steps: ${SAVE_EVERY_STEPS}"
    echo "num_processes: ${NUM_PROCESSES}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "data: ${DATA_FILE}"
    echo "arniefile: ${ARNIEFILE}"
    echo "git: $(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo 'n/a')"
    echo "==============="
} > "${LOGFILE}"

ARNIEFILE="${ARNIEFILE}" PYTHONUNBUFFERED=1 accelerate launch \
    --mixed_precision bf16 \
    --num_processes "${NUM_PROCESSES}" \
    run.py \
    --config config_brev_8gpu.yaml \
    --target_structure_file "${DATA_FILE}" \
    --order-agnostic \
    --save-every-steps "${SAVE_EVERY_STEPS}" \
    ${CHECKPOINT_ARG} \
    ${SKIP_PLAY_ARG} 2>&1 | tee -a "${LOGFILE}"

echo ""
echo "=== Training Complete ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs:"
echo "  best_policy_network.pt  (best test reward)"
echo "  final_policy_network.pt (final episode)"
echo "  stats/                  (per-episode rewards)"
echo "  rewards_log.csv         (train/test reward curve)"
