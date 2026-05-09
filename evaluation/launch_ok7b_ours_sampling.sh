#!/usr/bin/env bash
# Run OUR model on 240mer with the two sampling variants matching
# Shujun's strategies 1 and 2 (epsilon-greedy + Q-softmax) — for
# symmetric comparison with the orig Struct2SeQ 3-strategy AR baseline.
# Skip beam (joint order x token search is a research project).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_SET:-0,1,3,4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
K_PER_STRAT="${K_PER_STRAT:-333}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCEL=/home/nvidia/miniconda3/envs/struct2seq/bin/accelerate
TARGETS=/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round4_targets.csv
OUT_ROOT=results/ok7b_eval

if [ ! -f arnie_file.txt ]; then
    cat > arnie_file.txt <<EOF
linearpartition: .
TMP: /tmp
EOF
fi
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

mkdir -p "${OUT_ROOT}"

run_variant() {
    local mode="$1"
    local tag="bidir_random_${mode}_240mer"
    local out="${OUT_ROOT}/${tag}"
    mkdir -p "${out}"
    echo ""
    echo "[$(date -u '+%H:%M:%S UTC')] Launching ${tag}"
    "${ACCEL}" launch --mixed_precision bf16 --num_processes 4 \
        --main_process_port 29510 \
        evaluation/run_ok7_eval.py \
        --targets-csv "${TARGETS}" \
        --k-samples "${K_PER_STRAT}" \
        --batch-size "${BATCH_SIZE}" \
        --out-dir "${out}" \
        --model bidir \
        --inference-mode random \
        --sampling-mode "${mode}"
    echo "[$(date -u '+%H:%M:%S UTC')] Done ${tag}"
}

run_variant "qsoftmax"
run_variant "epsilon"

echo "[$(date -u '+%H:%M:%S UTC')] Both sampling variants done."
