#!/usr/bin/env bash
# Run OUR model under Shujun's autoregressive L→R + sampling protocol
# on 240mer. Replicates test_240.py exactly but with our checkpoint
# (random-perm-trained) forced into L→R inference mode (--inference-mode
# identity) plus the same per-step sampling rules.
#
# Output:  results/ok7b_eval/bidir_identity_<strategy>_240mer/
#   for strategy in {epsilon05, epsilon, qsoftmax}.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_SET:-0,1,3,4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
K_PER_STRAT="${K_PER_STRAT:-333}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCEL=/home/nvidia/miniconda3/envs/struct2seq/bin/accelerate
TARGETS=/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round4_targets.csv

if [ ! -f arnie_file.txt ]; then
    cat > arnie_file.txt <<EOF
linearpartition: .
TMP: /tmp
EOF
fi
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

run_variant() {
    local mode="$1"; shift
    local p_eps="$1"; shift
    local tag="bidir_identity_${mode}_240mer"
    if [ "${mode}" = "epsilon" ] && [ "${p_eps}" = "0.05" ]; then
        tag="bidir_identity_epsilon05_240mer"
    fi
    local out="results/ok7b_eval/${tag}"
    mkdir -p "${out}"
    echo ""
    echo "[$(date -u '+%H:%M:%S UTC')] Launching ${tag} (sampling-mode=${mode}, p=${p_eps})"
    "${ACCEL}" launch --mixed_precision bf16 --num_processes 4 \
        --main_process_port 29560 \
        evaluation/run_ok7_eval.py \
        --targets-csv "${TARGETS}" \
        --k-samples "${K_PER_STRAT}" \
        --batch-size "${BATCH_SIZE}" \
        --out-dir "${out}" \
        --model bidir \
        --inference-mode identity \
        --sampling-mode "${mode}" \
        --sampling-p "${p_eps}"
    echo "[$(date -u '+%H:%M:%S UTC')] Done ${tag}"
}

# Faithful test_240.py: eps0.05 + eps0.10 + qsoftmax (no topk per Shujun's commented-out line)
run_variant "epsilon"  "0.05"
run_variant "epsilon"  "0.10"
run_variant "qsoftmax" "1.0"

echo "[$(date -u '+%H:%M:%S UTC')] All 3 sampling variants done."
