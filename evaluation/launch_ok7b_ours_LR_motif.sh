#!/usr/bin/env bash
# Run OUR checkpoint forced into L→R + teacher-forced motif preservation.
# Four variants: argmax, eps p=0.05, eps p=0.10, qsoftmax. Each at
# K=333 (sampling) or K=1000 (argmax).
#
# Output: results/ok7b_eval/bidir_motifs_structural_LR_<mode>_240mer/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_SET:-0,1,3,4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
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
    local mode="$1" k="$2" p="$3" tag_suffix="$4"
    local tag="bidir_motifs_structural_LR_${tag_suffix}_240mer"
    local out="results/ok7b_eval/${tag}"
    mkdir -p "${out}"
    echo "[$(date -u '+%H:%M:%S UTC')] Launching ${tag}"
    "${ACCEL}" launch --mixed_precision bf16 --num_processes 4 \
        --main_process_port 29580 \
        evaluation/run_ok7_eval.py \
        --targets-csv "${TARGETS}" \
        --k-samples "${k}" --batch-size "${BATCH_SIZE}" \
        --out-dir "${out}" \
        --model bidir \
        --inference-mode inpaint \
        --motif-mode structural \
        --decode-order identity \
        --sampling-mode "${mode}" \
        --sampling-p "${p}"
    echo "[$(date -u '+%H:%M:%S UTC')] Done ${tag}"
}

# Argmax baseline at full budget (parallel to bidir_motifs_structural_240mer)
run_variant "argmax"   "1000" "0.0"  "argmax"
# 3-strategy at K=333 each (parallel to AR's faithful test_240.py protocol)
run_variant "epsilon"  "333"  "0.05" "epsilon05"
run_variant "epsilon"  "333"  "0.10" "epsilon10"
run_variant "qsoftmax" "333"  "1.0"  "qsoftmax"

echo "[$(date -u '+%H:%M:%S UTC')] All 4 variants done."
