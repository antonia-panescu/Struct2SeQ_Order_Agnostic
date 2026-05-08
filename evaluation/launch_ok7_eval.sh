#!/usr/bin/env bash
# Launch OpenKnot 7 (100mer) eval suite on 4 free GPUs while training holds
# 2,5,6,7. Runs 8 configurations sequentially:
#   1. bidir + identity (L→R from bidir checkpoint)
#   2. bidir + random
#   3. bidir + paired_first
#   4–8. bidir + inpaint at K∈{0.00, 0.25, 0.50, 0.75, 0.95}
#
# Each config = K samples per puzzle × 20 puzzles. Per-rank checkpointing
# means re-running the same command after a crash will fast-skip completed
# batches.
#
# Usage:
#   bash evaluation/launch_ok7_eval.sh                      # full sweep, K=1000
#   K_SAMPLES=100 bash evaluation/launch_ok7_eval.sh        # smaller sweep
#   GPU_SET=0,1,3,4 bash evaluation/launch_ok7_eval.sh      # override GPU set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

# Default the four GPUs that the training run does NOT use (training holds 2,5,6,7).
export CUDA_VISIBLE_DEVICES="${GPU_SET:-0,1,3,4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
K_SAMPLES="${K_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_ROOT="${OUT_ROOT:-results/ok7_eval}"

# Arnie dummy config (Env.py needs ARNIEFILE set).
if [ ! -f arnie_file.txt ]; then
    cat > arnie_file.txt <<EOF
linearpartition: .
TMP: /tmp
EOF
fi
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

mkdir -p "${OUT_ROOT}"

run_config() {
    local tag="$1"; shift
    local out="${OUT_ROOT}/${tag}"
    mkdir -p "${out}"
    echo ""
    echo "=========================================================="
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Launching: ${tag}"
    echo "  out: ${out}"
    echo "=========================================================="
    /home/nvidia/miniconda3/envs/struct2seq/bin/accelerate launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        evaluation/run_ok7_eval.py \
        --k-samples "${K_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --out-dir "${out}" \
        "$@"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Done: ${tag}"
}

# 1. bidir + identity (L→R from bidir checkpoint)
run_config "bidir_identity" --model bidir --inference-mode identity

# 2. bidir + random
run_config "bidir_random" --model bidir --inference-mode random

# 3. bidir + paired_first (Shujun §4 hypothesis: decode paired positions together)
run_config "bidir_paired_first" --model bidir --inference-mode paired_first

# 4–8. in-painting K-sweep
# Pairs: (K_value_for_script, tag_suffix_int)
for entry in "0.00 00" "0.25 25" "0.50 50" "0.75 75" "0.95 95"; do
    K=$(echo "$entry" | cut -d' ' -f1)
    KK=$(echo "$entry" | cut -d' ' -f2)
    tag="bidir_inpaint_K${KK}"
    run_config "${tag}" --model bidir --inference-mode inpaint --inpaint-k "${K}"
done

# 9. principled structural-motif in-painting (each sample picks one motif
# at random from the puzzle's hairpin / internal_loop / multi_loop /
# pseudoknot_stem inventory; positions in that motif fixed to WT).
run_config "bidir_motifs_structural" \
    --model bidir --inference-mode inpaint --motif-mode structural

echo ""
echo "All configurations complete."
echo "Per-config summaries: ${OUT_ROOT}/<config>/summary.csv"
