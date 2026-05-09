#!/usr/bin/env bash
# Framing B (motif-redesign) with FIXED-FIRST decoding order.
#
# Same as launch_ok7_motif_redesign.sh, but uses --decode-order fixed_first
# so the decoder consumes the entire fixed scaffold (~85% of positions)
# in random order BEFORE designing the motif positions. This puts the
# whole WT scaffold in the decoder's KV cache before any free position
# is generated — closer to the "condition on the full scaffold, generate
# the motif" mental model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_SET:-0,1,3,4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
K_SAMPLES="${K_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_ROOT="${OUT_ROOT:-results/ok7_eval}"

if [ ! -f arnie_file.txt ]; then
    cat > arnie_file.txt <<EOF
linearpartition: .
TMP: /tmp
EOF
fi
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

mkdir -p "${OUT_ROOT}"

tag="bidir_motifs_redesign_fixedfirst"
out="${OUT_ROOT}/${tag}"
mkdir -p "${out}"

echo "=========================================================="
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Launching: ${tag}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  K_SAMPLES=${K_SAMPLES}  BATCH_SIZE=${BATCH_SIZE}"
echo "  out: ${out}"
echo "=========================================================="

/home/nvidia/miniconda3/envs/struct2seq/bin/accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    evaluation/run_ok7_eval.py \
    --k-samples "${K_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --out-dir "${out}" \
    --model bidir \
    --inference-mode inpaint \
    --motif-mode structural_redesign \
    --decode-order fixed_first

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Done: ${tag}"
