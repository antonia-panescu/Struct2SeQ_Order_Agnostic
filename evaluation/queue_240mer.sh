#!/usr/bin/env bash
# Queue all 11 OK7b 240mer configs, running 2 in parallel per round.
#
# Round 1 (already running externally): bidir_random + orig_struct2seq_l2r
# Round 2: bidir_motifs_structural + orig_struct2seq_motifs_structural
# Round 3: bidir_identity + bidir_paired_first
# Round 4: bidir_inpaint_K00 + bidir_inpaint_K25
# Round 5: bidir_inpaint_K50 + bidir_inpaint_K75
# Round 6: bidir_inpaint_K95 (single — runs on GPUs 0,1,2,3 alone)
#
# Logs into evaluation/_240mer_<tag>_run.log
# Outputs into results/ok7b_eval/<tag>_240mer/
#
# Run: nohup bash evaluation/queue_240mer.sh > evaluation/_240mer_queue.log 2>&1 < /dev/null & disown

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

ROUND4=/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round4_targets.csv
RESULTS_BASE=results/ok7b_eval
mkdir -p "${RESULTS_BASE}"

ACCELERATE=/home/nvidia/miniconda3/envs/struct2seq/bin/accelerate
KSAMPLES="${KSAMPLES:-1000}"
BSZ="${BSZ:-16}"

if [ ! -f arnie_file.txt ]; then
    cat > arnie_file.txt <<EOF
linearpartition: .
TMP: /tmp
EOF
fi
export ARNIEFILE="${SCRIPT_DIR}/arnie_file.txt"

# helper: wait for a list of summary.csv paths to all exist
wait_for_summaries() {
    for f in "$@"; do
        until [ -f "$f" ]; do sleep 20; done
    done
    echo "$(date -u '+%H:%M:%S UTC'): summaries present: $@"
}

# helper: launch a bidir-side eval (uses run_ok7_eval.py from this dir)
launch_bidir() {
    local gpus="$1" port="$2" tag="$3"; shift 3
    local out="${RESULTS_BASE}/${tag}_240mer"
    mkdir -p "${out}"
    echo "$(date -u '+%H:%M:%S UTC'): launching bidir [${tag}] on GPUs ${gpus}"
    CUDA_VISIBLE_DEVICES="${gpus}" NCCL_TIMEOUT=7200 \
        nohup "${ACCELERATE}" launch \
        --mixed_precision bf16 --num_processes 4 --main_process_port "${port}" \
        evaluation/run_ok7_eval.py \
        --targets-csv "${ROUND4}" \
        --k-samples "${KSAMPLES}" --batch-size "${BSZ}" \
        --out-dir "${out}" "$@" \
        > "evaluation/_240mer_${tag}_run.log" 2>&1 < /dev/null &
    disown
}

# helper: launch an orig-S2S-side eval (uses Struct2SeQ_training/run_ok7_orig.py)
launch_orig() {
    local gpus="$1" port="$2" tag="$3"; shift 3
    local out="${SCRIPT_DIR}/${RESULTS_BASE}/${tag}_240mer"
    mkdir -p "${out}"
    echo "$(date -u '+%H:%M:%S UTC'): launching orig [${tag}] on GPUs ${gpus}"
    cd /home/nvidia/haiwen/antonia/Struct2SeQ_training
    CUDA_VISIBLE_DEVICES="${gpus}" NCCL_TIMEOUT=7200 \
        nohup "${ACCELERATE}" launch \
        --mixed_precision bf16 --num_processes 4 --main_process_port "${port}" \
        run_ok7_orig.py \
        --targets-csv "${ROUND4}" \
        --k-samples "${KSAMPLES}" --batch-size "${BSZ}" \
        --out-dir "${out}" "$@" \
        > "${SCRIPT_DIR}/evaluation/_240mer_${tag}_run.log" 2>&1 < /dev/null &
    disown
    cd "${SCRIPT_DIR}"
}

# Round 1 already in flight (PIDs 3279091 + 3276656). Just wait.
echo "=== Round 1: bidir_random + orig_struct2seq_l2r (already in flight) ==="
wait_for_summaries \
    "${RESULTS_BASE}/bidir_random_240mer/summary.csv" \
    "${RESULTS_BASE}/orig_struct2seq_l2r_240mer/summary.csv"

echo "=== Round 2: motif modes ==="
launch_bidir "0,1,2,3" 29500 "bidir_motifs_structural" \
    --model bidir --inference-mode inpaint --motif-mode structural
launch_orig "4,5,6,7" 29501 "orig_struct2seq_motifs_structural" \
    --motif-mode structural
wait_for_summaries \
    "${RESULTS_BASE}/bidir_motifs_structural_240mer/summary.csv" \
    "${RESULTS_BASE}/orig_struct2seq_motifs_structural_240mer/summary.csv"

echo "=== Round 3: bidir_identity + bidir_paired_first ==="
launch_bidir "0,1,2,3" 29500 "bidir_identity" \
    --model bidir --inference-mode identity
launch_bidir "4,5,6,7" 29501 "bidir_paired_first" \
    --model bidir --inference-mode paired_first
wait_for_summaries \
    "${RESULTS_BASE}/bidir_identity_240mer/summary.csv" \
    "${RESULTS_BASE}/bidir_paired_first_240mer/summary.csv"

echo "=== Round 4: bidir_inpaint K=0.00 + K=0.25 ==="
launch_bidir "0,1,2,3" 29500 "bidir_inpaint_K00" \
    --model bidir --inference-mode inpaint --inpaint-k 0.00
launch_bidir "4,5,6,7" 29501 "bidir_inpaint_K25" \
    --model bidir --inference-mode inpaint --inpaint-k 0.25
wait_for_summaries \
    "${RESULTS_BASE}/bidir_inpaint_K00_240mer/summary.csv" \
    "${RESULTS_BASE}/bidir_inpaint_K25_240mer/summary.csv"

echo "=== Round 5: bidir_inpaint K=0.50 + K=0.75 ==="
launch_bidir "0,1,2,3" 29500 "bidir_inpaint_K50" \
    --model bidir --inference-mode inpaint --inpaint-k 0.50
launch_bidir "4,5,6,7" 29501 "bidir_inpaint_K75" \
    --model bidir --inference-mode inpaint --inpaint-k 0.75
wait_for_summaries \
    "${RESULTS_BASE}/bidir_inpaint_K50_240mer/summary.csv" \
    "${RESULTS_BASE}/bidir_inpaint_K75_240mer/summary.csv"

echo "=== Round 6: bidir_inpaint K=0.95 (alone) ==="
launch_bidir "0,1,2,3" 29500 "bidir_inpaint_K95" \
    --model bidir --inference-mode inpaint --inpaint-k 0.95
wait_for_summaries "${RESULTS_BASE}/bidir_inpaint_K95_240mer/summary.csv"

echo "$(date -u '+%H:%M:%S UTC'): all 11 configs complete."
