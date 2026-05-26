#!/usr/bin/env bash
# Run remaining Eterna100 protocols efficiently after bidir_random_k1000_v2 completes.
# Uses only clean GPUs passed via GPUS_LIST (default: 5,6,7), one independent eval per GPU.
#
# Required env (same as run_eterna100_benchmark.sh):
#   STRUCT2SEQ_ROOT, ORIG_S2S_ROOT, ORIG_S2S_CHECKPOINT, CONDA_SH, ENV_PREFIX

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${STRUCT2SEQ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ORIG_ROOT="${ORIG_S2S_ROOT:?ORIG_S2S_ROOT must point at the original Struct2SeQ_training repo}"
ORIG_CKPT="${ORIG_S2S_CHECKPOINT:?ORIG_S2S_CHECKPOINT must point at Struct2SeQ.pt}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_PREFIX="${ENV_PREFIX:?ENV_PREFIX must point at the conda env containing accelerate}"

source "$CONDA_SH"

TARGETS=${ROOT}/data/eterna100/eterna100_targets_v2.csv
RESULTS=${ROOT}/results/eterna100_eval
LOGDIR=${ROOT}/evaluation/eterna100_logs
ACCEL=${ENV_PREFIX}/bin/accelerate
GPUS_LIST=${GPUS_LIST:-5,6,7}
BSZ=${BSZ:-8}
mkdir -p "$RESULTS" "$LOGDIR"

IFS=',' read -r -a GPUS <<< "$GPUS_LIST"
if [ "${#GPUS[@]}" -lt 3 ]; then
  echo "Need at least 3 GPUs in GPUS_LIST, got: $GPUS_LIST" >&2
  exit 2
fi

write_arnie() {
  local d="$1"
  printf 'linearpartition: .\nTMP: /tmp\n' > "$d/arnie_file.txt"
}

run_orig_one_gpu() {
  local gpu="$1" tag="$2" mode="$3" peps="$4" k="$5" port="$6"
  local out="$RESULTS/${tag}_v2"
  rm -rf "$out"
  (
    cd "$ORIG_ROOT"
    write_arnie "$ORIG_ROOT"
    export ARNIEFILE="$ORIG_ROOT/arnie_file.txt"
    echo "[$(date -u)] START $tag GPU=$gpu mode=$mode p=$peps K=$k"
    CUDA_VISIBLE_DEVICES="$gpu" NCCL_TIMEOUT=7200 "$ACCEL" launch \
      --mixed_precision bf16 --num_processes 1 --main_process_port "$port" \
      run_ok7_orig.py \
      --targets-csv "$TARGETS" \
      --k-samples "$k" --batch-size "$BSZ" \
      --out-dir "$out" \
      --checkpoint "$ORIG_CKPT" \
      --sampling-mode "$mode" --p-eps "$peps"
    echo "[$(date -u)] DONE $tag"
  ) > "$LOGDIR/${tag}_parallel.log" 2>&1
}

merge_orig_3strategies() {
  # Use the env's Python directly: `conda run ... python - <<'PY'` did not reliably
  # execute this heredoc in the observed benchmark run.
  STRUCT2SEQ_ROOT="$ROOT" "$ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path
import pandas as pd
root=Path(os.environ['STRUCT2SEQ_ROOT'])
base=root/'results/eterna100_eval'
inputs=[
    ('eps05', base/'orig_eps05_k334_v2'/'samples.csv'),
    ('eps10', base/'orig_eps10_k333_v2'/'samples.csv'),
    ('qsoftmax', base/'orig_qsoftmax_k333_v2'/'samples.csv'),
]
out=base/'orig_3strategies_k1000_v2'
out.mkdir(parents=True, exist_ok=True)
parts=[]
for tag,p in inputs:
    df=pd.read_csv(p)
    df['sampling_strategy']=tag
    offset={'eps05':0,'eps10':400000,'qsoftmax':800000}[tag]
    df['sample_idx']=df['sample_idx'].astype(int)+offset
    parts.append(df)
all_df=pd.concat(parts, ignore_index=True)
all_df.to_csv(out/'samples.csv', index=False)
rows=[]
for pi, sub in all_df.groupby('puzzle_idx'):
    rows.append({
        'config_tag':'orig_struct2seq_3strategies_eterna100_v2',
        'puzzle_idx':int(pi),
        'puzzle_id':str(sub['puzzle_id'].iloc[0]),
        'title':str(sub['title'].iloc[0]),
        'n_samples':len(sub),
        'n_perfect_jaccard':int((sub['jaccard_vs_target']==1.0).sum()),
        'p80_ok_score':float(sub['ok_score'].dropna().quantile(0.80)) if sub['ok_score'].dropna().size else float('nan'),
        'mean_jaccard':float(sub['jaccard_vs_target'].mean()),
        'mean_ok_score':float(sub['ok_score'].dropna().mean()) if sub['ok_score'].dropna().size else float('nan'),
    })
pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
print(f'wrote {out}/samples.csv rows={len(all_df)}')
print(f'wrote {out}/summary.csv rows={len(rows)}')
PY
}

run_rescue_and_report() {
  cd "$ROOT"
  write_arnie "$ROOT"
  export ARNIEFILE="$ROOT/arnie_file.txt"
  rm -rf "$RESULTS/orig_3strategies_rescue_v2"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" conda run -p "$ENV_PREFIX" python evaluation/run_rescue.py \
    --in-samples "$RESULTS/orig_3strategies_k1000_v2/samples.csv" \
    --out-dir "$RESULTS/orig_3strategies_rescue_v2" \
    --targets-csv "$TARGETS" \
    --device cuda:0 \
    > "$LOGDIR/orig_3strategies_rescue_parallel.log" 2>&1

  conda run -p "$ENV_PREFIX" python evaluation/write_eterna100_report.py \
    --targets "$TARGETS" \
    --results-dir "$RESULTS" \
    --out-md "$ROOT/eterna100_results.md" \
    > "$LOGDIR/write_report_parallel.log" 2>&1
}

cd "$ROOT"
if [ ! -f "$RESULTS/bidir_random_k1000_v2/summary.csv" ]; then
  echo "Missing bidir summary; wait for current bidir run first: $RESULTS/bidir_random_k1000_v2/summary.csv" >&2
  exit 3
fi

echo "[$(date -u)] Parallel remaining Eterna100 protocols on GPUs ${GPUS[*]}"

# Wave 1: fill all three clean GPUs with independent jobs.
run_orig_one_gpu "${GPUS[0]}" orig_l2r_argmax_k1000 epsilon 0.0 1000 29710 & p1=$!
run_orig_one_gpu "${GPUS[1]}" orig_eps05_k334 epsilon 0.05 334 29711 & p2=$!
run_orig_one_gpu "${GPUS[2]}" orig_eps10_k333 epsilon 0.10 333 29712 & p3=$!
wait "$p1" "$p2" "$p3"

# Wave 2: qsoftmax is the remaining original component.
run_orig_one_gpu "${GPUS[0]}" orig_qsoftmax_k333 qsoftmax 0.0 333 29713

merge_orig_3strategies
run_rescue_and_report

echo "[$(date -u)] DONE parallel remaining Eterna100 benchmark"
