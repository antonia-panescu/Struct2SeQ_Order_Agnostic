#!/usr/bin/env bash
# Eterna100 V2 benchmark: ours bidir_random and original Struct2SeQ baselines.
#
# Required environment (override defaults if your layout differs):
#   STRUCT2SEQ_ROOT   path to this repo (auto-derived from script location)
#   ORIG_S2S_ROOT     path to the original Struct2SeQ_training repo
#                     (needs run_ok7_orig.py at its root)
#   ORIG_S2S_CHECKPOINT
#                     path to the original Struct2SeQ.pt baseline checkpoint
#   CONDA_SH          path to conda.sh (e.g. ~/miniconda3/etc/profile.d/conda.sh)
#   ENV_PREFIX        conda env prefix containing accelerate + python deps
#   GPUS, NPROC, BSZ, KS  optional knobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${STRUCT2SEQ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ORIG_ROOT="${ORIG_S2S_ROOT:?ORIG_S2S_ROOT must point at the original Struct2SeQ_training repo}"
ORIG_CKPT="${ORIG_S2S_CHECKPOINT:?ORIG_S2S_CHECKPOINT must point at Struct2SeQ.pt}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_PREFIX="${ENV_PREFIX:?ENV_PREFIX must point at the conda env containing accelerate}"

source "$CONDA_SH"
ACCEL=${ENV_PREFIX}/bin/accelerate
TARGETS=${ROOT}/data/eterna100/eterna100_targets_v2.csv
RESULTS=${ROOT}/results/eterna100_eval
LOGDIR=${ROOT}/evaluation/eterna100_logs
GPUS=${GPUS:-5,6,7}
NPROC=${NPROC:-3}
BSZ=${BSZ:-8}
KS=${KS:-1000}

mkdir -p "$RESULTS" "$LOGDIR"
cd "$ROOT"
printf 'linearpartition: .\nTMP: /tmp\n' > "$ROOT/arnie_file.txt"
export ARNIEFILE="$ROOT/arnie_file.txt"

run_bidir_random() {
  local out="$RESULTS/bidir_random_k1000_v2"
  rm -rf "$out"
  echo "[$(date -u)] bidir_random K=$KS on Eterna100 V2" | tee "$LOGDIR/bidir_random_k1000.log"
  CUDA_VISIBLE_DEVICES="$GPUS" NCCL_TIMEOUT=7200 "$ACCEL" launch \
    --mixed_precision bf16 --num_processes "$NPROC" --main_process_port 29610 \
    evaluation/run_ok7_eval.py \
    --targets-csv "$TARGETS" \
    --k-samples "$KS" --batch-size "$BSZ" \
    --out-dir "$out" \
    --model bidir --inference-mode random \
    2>&1 | tee -a "$LOGDIR/bidir_random_k1000.log"
}

run_orig_one() {
  local tag="$1" mode="$2" peps="$3" k="$4" port="$5"
  local out="$RESULTS/${tag}_v2"
  rm -rf "$out"
  echo "[$(date -u)] orig ${tag} mode=${mode} p=${peps} K=${k}" | tee "$LOGDIR/${tag}.log"
  cd "$ORIG_ROOT"
  printf 'linearpartition: .\nTMP: /tmp\n' > "$ORIG_ROOT/arnie_file.txt"
  export ARNIEFILE="$ORIG_ROOT/arnie_file.txt"
  CUDA_VISIBLE_DEVICES="$GPUS" NCCL_TIMEOUT=7200 "$ACCEL" launch \
    --mixed_precision bf16 --num_processes "$NPROC" --main_process_port "$port" \
    run_ok7_orig.py \
    --targets-csv "$TARGETS" \
    --k-samples "$k" --batch-size "$BSZ" \
    --out-dir "$out" \
    --checkpoint "$ORIG_CKPT" \
    --sampling-mode "$mode" --p-eps "$peps" \
    2>&1 | tee -a "$LOGDIR/${tag}.log"
  cd "$ROOT"
  export ARNIEFILE="$ROOT/arnie_file.txt"
}

merge_orig_3strategies() {
  # Use the env's Python directly: `conda run ... python - <<'PY'` did not reliably
  # execute this heredoc in the observed run, causing rescue to miss merged samples.
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
    # make sample_idx unique across merged strategy files
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

run_rescue_orig3() {
  local out="$RESULTS/orig_3strategies_rescue_v2"
  rm -rf "$out"
  echo "[$(date -u)] rescue original 3-strategy" | tee "$LOGDIR/orig_3strategies_rescue.log"
  CUDA_VISIBLE_DEVICES=5 conda run -p "$ENV_PREFIX" python evaluation/run_rescue.py \
    --in-samples "$RESULTS/orig_3strategies_k1000_v2/samples.csv" \
    --out-dir "$out" \
    --targets-csv "$TARGETS" \
    --device cuda:0 \
    2>&1 | tee -a "$LOGDIR/orig_3strategies_rescue.log"
}

run_orig_l2r_argmax() {
  run_orig_one orig_l2r_argmax_k1000 epsilon 0.0 "$KS" 29611
}

write_report() {
  conda run -p "$ENV_PREFIX" python evaluation/write_eterna100_report.py \
    --targets "$TARGETS" \
    --results-dir "$RESULTS" \
    --out-md "$ROOT/eterna100_results.md"
}

run_bidir_random
run_orig_l2r_argmax
run_orig_one orig_eps05_k334 epsilon 0.05 334 29612
run_orig_one orig_eps10_k333 epsilon 0.10 333 29613
run_orig_one orig_qsoftmax_k333 qsoftmax 0.0 333 29614
merge_orig_3strategies
run_rescue_orig3
write_report

echo "[$(date -u)] DONE Eterna100 benchmark"
