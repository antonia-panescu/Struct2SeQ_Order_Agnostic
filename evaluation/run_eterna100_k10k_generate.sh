#!/usr/bin/env bash
# K=10,000 Eterna100-V2 generation for the diversity variants only:
#   - S2S-bidir (random-perm, argmax)
#   - Original S2S 3-strategies (eps05 + eps10 + qsoftmax, ~3334 each) + merge
#   - Original S2S 3-strategies + RNet rescue
# (L->R argmax is deterministic -> skipped; gains nothing from more samples.)
#
# RibonanzaNet-scored inline. Outputs to /data (root FS is full). The Vienna /
# EternaFold re-folding is a separate step (run_eterna100_k10k_oracles.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SS="${ENV_PREFIX:-/home/nvidia/miniconda3/envs/struct2seq}"     # torch + RNet + accelerate
ORIG_ROOT="${ORIG_S2S_ROOT:-/home/nvidia/haiwen/antonia/Struct2SeQ_training}"
ORIG_CKPT="${ORIG_S2S_CHECKPOINT:-/home/nvidia/haiwen/antonia/Struct2SeQ/Struct2SeQ.pt}"
ACCEL="$SS/bin/accelerate"
TARGETS="$ROOT/data/eterna100/eterna100_targets_v2.csv"
RESULTS="${RESULTS:-/data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k}"
LOGDIR="$RESULTS/logs"
GPUS="${GPUS:-0,3,4}"
NPROC="${NPROC:-3}"
BSZ="${BSZ:-8}"
KS="${KS:-10000}"
# 3-strategy split (sums to KS)
K1="${K1:-3334}"; K2="${K2:-3333}"; K3="${K3:-3333}"

mkdir -p "$RESULTS" "$LOGDIR"
echo "[$(date -u)] K=$KS generation -> $RESULTS  (GPUS=$GPUS)"

run_bidir() {
  local out="$RESULTS/bidir_random_k10000_v2"; rm -rf "$out"
  echo "[$(date -u)] bidir_random K=$KS" | tee "$LOGDIR/bidir.log"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$GPUS" NCCL_TIMEOUT=14400 "$ACCEL" launch \
    --mixed_precision bf16 --num_processes "$NPROC" --main_process_port 29620 \
    evaluation/run_ok7_eval.py \
    --targets-csv "$TARGETS" --k-samples "$KS" --batch-size "$BSZ" \
    --out-dir "$out" --model bidir --inference-mode random \
    2>&1 | tee -a "$LOGDIR/bidir.log"
}

run_orig_one() {
  local tag="$1" mode="$2" peps="$3" k="$4" port="$5"
  local out="$RESULTS/${tag}_v2"; rm -rf "$out"
  echo "[$(date -u)] orig $tag mode=$mode p=$peps K=$k" | tee "$LOGDIR/${tag}.log"
  cd "$ORIG_ROOT"
  CUDA_VISIBLE_DEVICES="$GPUS" NCCL_TIMEOUT=14400 "$ACCEL" launch \
    --mixed_precision bf16 --num_processes "$NPROC" --main_process_port "$port" \
    run_ok7_orig.py \
    --targets-csv "$TARGETS" --k-samples "$k" --batch-size "$BSZ" \
    --out-dir "$out" --checkpoint "$ORIG_CKPT" \
    --sampling-mode "$mode" --p-eps "$peps" \
    2>&1 | tee -a "$LOGDIR/${tag}.log"
  cd "$ROOT"
}

merge_3strat() {
  RESULTS="$RESULTS" K1="$K1" K2="$K2" K3="$K3" "$SS/bin/python" - <<'PY'
import os
from pathlib import Path
import pandas as pd
base=Path(os.environ['RESULTS'])
inputs=[('eps05',base/f"orig_eps05_k{os.environ['K1']}_v2"/'samples.csv'),
        ('eps10',base/f"orig_eps10_k{os.environ['K2']}_v2"/'samples.csv'),
        ('qsoftmax',base/f"orig_qsoftmax_k{os.environ['K3']}_v2"/'samples.csv')]
out=base/'orig_3strategies_k10000_v2'; out.mkdir(parents=True,exist_ok=True)
parts=[]
for tag,p in inputs:
    df=pd.read_csv(p); df['sampling_strategy']=tag
    off={'eps05':0,'eps10':4_000_000,'qsoftmax':8_000_000}[tag]
    df['sample_idx']=df['sample_idx'].astype(int)+off
    parts.append(df)
alld=pd.concat(parts,ignore_index=True)
alld.to_csv(out/'samples.csv',index=False)
rows=[]
for pi,sub in alld.groupby('puzzle_idx'):
    rows.append({'config_tag':'orig_3strategies_k10000_v2','puzzle_idx':int(pi),
        'puzzle_id':str(sub['puzzle_id'].iloc[0]),'title':str(sub['title'].iloc[0]),
        'n_samples':len(sub),'n_perfect_jaccard':int((sub['jaccard_vs_target']==1.0).sum()),
        'p80_ok_score':float(sub['ok_score'].dropna().quantile(0.80)) if sub['ok_score'].dropna().size else float('nan'),
        'mean_jaccard':float(sub['jaccard_vs_target'].mean()),
        'mean_ok_score':float(sub['ok_score'].dropna().mean()) if sub['ok_score'].dropna().size else float('nan')})
pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
print(f'merged -> {out} rows={len(alld)} puzzles={len(rows)}')
PY
}

run_rescue_rnet() {
  local out="$RESULTS/orig_3strategies_rescue_k10000_v2"; rm -rf "$out"
  echo "[$(date -u)] RNet rescue" | tee "$LOGDIR/rescue_rnet.log"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" conda run -p "$SS" python evaluation/run_rescue.py \
    --in-samples "$RESULTS/orig_3strategies_k10000_v2/samples.csv" \
    --out-dir "$out" --targets-csv "$TARGETS" --device cuda:0 \
    2>&1 | tee -a "$LOGDIR/rescue_rnet.log"
}

run_bidir
run_orig_one orig_eps05_k${K1} epsilon 0.05 "$K1" 29621
run_orig_one orig_eps10_k${K2} epsilon 0.10 "$K2" 29622
run_orig_one orig_qsoftmax_k${K3} qsoftmax 0.0 "$K3" 29623
merge_3strat
run_rescue_rnet
echo "[$(date -u)] DONE K=$KS generation + RNet scoring/rescue -> $RESULTS"
