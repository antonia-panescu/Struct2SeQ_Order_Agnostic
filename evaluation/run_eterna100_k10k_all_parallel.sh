#!/usr/bin/env bash
# Parallel K=10k Eterna100 pipeline: bidir generation (GPUs 0,3,4, resumes from
# its existing per-rank checkpoint at NPROC=3) runs CONCURRENTLY with the orig
# 3-strategies generation (free GPUs 1,5,6,7), instead of sequentially. Then
# merge + RNet rescue + Vienna/EternaFold re-fold + report. Fully detached.
#
#   cd <repo>; setsid bash evaluation/run_eterna100_k10k_all_parallel.sh \
#     > /data/.../eterna100_eval_k10k/master.log 2>&1 < /dev/null &
set -uo pipefail   # NOT -e: failures are handled by per-job resume-retry loops

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SS="${ENV_PREFIX:-/home/nvidia/miniconda3/envs/struct2seq}"
ORIG_ROOT="${ORIG_S2S_ROOT:-/home/nvidia/haiwen/antonia/Struct2SeQ_training}"
ORIG_CKPT="${ORIG_S2S_CHECKPOINT:-/home/nvidia/haiwen/antonia/Struct2SeQ/Struct2SeQ.pt}"
ACCEL="$SS/bin/accelerate"
TARGETS="$ROOT/data/eterna100/eterna100_targets_v2.csv"
DATA="${DATA:-/data/haiwen/antonia/struct2seq_bidir_rl}"
RESULTS="$DATA/eterna100_eval_k10k"; LOGDIR="$RESULTS/logs"
DONE_MARK="$RESULTS/.PIPELINE_COMPLETE"
mkdir -p "$RESULTS" "$LOGDIR"; rm -f "$DONE_MARK"

# bidir MUST stay NPROC=3 on GPUs 0,3,4 so its per-rank checkpoint stays valid.
BIDIR_GPUS="${BIDIR_GPUS:-0,3,4}"; BIDIR_NPROC=3
# orig sub-runs share the free GPUs, run sequentially each using all of them.
ORIG_GPUS="${ORIG_GPUS:-1,5,6,7}"; ORIG_NPROC="${ORIG_NPROC:-4}"
KS="${KS:-10000}"; BSZ="${BSZ:-8}"; K1=3334; K2=3333; K3=3333
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"

is_complete() { [ -f "$1/summary.csv" ] && [ "$(wc -l < "$1/summary.csv" 2>/dev/null || echo 0)" -ge 101 ]; }

gen_bidir() {
  local out="$RESULTS/bidir_random_k10000_v2" a=0
  cd "$ROOT"
  until is_complete "$out"; do
    a=$((a+1)); [ "$a" -gt "$MAX_ATTEMPTS" ] && { echo "FATAL bidir x$MAX_ATTEMPTS"; return 1; }
    echo "[$(date -u)] bidir K=$KS gpus=$BIDIR_GPUS (attempt $a, resumes)" | tee -a "$LOGDIR/bidir.log"
    CUDA_VISIBLE_DEVICES="$BIDIR_GPUS" NCCL_TIMEOUT=14400 "$ACCEL" launch \
      --mixed_precision bf16 --num_processes "$BIDIR_NPROC" --main_process_port 29620 \
      evaluation/run_ok7_eval.py --targets-csv "$TARGETS" --k-samples "$KS" \
      --batch-size "$BSZ" --out-dir "$out" --model bidir --inference-mode random \
      >> "$LOGDIR/bidir.log" 2>&1 || echo "[$(date -u)] bidir attempt $a nonzero; resume"
    sleep 5
  done
  echo "[$(date -u)] bidir COMPLETE"
}

orig_one() {  # tag mode peps k port
  local tag="$1" mode="$2" peps="$3" k="$4" port="$5"
  local out="$RESULTS/${tag}_v2" a=0
  until is_complete "$out"; do
    a=$((a+1)); [ "$a" -gt "$MAX_ATTEMPTS" ] && { echo "FATAL $tag x$MAX_ATTEMPTS"; return 1; }
    echo "[$(date -u)] orig $tag gpus=$ORIG_GPUS (attempt $a)" | tee -a "$LOGDIR/${tag}.log"
    cd "$ORIG_ROOT"
    CUDA_VISIBLE_DEVICES="$ORIG_GPUS" NCCL_TIMEOUT=14400 "$ACCEL" launch \
      --mixed_precision bf16 --num_processes "$ORIG_NPROC" --main_process_port "$port" \
      run_ok7_orig.py --targets-csv "$TARGETS" --k-samples "$k" --batch-size "$BSZ" \
      --out-dir "$out" --checkpoint "$ORIG_CKPT" --sampling-mode "$mode" --p-eps "$peps" \
      >> "$LOGDIR/${tag}.log" 2>&1 || echo "[$(date -u)] $tag attempt $a nonzero; resume"
    cd "$ROOT"; sleep 5
  done
  echo "[$(date -u)] orig $tag COMPLETE"
}

gen_orig_group() {   # 3 sub-runs sequential, each on the free-GPU pool
  orig_one orig_eps05_k${K1} epsilon 0.05 "$K1" 29631
  orig_one orig_eps10_k${K2} epsilon 0.10 "$K2" 29632
  orig_one orig_qsoftmax_k${K3} qsoftmax 0.0 "$K3" 29633
}

echo "[$(date -u)] ===== K=10k PARALLEL PIPELINE START (pid $$) ====="
echo "[$(date -u)] bidir on $BIDIR_GPUS || orig 3-strat on $ORIG_GPUS"

# ---- Phase 1: bidir and orig generation CONCURRENTLY ----
gen_bidir &      BPID=$!
gen_orig_group & OPID=$!
wait "$BPID"; brc=$?
wait "$OPID"; orc=$?
echo "[$(date -u)] generation finished (bidir rc=$brc orig rc=$orc)"

# ---- merge 3-strategies (idempotent) ----
RESULTS="$RESULTS" K1="$K1" K2="$K2" K3="$K3" "$SS/bin/python" - <<'PY'
import os; from pathlib import Path; import pandas as pd
base=Path(os.environ['RESULTS'])
inp=[('eps05',base/f"orig_eps05_k{os.environ['K1']}_v2"/'samples.csv'),
     ('eps10',base/f"orig_eps10_k{os.environ['K2']}_v2"/'samples.csv'),
     ('qsoftmax',base/f"orig_qsoftmax_k{os.environ['K3']}_v2"/'samples.csv')]
out=base/'orig_3strategies_k10000_v2'; out.mkdir(parents=True,exist_ok=True)
parts=[]
for tag,p in inp:
    df=pd.read_csv(p); df['sampling_strategy']=tag
    df['sample_idx']=df['sample_idx'].astype(int)+{'eps05':0,'eps10':4_000_000,'qsoftmax':8_000_000}[tag]
    parts.append(df)
alld=pd.concat(parts,ignore_index=True); alld.to_csv(out/'samples.csv',index=False)
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

# ---- RNet rescue (retry) ----
ro="$RESULTS/orig_3strategies_rescue_k10000_v2"; a=0
until is_complete "$ro"; do
  a=$((a+1)); [ "$a" -gt "$MAX_ATTEMPTS" ] && { echo "FATAL rnet-rescue"; exit 1; }
  echo "[$(date -u)] RNet rescue (attempt $a)" | tee -a "$LOGDIR/rescue_rnet.log"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="${BIDIR_GPUS%%,*}" conda run -p "$SS" python evaluation/run_rescue.py \
    --in-samples "$RESULTS/orig_3strategies_k10000_v2/samples.csv" \
    --out-dir "$ro" --targets-csv "$TARGETS" --device cuda:0 \
    >> "$LOGDIR/rescue_rnet.log" 2>&1 || echo "[$(date -u)] rnet-rescue $a nonzero; retry"
  sleep 5
done

# ---- Phase 2: Vienna + EternaFold re-fold + rescue + report (retry) ----
a=0
until [ -f "$ROOT/eterna100_oracle_comparison_k10k.md" ]; do
  a=$((a+1)); [ "$a" -gt 50 ] && { echo "FATAL phase2"; exit 1; }
  echo "[$(date -u)] PHASE 2 (attempt $a)"
  bash evaluation/run_eterna100_k10k_oracles.sh || echo "[$(date -u)] phase2 $a nonzero; retry"
  sleep 5
done

date -u > "$DONE_MARK"
echo "[$(date -u)] ===== K=10k PARALLEL PIPELINE COMPLETE ====="
