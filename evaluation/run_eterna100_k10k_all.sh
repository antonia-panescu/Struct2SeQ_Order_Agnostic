#!/usr/bin/env bash
# Self-contained K=10k Eterna100 pipeline: generation (RNet) -> oracle re-fold
# (Vienna + EternaFold) -> comparison report. Designed to run fully detached
# (setsid/nohup) so it completes regardless of any SSH session.
#
# Launch:
#   cd <repo>; setsid nohup bash evaluation/run_eterna100_k10k_all.sh \
#       > /data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k/master.log 2>&1 < /dev/null &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${DATA:-/data/haiwen/antonia/struct2seq_bidir_rl}"
DONE_MARK="$DATA/eterna100_eval_k10k/.PIPELINE_COMPLETE"
mkdir -p "$DATA/eterna100_eval_k10k"
rm -f "$DONE_MARK"

echo "[$(date -u)] ===== K=10k PIPELINE START (pid $$) ====="

echo "[$(date -u)] ----- PHASE 1: generation + RNet scoring/rescue -----"
# generate script self-retries/resumes per variant; if the whole script is
# pre-empted, this loop relaunches it and it fast-skips completed variants.
a=0
until [ -f "$DATA/eterna100_eval_k10k/orig_3strategies_rescue_k10000_v2/summary.csv" ]; do
  a=$((a+1)); [ "$a" -gt 100 ] && { echo "FATAL: phase 1 incomplete after 100 relaunches"; exit 1; }
  bash evaluation/run_eterna100_k10k_generate.sh || echo "[$(date -u)] phase1 relaunch $a"
  sleep 5
done

echo "[$(date -u)] ----- PHASE 2: Vienna + EternaFold re-fold + rescue + report -----"
a=0
until [ -f "$ROOT/eterna100_oracle_comparison_k10k.md" ]; do
  a=$((a+1)); [ "$a" -gt 50 ] && { echo "FATAL: phase 2 incomplete after 50 relaunches"; exit 1; }
  bash evaluation/run_eterna100_k10k_oracles.sh || echo "[$(date -u)] phase2 relaunch $a"
  sleep 5
done

date -u > "$DONE_MARK"
echo "[$(date -u)] ===== K=10k PIPELINE COMPLETE ====="
echo "report: $ROOT/eterna100_oracle_comparison_k10k.md"
