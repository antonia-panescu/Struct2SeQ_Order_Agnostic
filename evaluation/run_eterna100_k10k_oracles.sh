#!/usr/bin/env bash
# Phase 2 of the K=10k experiment: re-fold the RNet-generated K=10k sequences
# under ViennaRNA 2 and EternaFold, re-run rescue per oracle, and build the
# RNet-vs-Vienna-vs-EternaFold comparison. Run AFTER
# run_eterna100_k10k_generate.sh completes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
KN="${KN:-/home/nvidia/haiwen/yijun/miniconda3/envs/knitnet}"   # ViennaRNA + arnie
WORKERS="${WORKERS:-64}"
TARGETS="$ROOT/data/eterna100/eterna100_targets_v2.csv"
DATA="${DATA:-/data/haiwen/antonia/struct2seq_bidir_rl}"
RNET="$DATA/eterna100_eval_k10k"                       # generation output (RNet)
export ARNIEFILE="$ROOT/arnie_file.txt"
PY() { conda run -p "$KN" python "$@"; }

# Env.py rewrote arnie_file.txt during generation -> restore the eternafold entry.
printf 'linearpartition: .\neternafold: /home/nvidia/haiwen/antonia/EternaFold/src\nTMP: /tmp\n' > "$ARNIEFILE"

GEN=(bidir_random_k10000_v2 orig_3strategies_k10000_v2)

for ORACLE in vienna eternafold; do
  VROOT="$DATA/eterna100_eval_k10k_${ORACLE}"
  echo "==================== ORACLE=$ORACLE ===================="
  for V in "${GEN[@]}"; do
    echo "--- refold $V ($ORACLE) ---"
    PY evaluation/refold_score.py \
      --in-samples "$RNET/$V/samples.csv" \
      --out-dir "$VROOT/$V" \
      --targets-csv "$TARGETS" --oracle "$ORACLE" --workers "$WORKERS"
  done
  echo "--- rescue ($ORACLE) ---"
  PY evaluation/run_rescue_oracle.py \
    --in-samples "$VROOT/orig_3strategies_k10000_v2/samples.csv" \
    --out-dir "$VROOT/orig_3strategies_rescue_k10000_v2" \
    --targets-csv "$TARGETS" --oracle "$ORACLE" --workers "$WORKERS"
done

echo "==================== comparison report ===================="
CMP_VARIANTS="bidir_random_k10000_v2:S2S-bidir (random-perm, argmax);orig_3strategies_k10000_v2:Original S2S 3-strategies;orig_3strategies_rescue_k10000_v2:Original S2S 3-strategies + rescue" \
CMP_RNET_ROOT="$RNET" \
CMP_VIENNA_ROOT="$DATA/eterna100_eval_k10k_vienna" \
CMP_ETERNA_ROOT="$DATA/eterna100_eval_k10k_eternafold" \
CMP_OUT="$ROOT/eterna100_oracle_comparison_k10k.md" \
  PY evaluation/write_oracle_comparison.py
echo "done -> eterna100_oracle_comparison_k10k.md"
