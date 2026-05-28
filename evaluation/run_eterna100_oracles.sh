#!/usr/bin/env bash
# Re-score the 4 Eterna100 variants under ViennaRNA and EternaFold oracles by
# re-folding the cached generated sequences (generation is oracle-independent),
# and re-running the 4^k rescue per oracle. Originals under
# results/eterna100_eval/ are never touched.
#
#   KN=/path/to/knitnet_env  WORKERS=16  bash evaluation/run_eterna100_oracles.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export ARNIEFILE="$ROOT/arnie_file.txt"

KN="${KN:-/home/nvidia/haiwen/yijun/miniconda3/envs/knitnet}"
WORKERS="${WORKERS:-16}"
TARGETS="$ROOT/data/eterna100/eterna100_targets_v2.csv"
SRC="$ROOT/results/eterna100_eval"                 # RNet-scored cache (inputs)
PY() { conda run -p "$KN" python "$@"; }

GEN_VARIANTS=(bidir_random_k1000_v2 orig_l2r_argmax_k1000_v2 orig_3strategies_k1000_v2)

for ORACLE in vienna eternafold; do
  OUT="$ROOT/results/eterna100_eval_${ORACLE}"
  echo "==================== ORACLE=$ORACLE ===================="
  # 1) re-fold the three generation variants
  for V in "${GEN_VARIANTS[@]}"; do
    echo "--- refold $V ($ORACLE) ---"
    PY evaluation/refold_score.py \
      --in-samples "$SRC/$V/samples.csv" \
      --out-dir "$OUT/$V" \
      --targets-csv "$TARGETS" \
      --oracle "$ORACLE" --workers "$WORKERS"
  done
  # 2) re-run rescue on the oracle-rescored 3-strategies samples
  echo "--- rescue ($ORACLE) ---"
  PY evaluation/run_rescue_oracle.py \
    --in-samples "$OUT/orig_3strategies_k1000_v2/samples.csv" \
    --out-dir "$OUT/orig_3strategies_rescue_v2" \
    --targets-csv "$TARGETS" \
    --oracle "$ORACLE" --workers "$WORKERS"
done

echo "==================== building comparison report ===================="
PY evaluation/write_oracle_comparison.py
echo "done -> eterna100_oracle_comparison.md"
