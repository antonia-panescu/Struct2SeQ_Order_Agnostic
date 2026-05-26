#!/usr/bin/env bash
# Wait for current bidir Eterna100 run to finish, stop the old sequential driver,
# then launch the optimized parallel remaining-protocol runner.
#
# NOTE: this script was written for a specific live recovery (the pkill PIDs
# below were the actual running driver on 2026-05-25). On a fresh machine the
# pkill block is a no-op; the rest is portable.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${STRUCT2SEQ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS=$ROOT/results/eterna100_eval
LOGDIR=$ROOT/evaluation/eterna100_logs
mkdir -p "$LOGDIR"
cd "$ROOT"

echo "[$(date -u)] watcher started" >> "$LOGDIR/parallel_takeover_watch.log"
while [ ! -f "$RESULTS/bidir_random_k1000_v2/summary.csv" ]; do
  sleep 60
done

echo "[$(date -u)] bidir summary detected; stopping old sequential driver if still present" >> "$LOGDIR/parallel_takeover_watch.log"
# Stop the old sequential bash driver before it begins/continues serial original runs.
# If it has already started original eval briefly, terminate only Eterna100 original eval descendants.
pkill -TERM -P 598314 2>/dev/null || true
kill -TERM 598314 598306 2>/dev/null || true
sleep 10
pkill -KILL -P 598314 2>/dev/null || true
kill -KILL 598314 598306 2>/dev/null || true

# If the race allowed original eval to start, remove partial original dirs before clean parallel relaunch.
rm -rf \
  "$RESULTS/orig_l2r_argmax_k1000_v2" \
  "$RESULTS/orig_eps05_k334_v2" \
  "$RESULTS/orig_eps10_k333_v2" \
  "$RESULTS/orig_qsoftmax_k333_v2" \
  "$RESULTS/orig_3strategies_k1000_v2" \
  "$RESULTS/orig_3strategies_rescue_v2"

echo "[$(date -u)] launching parallel remaining runner" >> "$LOGDIR/parallel_takeover_watch.log"
GPUS_LIST=5,6,7 BSZ=8 bash evaluation/run_eterna100_remaining_parallel.sh \
  > "$LOGDIR/remaining_parallel_driver.log" 2>&1
status=$?
echo "[$(date -u)] parallel runner exited status=$status" >> "$LOGDIR/parallel_takeover_watch.log"
exit "$status"
