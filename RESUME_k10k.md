# RESUME — Eterna100 K=10,000 oracle experiment (started 2026-05-28 22:16 UTC)

## TL;DR
A fully-detached pipeline is generating K=10,000 samples/target for the
diversity variants and will auto-score them under RibonanzaNet / ViennaRNA 2 /
EternaFold, then write the comparison report. Nothing to babysit.

## Is it done?
```
ls /data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k/.PIPELINE_COMPLETE   # exists => done
cat /home/nvidia/haiwen/antonia/struct2seq_bidir_rl/eterna100_oracle_comparison_k10k.md  # the result
```

## Is it still running / progress?
```
ps -o pid,etime,cmd -p 3994227                       # master (PPID 1, detached)
tail -40 /data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k/master.log
tail -2  /data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k/logs/bidir.log   # tqdm %
nvidia-smi -i 0,3,4
```

## What it does (one detached chain: evaluation/run_eterna100_k10k_all.sh)
- Phase 1 (GPUs 0,3,4, struct2seq env): bidir random-perm K=10k; orig
  3-strategies (eps05+eps10+qsoftmax ~3334 each) -> merge; RNet rescue.
  Output: `/data/.../eterna100_eval_k10k/` (RNet-scored).
- Phase 2 (knitnet env): refold all under Vienna 2 + EternaFold
  (`refold_score.py`), per-oracle rescue (`run_rescue_oracle.py`), then
  `write_oracle_comparison.py` -> `eterna100_oracle_comparison_k10k.md`.
- L->R argmax is intentionally SKIPPED (deterministic; gains nothing from more K).

## Resilience (pre-emption / disconnect)
- Detached via `setsid` -> survives SSH and any Claude session ending.
- run_ok7_*.py checkpoint per batch (per-rank append CSV + batches_done.txt).
  The launcher does NOT rm -rf, and wraps each variant/phase in a bounded
  resume-retry loop, so a killed/pre-empted job RESUMES from the last batch.
- A variant is "done" only when its summary.csv has 101 lines (100 puzzles).

## If it died and didn't auto-restart
Relaunch the PARALLEL master (current) — fast-skips completed batches/variants.
bidir MUST stay on the same 3 GPUs (NPROC=3) so its per-rank checkpoint is valid:
```
cd /home/nvidia/haiwen/antonia/struct2seq_bidir_rl
BIDIR_GPUS=0,3,4 ORIG_GPUS=1,5,6,7 setsid bash evaluation/run_eterna100_k10k_all_parallel.sh \
  > /data/haiwen/antonia/struct2seq_bidir_rl/eterna100_eval_k10k/master.log 2>&1 < /dev/null &
```
(Change BIDIR_GPUS/ORIG_GPUS to whatever is free; `evaluation/run_eterna100_k10k_all.sh`
is the older sequential fallback. Kill stale procs by PID, not `pkill -f` — the
script text matches the patterns and would kill your own shell.)

## When complete — to finish the task
1. Read `eterna100_oracle_comparison_k10k.md` (solved + graded-Jaccard tables).
2. Append results to lab notebook (`lab_notebook/2026-05-28.md`, local/gitignored).
3. `git add` the K=10k launchers + report and push (already committed locally:
   036289b, 38dca71) — push is the "update lab notebook" trigger.

## Context / status as of sign-off
- 2026-05-28 ~23:21 UTC: bidir ~5% (2096/41667 batches/rank). ETA volatile
  (2.7–7+ s/it) because **GPU 0 is shared** with another user's job; total
  wall-clock ~2–4 days depending on contention. Won't fail (NCCL_TIMEOUT=4h).
- Completed earlier today: the **K=1000** oracle comparison
  (`eterna100_oracle_comparison.md`, pushed as 61c6de4). Solved:
  bidir 59/52/64, orig-argmax 25/30/36, 3-strat 52/45/58, +rescue 59/49/61
  (RNet/Vienna/EternaFold). bidir best or tied-best under all 3 oracles.
- Targets: Eterna100 **V2**; Vienna engine = **v2** (RNAfold -d2, Turner 2004).
- Checkpoint under eval = episode-5 / May-8 `best_policy_network.pt` (0.8937).
