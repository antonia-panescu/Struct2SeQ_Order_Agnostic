# RESUME — original Struct2SeQ +5-episode continuation (L→R)

> **2026-06-02 rewrite.** The original run (started 2026-05-30) was lost when the
> cluster admin unmounted `/data` ~2026-06-01 — all of `struct2seq_continue/`
> (checkpoints 10–14, `master.log`, `rewards_log.csv`) is gone, no off-disk copy.
> No progress logs survived. Re-running from scratch with a loss-resistant layout.
> See `lab_notebook/2026-06-02.md`.

## What
Train the ORIGINAL Struct2SeQ (LSTM-PE, **L→R** generation) 5 more episodes in its
own recipe, from `Struct2SeQ.pt`, as the fair control vs S2S-bidir. Code:
`/home/nvidia/haiwen/antonia/Struct2SeQ_training/run_continue.py` (+ `.sh`, `continue_5ep.yaml`).

## Storage layout (NEW — survives the next /data-style unmount)
- **Checkpoints + logs + `rewards_log.csv`** → `/home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/`
  on the **root disk** (`/dev/root`) — can't be unmounted without downing the VM.
- **Bulky play data** (`tmp/episode*/process*/data.pt`) → `/scratch/haiwen/antonia/struct2seq_orig_LtoR_continue/playdata/`
  (9.7T persistent managed disk), symlinked in as `$OUT/tmp`. Regenerable, so OK to live on scratch.
- Both dirs carry a `README_LABEL.txt` describing the L→R regime.

## Status / done?
```
ls /home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/final_policy_network.pt  # exists => done
cat /home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/rewards_log.csv          # per-episode test reward
tail -5 /home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/master.log
ls /home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/policy_network_1?.pt      # 10..14 as episodes finish
nvidia-smi -i 1,2,5,6
```

## Resilience
- Detached via `setsid` (survives SSH/session end). 4 GPUs = 1,2,5,6.
- Per-episode resume: on relaunch loads the highest `policy_network_{e}.pt` and
  continues; per-epoch + mid-epoch (every 5000 steps) checkpoints too. Play
  data.pt skip-if-exists. Bounded resume-retry loop in `run_continue.sh`.
- If `/scratch` play data is ever lost but root checkpoints survive, resume just
  regenerates play data for the unfinished episode and skips ahead — no real loss.

## To launch / relaunch
```
cd /home/nvidia/haiwen/antonia/Struct2SeQ_training
GPUS=1,2,5,6 setsid bash run_continue.sh \
  > /home/nvidia/haiwen/antonia/struct2seq_orig_LtoR_continue/master.log 2>&1 < /dev/null &
```
(Kill stale procs by PID, not `pkill -f` — the pattern matches your own shell.)

## Semantics (faithful continuation)
continue_5ep.yaml: n_episodes=15, launched `--start-episode 10` → episodes 10-14
(displayed "11/15".."15/15"), all CosineAnnealingLR (no warmup), train_batch_size
fast-forwarded to the converged 64, p held at 0.2 (original's final_p). Recipe =
original default_config.yaml (lr=1e-3). LSTM-PE arch + L→R generation unchanged.

## ETA
~16h play/episode (CPU-bound RNet reward) → ~3.5-4 days for 5 episodes.

## When done → downstream
Score the new checkpoint (best_policy_network.pt or final) on Eterna100-V2 with
the OFFICIAL Vienna 2 MFE criterion (run_ok7_orig.py + refold pipeline) at matched
K, and add an "original +5ep (L→R, 15 episodes)" row beside original-10ep & bidir.
