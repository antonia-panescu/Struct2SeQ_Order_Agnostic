# Eterna training dump analysis — eterna_puzzles_20250227.csv

Source: Rhiju Das Google Drive file id `1SWGsO3kaRP43Hj7uLkoQh7I3Gb6i4YQy`.
Downloaded filename from Drive headers: `eterna_puzzles_20250227.csv`.
Local path: `/home/nvidia/haiwen/antonia/struct2seq_bidir_rl/data/eterna_training_dump/eterna_puzzles_20250227.csv`.
SHA256: `3181268938649006abd9492831c1bbb026bdcf1e1a94a30c00c1496097c9bc47`.
Size: 45320822 bytes.

## Schema / scale

- Rows: 31633
- Columns: 82 (raw export has duplicate `vid`, `nid`, `title`, and `uid` column names; analysis disambiguated them with suffixes.)
- Rows with `field_structure_value`: 31633
- Unique exact `field_structure_value` strings: 23901
- Structure length range: 1–12000 nt/chars
- Puzzle type counts: {'Challenge': 26707, 'Experimental': 4754, 'Progression': 149, 'Basic': 16, 'SwitchBasic': 6, '': 1}
- Created unix range: 1265432977–1740705842
- Changed unix max: 1740706274

## Relationship to Eterna100 benchmark

Benchmark source checked: `/home/nvidia/haiwen/antonia/struct2seq_bidir_rl/data/eterna100/eterna100_puzzles.tsv`.

- Eterna100 V1 structures present in dump: 100/100
- Eterna100 V2 structures present in dump: 100/100
- Unique Eterna IDs from benchmark present in dump `nid`/`vid` columns: 238/238
- Therefore, training on this dump as-is would leak Eterna100/Eterna100-V2 benchmark targets.
- The dump is much larger than the benchmark: it is a broad historical Eterna puzzle export, not just Eterna100.
- Use `/home/nvidia/haiwen/antonia/struct2seq_bidir_rl/data/eterna_training_dump/eterna100_overlap_report.csv` for per-benchmark-puzzle overlap details.

## Recommendation

For any model trained or fine-tuned on this dump and evaluated on Eterna100 V2, construct a leakage-controlled split that excludes at least:
1. all rows whose `nid`/`vid` matches any Eterna100 benchmark ID;
2. all rows whose exact `field_structure_value` matches any Eterna100 V1 or V2 target structure;
3. preferably near-duplicates / trivially edited variants, if the goal is strict generalization.
