# 2.1 Dataset development

Builds the training data for the embedding models: one row per **coverage (`CLM_CVG_PK`) × day-grid cutoff**,
with the claim's note embeddings summarized up to that cutoff. It applies the same boundary, margin-rule and
day-grid rules as `../../3-ManualFeatureBasedApproach/3.1-Dataset-Development.ipynb`. The difference is the row content: instead of the
cumulative note text, each row holds 8 summaries of the note **embeddings**.

All paths (local notes folder, S3 bucket and prefixes, region) come from `config.py` at the repo root.

## Run order

```
python 2.1.1-build_keyword_registry.py     # once: flags event keywords in all 4.6M notes          (needs NOTES_DIR)
python 2.1.2-assign_global_splits.py       # once: claim-level 70/15/15 split shared by all targets
#  ... run ../2.2-Embedding (2.2.1, 2.2.2) so the per-note embeddings exist in S3 ...
python 2.1.3-build_cutoff_dataset.py       # builds the Attorney / Litigation / Severity datasets  (reads the embeddings)
python 2.1.3-build_cutoff_dataset.py --sample-per-shard 2000 --max-shards 2    # quick test, writes under _sample/
```

The first two scripts don't need the embeddings. `2.1.3-build_cutoff_dataset.py` reads the vectors written by
`../2.2-Embedding/2.2.2-Embedding-Inference.ipynb`, so it runs after step 2.2.

## Files

| File | What it does | Output (under `CUTOFF_DATASET_PREFIX`) |
|---|---|---|
| `2.1.1-build_keyword_registry.py` | Reads every `batch_*.csv` in the same order as `2.2.1`, so `seq` matches `note_id = <claim>_<seq>`. Strips HTML and flags whether each note matches the attorney, litigation or severity event-keyword regex. Vectorized with pandas, it takes a few minutes. The margin rule can then run later without the note text. | `_keyword_registry.parquet` (`seq, claim_num, date, kw_attorney, kw_litigation, kw_severity`) |
| `2.1.2-assign_global_splits.py` | Global **claim-level** stratified 70/15/15 split, seed 42. Strata are `ORG_GROUP_NAME` + `POL_STATE` + `LOSS_YEAR` + max attorney / litigation / severity per claim. A stratum with fewer than 3 claims goes to train. It runs once, so a claim lands in the same split for all three targets. | `_SPLITS.parquet`, `_SPLITS.json` |
| `2.1.3-build_cutoff_dataset.py` | The main builder; see the pseudocode below. | `{model}/{split}/{feature}/part-NNNNN.parquet`, `_DATASET.json` |

## Pseudocode (`2.1.3-build_cutoff_dataset.py`)

```
load closed coverages + severity table; boundary per model:
    Attorney   = ATTNY_DT if attorney else CVG_CLOSE_DT
    Litigation = LITIG_DT if lawsuit  else CVG_CLOSE_DT
    Severity   = LAST_CLOSE_DT
load the split map and the keyword registry
load all per-note embeddings (2.2.2 output); sort by (claim, date, seq)
for each claim:
    running sum / max / min over its notes (so any prefix 1..n is O(1) to read)
    for each model, for each coverage of the claim:
        n_max = notes strictly before the boundary
        if no note falls on the boundary and the last kept note matches the event keyword: n_max -= 1
        for day in the model's day grid (stop once open_date + day reaches the boundary):
            n = notes before open_date + day (capped at n_max)
            skip if n == 0 or n is unchanged from the previous kept cutoff
            write 8 vectors of 1,536 dims: mean, max, min, sum, max/min, max-min, first/last, first-last
write _DATASET.json (rules, grids, schema, row counts)
```

Divisions are guarded (`|denominator| < 1e-6` is replaced by ±1e-6) and clipped to the float16 range.

## Day grids (days after `CLM_OPEN_DT`)

| Target | Grid |
|---|---|
| Attorney | 1, 7, 15, 30, 60, 90, 180, 365, 730, 1095 |
| Litigation | 1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460 |
| Severity | 1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460, 1825 |

## Result

| Target | Rows | Coverages | Label |
|---|---|---|---|
| Attorney | 466,090 | 120,502 | 191,778 positive rows (41.1%) |
| Litigation | 388,956 | 129,804 | 64,444 positive rows (16.6%) |
| Severity | 619,029 | 132,737 | band 0–4 |

Claims per split: 81,212 / 17,418 / 17,267 (train / val / holdout). The full manifests are in
`../2.4-ModelArtifacts/manifests/_DATASET.json` and `_SPLITS.json`.
