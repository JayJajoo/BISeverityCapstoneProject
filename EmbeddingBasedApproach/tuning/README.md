# Optuna Tuning Job for ClaimNet

Hyperparameter search for the embedding model (`ClaimNet`) in `../05-Model-Training.ipynb`. One job per target
(**Attorney**, **Litigation**) runs on a SageMaker GPU instance. It tries 50 configurations, keeps the one with the
best **validation Gini**, and writes every trial's result to S3. `05` then reads those results back (Step 4b) and
retrains the winning configuration (Step 4c).

```
05, Steps 1–2 ──▶ att_data / lit_data (train / val / holdout arrays)
                     │
                     ▼  launch_tuning.upload_arrays()   →  S3: tuning_data/<target>/<split>/{X,y,cvg}.npy
                     ▼  launch_tuning.launch()          →  SageMaker training job (ml.g5.4xlarge)
                                                             │  runs train_optuna.py
                                                             ▼
                                         S3: runs/<study>/trials/, all_trials.json, _SUMMARY.json
                                                             │
05, Step 4b (show every trial) ◀─────────────────────────────┘ ──▶ Step 4c (rebuild best model)
```

## Files

| File | What it does |
|---|---|
| `claimnet.py` | The shared model code: the `ClaimNet` network, `train_one` (training loop with early stopping), `predict`, the `gini` metric, `last_snapshot_gini`, and `sample_cfg` (the Optuna search space). These are the same definitions as Steps 3–4 of `05`. |
| `train_optuna.py` | Entry point inside the training job. Loads the arrays, runs the Optuna study, saves each trial to S3 as it finishes, then writes the summary. |
| `launch_tuning.py` | Runs from the notebook. `upload_arrays()` saves `X.npy` / `y.npy` / `cvg.npy` for each split to S3. `launch()` starts the SageMaker PyTorch 2.6 job. |
| `requirements.txt` | Extra packages installed in the training container: `optuna`, `scikit-learn`, `pandas`. |

## How the search works

**Pseudocode (`train_optuna.py`)**

```
load train / val / holdout arrays   (X: rows x 8 summaries x 1536, y: 0/1, cvg: coverage id per row)
replace any inf / NaN in X with finite values   (ratio features can overflow float16)
for trial in 1..50 (TPE sampler, seed 42):
    sample a configuration from the search space
    train ClaimNet on train; after every epoch compute validation Gini; stop after 8 epochs without gain
    record validation Gini (used for selection), holdout Gini (reporting only),
        last-snapshot Gini per coverage, best epoch, run time
    save the trial record to S3 immediately
    if the trial crashes (e.g. GPU out of memory) -> mark it failed and continue
best trial = highest validation Gini
write all_trials.json and _SUMMARY.json
```

**Search space**

| Parameter | Range |
|---|---|
| Learning rate | 1e-4 – 5e-3 (log scale) |
| Dropout | 0 – 0.7 |
| Weight decay | 1e-6 – 3e-2 (log scale) |
| Input noise std | 0 – 0.05 |
| Batch size | 32, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768 |
| Fusion dim | 128, 256, 384, 512 |
| Encoder dim per summary (8 separate choices) | 64, 96, 128, 192, 256 |

**Fixed settings:**
- loss: `BCEWithLogitsLoss` with `pos_weight = negatives / positives`, for class imbalance;
- optimizer and schedule: AdamW with a cosine learning-rate schedule;
- training length: at most 60 epochs, early stopping with patience 8;
- seed: 42.

**Why validation Gini:** Gini (= 2·AUC − 1) is the ranking metric the project reports, and it doesn't depend on
a threshold or on class balance. The holdout score is recorded for every trial, but it is never used to pick one.

## How to run

From `05-Model-Training.ipynb`, after Steps 1–2 have built `att_data` and `lit_data` (Step 4a):

```python
from tuning.launch_tuning import upload_arrays, launch

att_job = launch('Attorney', 'attorney', upload_arrays(att_data, 'Attorney'))
lit_job = launch('Litigation', 'litigation', upload_arrays(lit_data, 'Litigation'))
```

- `launch()` returns right away (`wait=False`); the job runs in the background. A full run took about 9 h for
  attorney and 13 h for litigation, and the job is capped at 24 h.
- Bucket, prefixes, region and the SageMaker role all come from `config.py` at the repo root. Set
  `SAGEMAKER_ROLE_ARN` there before launching.
- Instance: `ml.g5.4xlarge`, with one A10G (24 GB) GPU and 64 GB RAM, enough to hold the attorney training
  array (about 8 GB in float16) in memory.

## Outputs (under `{EMBEDDING_PREFIX}` in `config.py`)

| Path | Content |
|---|---|
| `tuning_data/<target>/<split>/X.npy, y.npy, cvg.npy` | arrays the job trains and scores on |
| `runs/<study>/trials/trial_XXX.json` | one record per finished trial |
| `runs/<study>/all_trials.json` | all completed trials: `trial`, `val_gini`, `holdout_gini`, `claim_gini` (last snapshot per coverage, val and holdout), `best_epoch`, `seconds`, `params` |
| `runs/<study>/_SUMMARY.json` | `n_trials`, `n_completed`, `elapsed_sec`, `best_trial`, `best_val_gini`, `best_holdout_gini`, `best_params` |

## Results

| Study | Trials completed | Best trial | Val Gini | Holdout Gini |
|---|---|---|---|---|
| `attorney` | 43 of 50 | 42 | 0.500 | 0.461 |
| `litigation` | 49 of 50 | 24 | 0.449 | 0.465 |

The full trial tables, best parameters and per-day holdout Gini are in `05` (Steps 4b–6) and in `../README.md`.
