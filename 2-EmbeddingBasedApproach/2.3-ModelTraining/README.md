# 2.3 Model training (Optuna on SageMaker)

Trains one neural network per study on the step 2.1 datasets. Hyperparameters are tuned with Optuna in a
single-GPU SageMaker job (`ml.g5.4xlarge`).

| Study | Target | Outputs | Loss | Selected on (validation) |
|---|---|---|---|---|
| `attorney` | Attorney | 1 logit | `BCEWithLogits`, `pos_weight = neg/pos` | Gini |
| `litigation` | Litigation | 1 logit | `BCEWithLogits`, `pos_weight = neg/pos` | Gini |
| `severity_gini` | Severity | 5 logits | class-weighted cross-entropy | ordinal Gini |
| `severity_qwk` | Severity | 5 logits | class-weighted cross-entropy | QWK |

The two severity studies share the data, the network and the search space. They differ only in the metric that
picks the best epoch and the best trial. Both metrics are logged every epoch either way.

## Files

| File | What it does |
|---|---|
| `model.py` | `CutoffNet`: one encoder per summary (Linear → LayerNorm → GELU → Dropout), then concatenation, a fusion layer, an MLP head and `num_classes` logits. |
| `data.py` | Streams each (target, split) parquet folder from S3 into a float16 memmap of shape (N, 8, 1536). `last_snapshot_indices` finds each coverage's final cutoff row. |
| `engine.py` | Training loop: AdamW, cosine LR schedule, Gaussian input noise and early stopping. Logs train / val / holdout Gini every epoch, plus QWK and macro-F1 for severity, and last-snapshot scores. Also defines the metrics, including `ordinal_cumulative_gini`. |
| `2.3.2-tune.py` | Optuna entry point (`--model`, `--primary-metric`): TPE sampler (seed 42), median pruner, 50 trials. Saves every trial's checkpoint, a trial JSON with the full per-epoch history, a resumable `study.db` and `_SUMMARY.json` to S3. |
| `2.3.1-launch_sagemaker.py` | Runs on your machine. Reads bucket, prefixes, region and role from `config.py` and launches `2.3.2-tune.py` as a SageMaker (SDK v3 `ModelTrainer`) job. Run it once per study. |
| `requirements.txt` | Packages installed in the training container. |

## How to run

```
python 2.3.1-launch_sagemaker.py --model Attorney   --dry-run        # print the job config, launch nothing
python 2.3.1-launch_sagemaker.py --model Attorney
python 2.3.1-launch_sagemaker.py --model Litigation
python 2.3.1-launch_sagemaker.py --model Severity --primary-metric gini
python 2.3.1-launch_sagemaker.py --model Severity --primary-metric qwk
```

The training job only receives this folder, so it can't import `config.py`. `2.3.1-launch_sagemaker.py` passes
`S3_BUCKET`, `CUTOFF_DATASET_PREFIX`, `CUTOFF_RUNS_PREFIX` and `AWS_REGION` to `2.3.2-tune.py` as hyperparameters.
Set `SAGEMAKER_ROLE_ARN` in `config.py` first.

## Pseudocode (`2.3.2-tune.py` + `engine.py`)

```
pack train / val / holdout into float16 arrays (N x 8 x 1536); find each coverage's last snapshot
for trial in 1..50 (TPE, seed 42):
    sample lr, dropout, weight decay, noise, batch, fusion dim, 8 per-summary dims
    for epoch in 1..60:
        train one epoch (input + Gaussian noise)
        score train / val / holdout: Gini (+ QWK, F1 for severity), row-level and last snapshot
        report val metric to Optuna -> prune if below the median of earlier trials (after 5 warm-up epochs)
        keep the epoch with the best val metric; stop after 8 epochs without gain
    save the checkpoint + trial JSON to S3
best trial = highest val metric -> best/model.pt, _SUMMARY.json
```

## Search space

| Parameter | Range |
|---|---|
| Learning rate | 1e-4 – 5e-3 (log) |
| Dropout | 0 – 0.7 |
| Weight decay | 1e-6 – 3e-2 (log) |
| Input noise std | 0 – 0.05 |
| Batch size | 32, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768 |
| Fusion dim | 128, 256, 384, 512 |
| Encoder dim per summary (8 separate choices) | 64, 96, 128, 192, 256 |

Fixed: at most 60 epochs, patience 8, AdamW, cosine schedule, seed 42.

## Metrics

- **Binary Gini** = 2·AUC − 1.
- **Ordinal severity Gini:** the bands are ordered, so the 5-class problem becomes 4 binary ones: band ≥ 1, ≥ 2,
  ≥ 3 and ≥ 4. Each is scored by P(band ≥ k), which is the sum of the softmax probabilities from k up, and the
  4 Ginis are averaged with weights min(positives, negatives), so the sparse extreme thresholds don't dominate.
- **QWK** (quadratic-weighted kappa) between the argmax band and the true band. Predicting a band that's
  further off costs more.

## S3 output (under `CUTOFF_RUNS_PREFIX/<study>/`)

`checkpoints/trial_N.pt`, `trials/trial_N.json` (params, metrics, per-epoch history), `best/model.pt`, `study.db`,
`_SUMMARY.json`. The best checkpoints and summaries are copied into `../2.4-ModelArtifacts/`.
