# 2.4 Model artifacts

The saved outputs of steps 2.1 and 2.3: each study's best model, plus the manifests that record how the data and
the studies were built.

## `best_models/`

Each file is the best checkpoint of one study, saved as a PyTorch `torch.save` dict.

| File | Study | Best trial | Output |
|---|---|---|---|
| `attorney_best_model.pt` | Attorney | 42 | 1 logit; sigmoid gives P(attorney) |
| `litigation_best_model.pt` | Litigation | 24 | 1 logit; sigmoid gives P(lawsuit) |
| `severity_gini_best_model.pt` | Severity, selected on ordinal Gini | 32 | 5 logits; softmax gives P(band 0..4) |
| `severity_qwk_best_model.pt` | Severity, selected on QWK | 23 | 5 logits; softmax gives P(band 0..4) |

Keys: `state_dict`, `groups`, `group_dims`, `fuse_dim`, `num_classes`, `cfg` (all hyperparameters),
`primary_metric`, `val_metric`, `holdout_gini`, `holdout_qwk`, and `cvg_gini` (last-snapshot score per split).

```python
import sys, torch
sys.path.insert(0, '../2.3-ModelTraining')
from model import CutoffNet

ck = torch.load('best_models/litigation_best_model.pt', map_location='cpu', weights_only=False)
net = CutoffNet(ck['groups'], ck['group_dims'], fuse_dim=ck['fuse_dim'],
                dropout=ck['cfg']['dropout'], num_classes=ck['num_classes'])
net.load_state_dict(ck['state_dict']); net.eval()
p = torch.sigmoid(net(x))     # x: (batch, 8, 1536) float tensor, the 8 summaries at one cutoff
```

## `manifests/`

| File | Content |
|---|---|
| `_DATASET.json` | Dataset manifest written by `2.1.3-build_cutoff_dataset.py`: sources, boundary rule, margin rule, day grids, dedup rule, feature formulas, parquet schema, and per-target coverages, rows and label sums. S3 locations appear as `<S3_BUCKET>/<…_PREFIX>` placeholders that match `config.py`. |
| `_SPLITS.json` | Split manifest written by `2.1.2-assign_global_splits.py`: stratification keys, seed, fractions and claim counts (81,212 / 17,418 / 17,267). |
| `attorney_SUMMARY.json` | Best trial 42: val Gini 0.334, holdout Gini 0.326, best hyperparameters, run time. |
| `litigation_SUMMARY.json` | Best trial 24: val Gini 0.479, holdout Gini 0.488. |
| `severity_gini_SUMMARY.json` | Best trial 32: val Gini 0.563, holdout Gini 0.569, holdout QWK 0.402. |
| `severity_qwk_SUMMARY.json` | Best trial 23: val QWK 0.425, holdout Gini 0.546, holdout QWK 0.430. |

The other trial checkpoints, about 1.3 GB in total, are kept in S3 only (`CUTOFF_RUNS_PREFIX/<study>/checkpoints/`).
