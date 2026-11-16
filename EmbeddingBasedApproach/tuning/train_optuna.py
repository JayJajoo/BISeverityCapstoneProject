"""SageMaker training-job entry point: Optuna tuning of ClaimNet for one target.

Launched by launch_tuning.py (called from 05-Model-Training.ipynb, Step 4) on one ml.g5.4xlarge GPU.

Pseudocode
    load train / val / holdout arrays (X: rows x 8 x 1536 float16, y: 0/1, cvg: CLM_CVG_PK per row)
    for each of n_trials Optuna trials (TPE sampler, seeded):
        sample a config from the search space
        train ClaimNet on train, early-stop on validation Gini
        record val Gini (selection metric), holdout Gini (reporting only), last-snapshot Ginis, best epoch
        write the trial record to S3 as soon as it finishes (so a stopped job loses nothing)
    best trial = highest validation Gini; write _SUMMARY.json and all_trials.json to S3

S3 output (read by 05, Step 4b):
    {s3_prefix}/{study}/trials/trial_XXX.json
    {s3_prefix}/{study}/all_trials.json      list of completed trials
    {s3_prefix}/{study}/_SUMMARY.json        n_trials, elapsed_sec, best_trial, best_val_gini,
                                             best_holdout_gini, best_params
"""
import argparse
import json
import os
import time

import boto3
import numpy as np
import optuna
import torch

from claimnet import gini, last_snapshot_gini, predict, sample_cfg, train_one


def load_split(channel_dir):
    X = np.load(os.path.join(channel_dir, 'X.npy'))          # float16, (rows, 8, 1536), loaded fully into RAM
    X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)    # guard: ratio features can overflow float16
    y = np.load(os.path.join(channel_dir, 'y.npy')).astype(np.float32)
    cvg = np.load(os.path.join(channel_dir, 'cvg.npy'))
    return X, y, cvg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['Attorney', 'Litigation'])
    ap.add_argument('--study-name', required=True)             # e.g. attorney_v2, litigation_v1
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--max-epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--s3-bucket', required=True)
    ap.add_argument('--s3-prefix', required=True)              # e.g. adjuster-notes/embedding/runs
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s3 = boto3.client('s3')
    base = f'{args.s3_prefix}/{args.study_name}'

    def put_json(key, obj):
        s3.put_object(Bucket=args.s3_bucket, Key=f'{base}/{key}', Body=json.dumps(obj, indent=1).encode())

    data = {s: load_split(os.environ[f'SM_CHANNEL_{s.upper()}']) for s in ['train', 'val', 'holdout']}
    for s, (X, y, _) in data.items():
        print(f'[{args.target}] {s}: {len(y):,} rows, positive rate {y.mean():.3f}', flush=True)
    Xtr, ytr, _ = data['train']
    Xva, yva, cva = data['val']
    Xho, yho, cho = data['holdout']

    completed = []
    t0 = time.time()

    def objective(trial):
        cfg = sample_cfg(trial)
        ts = time.time()
        net, val_g, best_epoch = train_one(cfg, Xtr, ytr, Xva, yva, args.max_epochs, args.patience, device)
        p_va, p_ho = predict(net, Xva, device), predict(net, Xho, device)
        rec = {'trial': trial.number, 'val_gini': float(val_g), 'holdout_gini': float(gini(yho, p_ho)),
               'claim_gini': {'val': float(last_snapshot_gini(yva, p_va, cva)),
                              'holdout': float(last_snapshot_gini(yho, p_ho, cho))},
               'best_epoch': best_epoch, 'seconds': round(time.time() - ts, 1), 'params': trial.params}
        completed.append(rec)
        put_json(f'trials/trial_{trial.number:03d}.json', rec)
        print(f"trial {trial.number}: val Gini {rec['val_gini']:.4f} | epoch {best_epoch} | "
              f"{rec['seconds'] / 60:.1f} min", flush=True)
        del net
        torch.cuda.empty_cache()
        return val_g

    study = optuna.create_study(direction='maximize', study_name=args.study_name,
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    # a trial that fails (e.g. GPU out of memory on the largest batch / dims) is recorded as failed and skipped
    study.optimize(objective, n_trials=args.n_trials, catch=(RuntimeError, ValueError))

    best = max(completed, key=lambda r: r['val_gini'])           # selection on validation Gini only
    put_json('all_trials.json', completed)
    put_json('_SUMMARY.json', {
        'target': args.target, 'n_trials': len(study.trials), 'n_completed': len(completed),
        'elapsed_sec': round(time.time() - t0, 1), 'best_trial': best['trial'],
        'best_val_gini': best['val_gini'], 'best_holdout_gini': best['holdout_gini'],
        'best_params': best['params']})
    print(f"DONE: {len(completed)}/{len(study.trials)} trials completed | best trial {best['trial']} "
          f"val Gini {best['val_gini']:.4f}", flush=True)


if __name__ == '__main__':
    main()
