"""
Optuna hyperparameter tuning, shared across 4 training runs:
  --model Attorney    (binary, primary metric fixed to gini)
  --model Litigation  (binary, primary metric fixed to gini)
  --model Severity --primary-metric gini  (5-class, support-weighted ordinal Gini)
  --model Severity --primary-metric qwk   (5-class, quadratic-weighted-kappa)
The two Severity variants share the exact same data/architecture/search space
-- they differ ONLY in what the validation split selects/prunes/early-stops
on. Both gini and qwk are always computed and logged every epoch regardless.

Tuned: lr, dropout, weight_decay, noise_std, batch size, fusion dim, the 8
per-group dims. Fixed: max_epochs, early-stopping patience, optimizer
(AdamW), split (global, precomputed by 2.1.2-assign_global_splits.py -- NOT
recomputed here).

Each trial's best checkpoint + a trial JSON (params/metrics/full per-epoch
train/val/test history) are written to S3; the Optuna sqlite study is synced
to S3 so the search is resumable. ALL trials' checkpoints are saved to S3
(not just the best), so any trial's model can be reused later.

Data: the 3 (train/val/holdout) parquet folders are streamed once into local
float16 memmaps (bounded RAM) and reused by every trial.

    # quick local check on CPU (2 trials x 3 epochs):
    python 2.3.2-tune.py --model Attorney --smoke --bucket <S3_BUCKET> --dataset-prefix <CUTOFF_DATASET_PREFIX> --runs-base <CUTOFF_RUNS_PREFIX>
    # full (on a GPU box):
    python 2.3.2-tune.py --model Severity --primary-metric qwk --study severity_qwk --n-trials 50 --local-data-dir /opt/ml/scratch \
        --bucket <S3_BUCKET> --dataset-prefix <CUTOFF_DATASET_PREFIX> --runs-base <CUTOFF_RUNS_PREFIX>

Bucket, prefixes and region come from config.py at the repo root. 2.3.1-launch_sagemaker.py reads them
and passes them in as hyperparameters, because the training job only receives this folder.
"""
import argparse
import json
import os
import time

import boto3
import numpy as np
import torch
import optuna

import data as D
from engine import run_training

GROUP_DIM_CHOICES = [64, 96, 128, 192, 256]
FUSE_CHOICES = [128, 256, 384, 512]
BATCH_CHOICES = [32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
NUM_CLASSES = {"Attorney": 1, "Litigation": 1, "Severity": 5}


def json_safe(o):
    # NaN / inf are not valid JSON (e.g. holdout_qwk for the binary studies): write them as null
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def s3_put_bytes(s3, key, b):
    s3.put_object(Bucket=D.BUCKET, Key=key, Body=b)


def s3_upload(s3, local, key):
    s3.upload_file(local, D.BUCKET, key)


def sample_cfg(trial, max_epochs, patience, primary_metric):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.7),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.05),
        "batch": trial.suggest_categorical("batch", BATCH_CHOICES),
        "fuse_dim": trial.suggest_categorical("fuse_dim", FUSE_CHOICES),
        "group_dims": {g: trial.suggest_categorical(f"dim_{g}", GROUP_DIM_CHOICES) for g in D.GROUPS},
        "cosine": True, "primary_metric": primary_metric,
        "max_epochs": max_epochs, "patience": patience, "seed": 42,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["Attorney", "Litigation", "Severity"])
    ap.add_argument("--primary-metric", default="gini", choices=["gini", "qwk"],
                    help="what val selection/pruning/early-stop watches; 'qwk' only valid for Severity")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--study", default=None)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--local-data-dir", default=None)
    ap.add_argument("--in-ram", type=int, default=1)
    ap.add_argument("--bucket", required=True, help="S3_BUCKET in config.py")
    ap.add_argument("--dataset-prefix", required=True, help="CUTOFF_DATASET_PREFIX in config.py")
    ap.add_argument("--runs-base", required=True, help="CUTOFF_RUNS_PREFIX in config.py")
    ap.add_argument("--region", default="us-east-1", help="AWS_REGION in config.py")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.model != "Severity" and args.primary_metric != "gini":
        raise SystemExit("--primary-metric qwk is only valid for --model Severity")

    default_study = (args.model.lower() if args.model != "Severity"
                     else f"severity_{args.primary_metric}")
    study_name = args.study or default_study
    local_dir = args.local_data_dir or f"/tmp/{args.model.lower()}mm"
    if args.smoke:
        args.n_trials = 2; args.max_epochs = 3; args.patience = 2
        args.device = "cpu"; study_name = f"{args.model.lower()}_{args.primary_metric}_smoke"

    num_classes = NUM_CLASSES[args.model]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    D.BUCKET, D.DS_PREFIX = args.bucket, args.dataset_prefix
    s3 = boto3.client("s3", region_name=args.region)
    run_prefix = f"{args.runs_base}/{study_name}"
    t0 = time.time()

    # ---- data: pack each split (already globally decided) into its own memmap ----
    print(f"model={args.model} | device={device} | study={study_name} | packing -> {local_dir}", flush=True)
    Ns, metas, Xs = {}, {}, {}
    for split in ["train", "val", "holdout"]:
        N, _, _ = D.build_memmap(args.model, split, local_dir, region=args.region)
        y, cvg, snum, _ = D.load_meta(local_dir, split)
        X = D.open_memmap(local_dir, split, N)
        if args.in_ram:
            t_r = time.time()
            X = np.ascontiguousarray(X)
            print(f"  [{split}] loaded into RAM: {X.nbytes/1e9:.1f} GB ({time.time()-t_r:.0f}s)", flush=True)
        Ns[split] = N; metas[split] = (y, cvg, snum); Xs[split] = X

    # concat train/val/holdout into one array with offset index ranges (keeps run_training's
    # single-X-array interface; each split's rows are a contiguous slice)
    offsets = np.cumsum([0, Ns["train"], Ns["val"], Ns["holdout"]])
    X = np.concatenate([Xs["train"], Xs["val"], Xs["holdout"]], axis=0)
    y = np.concatenate([metas[s][0] for s in ["train", "val", "holdout"]])
    cvg = np.concatenate([metas[s][1] for s in ["train", "val", "holdout"]])
    snum = np.concatenate([metas[s][2] for s in ["train", "val", "holdout"]])
    tr = np.arange(offsets[0], offsets[1])
    va = np.arange(offsets[1], offsets[2])
    ho = np.arange(offsets[2], offsets[3])

    is_last = np.zeros(len(y), dtype=bool)
    for split, idx in (("train", tr), ("val", va), ("ho", ho)):
        local_last = D.last_snapshot_indices(cvg[idx], snum[idx])
        is_last[idx[local_last]] = True
    last_idx = {"train": tr[is_last[tr]], "val": va[is_last[va]], "ho": ho[is_last[ho]]}
    print(f"packed tr/va/ho={len(tr)}/{len(va)}/{len(ho)} | "
          f"cvgs(last-snapshot) tr/va/ho={len(last_idx['train'])}/{len(last_idx['val'])}/{len(last_idx['ho'])} "
          f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("/tmp/optuna", exist_ok=True)
    db_local = f"/tmp/optuna/{study_name}.db"
    db_key = f"{run_prefix}/study.db"
    try:
        s3.download_file(D.BUCKET, db_key, db_local); print("resumed study.db from S3", flush=True)
    except Exception:
        pass
    study = optuna.create_study(
        direction="maximize", study_name=study_name,
        storage=f"sqlite:///{db_local}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))

    best_metric_global = [max([t.value for t in study.trials if t.value is not None], default=-2.0)]

    def objective(trial):
        cfg = sample_cfg(trial, args.max_epochs, args.patience, args.primary_metric)
        print(f"\n=== trial {trial.number} ({args.model}, primary={args.primary_metric}) === "
              f"{json.dumps({k: v for k, v in cfg.items() if k not in ('max_epochs', 'patience', 'seed', 'group_dims')})}",
              flush=True)
        res = run_training(X, y, tr, va, ho, cfg, device, D.GROUPS, log=print, trial=trial,
                            last_idx=last_idx, num_classes=num_classes)
        ho_gini = res["best_holdout_gini"]
        ho_qwk = res.get("best_holdout_qwk")
        cvg_g = res.get("best_cvg", {})
        dims = [cfg["group_dims"][g] for g in D.GROUPS]
        trial.set_user_attr("holdout_gini", ho_gini); trial.set_user_attr("holdout_qwk", ho_qwk)
        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("cvg_val_gini", cvg_g.get("val", {}).get("gini"))
        trial.set_user_attr("cvg_holdout_gini", cvg_g.get("ho", {}).get("gini"))

        ckpt = f"/tmp/optuna/trial_{trial.number}.pt"
        torch.save({"state_dict": res["best_state"], "groups": D.GROUPS, "group_dims": dims,
                    "fuse_dim": cfg["fuse_dim"], "num_classes": num_classes, "cfg": cfg,
                    "primary_metric": args.primary_metric, "val_metric": res["best_metric"],
                    "holdout_gini": ho_gini, "holdout_qwk": ho_qwk, "cvg_gini": cvg_g}, ckpt)
        s3_upload(s3, ckpt, f"{run_prefix}/checkpoints/trial_{trial.number}.pt")
        s3_put_bytes(s3, f"{run_prefix}/trials/trial_{trial.number}.json", json.dumps(json_safe(
            {"trial": trial.number, "model": args.model, "primary_metric": args.primary_metric,
             "params": trial.params, "val_metric": res["best_metric"], "holdout_gini": ho_gini,
             "holdout_qwk": ho_qwk, "cvg_gini": cvg_g, "best_epoch": res["best_epoch"],
             "history": res["history"]}), indent=2, default=str).encode())
        if res["best_metric"] > best_metric_global[0]:
            best_metric_global[0] = res["best_metric"]
            s3_upload(s3, ckpt, f"{run_prefix}/best/model.pt")
        os.remove(ckpt)
        print(f"trial {trial.number}: val_{args.primary_metric}={res['best_metric']:.4f} holdout_gini={ho_gini:.4f} "
              f"(best so far {best_metric_global[0]:.4f})", flush=True)
        return res["best_metric"]

    def sync_cb(study_, trial_):
        try: s3_upload(s3, db_local, db_key)
        except Exception as e: print("  (study sync)", e, flush=True)

    study.optimize(objective, n_trials=args.n_trials, timeout=(args.timeout or None), callbacks=[sync_cb])

    best = study.best_trial
    summary = {"model": args.model, "primary_metric": args.primary_metric, "study": study_name,
               "n_trials": len(study.trials), "best_trial": best.number, "best_val_metric": best.value,
               "best_holdout_gini": best.user_attrs.get("holdout_gini"),
               "best_holdout_qwk": best.user_attrs.get("holdout_qwk"),
               "best_params": best.params, "elapsed_sec": round(time.time() - t0, 0)}
    s3_put_bytes(s3, f"{run_prefix}/_SUMMARY.json", json.dumps(json_safe(summary), indent=2).encode())
    sync_cb(None, None)
    print("\nDONE:", json.dumps(summary, indent=2), flush=True)
    print(f"best model -> s3://{D.BUCKET}/{run_prefix}/best/model.pt", flush=True)


if __name__ == "__main__":
    main()
