"""
Launch the Optuna HPO (2.3.2-tune.py) for ONE training run as a SageMaker 1-GPU job
(SageMaker SDK v3 ModelTrainer). Run
this 4 times to train all 4: Attorney, Litigation, and Severity's 2 primary-
metric variants (gini, qwk), each as its own job on its own instance.

    python 2.3.1-launch_sagemaker.py --model Attorney   --instance-type ml.g5.4xlarge
    python 2.3.1-launch_sagemaker.py --model Litigation --instance-type ml.g5.4xlarge
    python 2.3.1-launch_sagemaker.py --model Severity --primary-metric gini --instance-type ml.g5.4xlarge
    python 2.3.1-launch_sagemaker.py --model Severity --primary-metric qwk  --instance-type ml.g5.4xlarge
    python 2.3.1-launch_sagemaker.py --model Attorney --dry-run
"""
import argparse
import os
import sys

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute
from sagemaker.core.shapes.shapes import StoppingCondition, OutputDataConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))   # repo root
from config import AWS_REGION as REGION, SAGEMAKER_ROLE_ARN as ROLE, S3_BUCKET as BUCKET  # noqa: E402
from config import CUTOFF_DATASET_PREFIX, CUTOFF_RUNS_PREFIX as RUNS_BASE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["Attorney", "Litigation", "Severity"])
    ap.add_argument("--primary-metric", default="gini", choices=["gini", "qwk"],
                    help="what val selection/pruning/early-stop watches; 'qwk' only valid for Severity")
    ap.add_argument("--study", default=None)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--instance-type", default="ml.g5.4xlarge")
    ap.add_argument("--volume-size", type=int, default=100, help="EBS GB")
    ap.add_argument("--max-run", type=int, default=0, help="job wall-clock cap (s); 0 = account max (5d)")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--pytorch-version", default="2.3")
    ap.add_argument("--role", default=ROLE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.model != "Severity" and args.primary_metric != "gini":
        raise SystemExit("--primary-metric qwk is only valid for --model Severity")

    default_study = (args.model.lower() if args.model != "Severity"
                     else f"severity_{args.primary_metric}")
    study_name = args.study or default_study
    image = image_uris.retrieve(framework="pytorch", region=REGION, version=args.pytorch_version,
                                image_scope="training", instance_type=args.instance_type, py_version="py311")
    run_prefix = f"{RUNS_BASE}/{study_name}"
    out_path = f"s3://{BUCKET}/{run_prefix}/_sm_job"

    source = SourceCode(source_dir=HERE, entry_script="2.3.2-tune.py", requirements="requirements.txt")
    compute = Compute(instance_type=args.instance_type, instance_count=1, volume_size_in_gb=args.volume_size)
    max_run = args.max_run or 432000
    stop = StoppingCondition(max_runtime_in_seconds=max_run)
    hp = {"model": args.model, "primary-metric": args.primary_metric, "study": study_name,
          "n-trials": args.n_trials, "max-epochs": args.max_epochs, "patience": args.patience,
          "local-data-dir": "/opt/ml/scratch", "device": "auto", "in-ram": 1,
          "bucket": BUCKET, "dataset-prefix": CUTOFF_DATASET_PREFIX, "runs-base": RUNS_BASE, "region": REGION}

    print("=== SageMaker v3 HPO job config ===")
    print(f"  model     : {args.model} (primary metric: {args.primary_metric})")
    print(f"  image     : {image}")
    cap_txt = "NO CAP (5d account max)" if not args.max_run else f"{args.max_run/3600:.0f}h"
    print(f"  instance  : {args.instance_type} x1 | EBS {args.volume_size} GB | time cap {cap_txt}")
    print(f"  study     : {study_name} | n_trials {args.n_trials} | epochs<= {args.max_epochs} patience {args.patience}")
    print(f"  runs ->   : s3://{BUCKET}/{run_prefix}/")
    print(f"  hp        : {hp}")
    if args.dry_run:
        print("dry-run: validated config, not launching.")
        return

    sess = Session(boto_session=boto3.Session(region_name=REGION),
                   default_bucket=BUCKET,
                   default_bucket_prefix=f"{run_prefix}/_sm_code")
    trainer = ModelTrainer(
        training_image=image,
        role=args.role,
        source_code=source,
        compute=compute,
        stopping_condition=stop,
        output_data_config=OutputDataConfig(s3_output_path=out_path),
        hyperparameters=hp,
        environment={"PYTHONUNBUFFERED": "1"},
        base_job_name=f"{args.model.lower()}-hpo-{study_name}",
        sagemaker_session=sess,
    )
    trainer.train(wait=False)
    print("launched. checkpoints/study/summary will appear under the runs/ path as trials complete.")


if __name__ == "__main__":
    main()
