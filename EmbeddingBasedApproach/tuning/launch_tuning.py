"""Upload one target's arrays to S3 and launch the Optuna tuning job on SageMaker.

Called from 05-Model-Training.ipynb, Step 4, after Steps 1-2 have built att_data / lit_data:

    from tuning.launch_tuning import upload_arrays, launch
    launch('Attorney', 'attorney', upload_arrays(att_data, 'Attorney'))
    launch('Litigation', 'litigation', upload_arrays(lit_data, 'Litigation'))
"""
import os
import sys

import boto3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
from config import AWS_REGION, EMBEDDING_PREFIX, S3_BUCKET, SAGEMAKER_ROLE_ARN, TMP_DIR  # noqa: E402

TUNING_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE = 'ml.g5.4xlarge'        # 1 x A10G 24 GB GPU, 64 GB RAM
MAX_RUN_HOURS = 24


def upload_arrays(data, target):
    """data: {'train'|'val'|'holdout': (X, y, cvg, day)} as built by to_arrays() in 05.
    Writes X.npy / y.npy / cvg.npy per split to S3 and returns the SageMaker input channels."""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    local = TMP_DIR / 'tuning_data' / target
    channels = {}
    for split, (X, y, cvg, _day) in data.items():
        os.makedirs(local / split, exist_ok=True)
        prefix = f'{EMBEDDING_PREFIX}/tuning_data/{target}/{split}'
        for name, arr in [('X', X.astype(np.float16)), ('y', y.astype(np.int8)), ('cvg', cvg)]:
            path = local / split / f'{name}.npy'
            np.save(path, arr)
            s3.upload_file(str(path), S3_BUCKET, f'{prefix}/{name}.npy')
            os.remove(path)
        channels[split] = f's3://{S3_BUCKET}/{prefix}/'
        print(f'[{target}] {split}: {len(y):,} rows -> {channels[split]}')
    return channels


def launch(target, study_name, channels, n_trials=50, wait=False):
    """Start the training job that runs train_optuna.py. Results land in {EMBEDDING_PREFIX}/runs/{study_name}/."""
    from sagemaker.pytorch import PyTorch
    import sagemaker

    est = PyTorch(
        entry_point='train_optuna.py',
        source_dir=TUNING_DIR,                       # ships claimnet.py and requirements.txt (optuna) with it
        role=SAGEMAKER_ROLE_ARN,
        framework_version='2.6.0',
        py_version='py312',
        instance_type=INSTANCE,
        instance_count=1,
        max_run=MAX_RUN_HOURS * 3600,
        base_job_name=f'claimnet-optuna-{target.lower()}',
        sagemaker_session=sagemaker.Session(boto3.Session(region_name=AWS_REGION)),
        hyperparameters={'target': target, 'study-name': study_name, 'n-trials': n_trials,
                         'max-epochs': 60, 'patience': 8, 'seed': 42,
                         's3-bucket': S3_BUCKET, 's3-prefix': f'{EMBEDDING_PREFIX}/runs'},
    )
    est.fit(channels, wait=wait)
    print(f'launched {est.latest_training_job.name} ({INSTANCE}); '
          f'results -> s3://{S3_BUCKET}/{EMBEDDING_PREFIX}/runs/{study_name}/')
    return est
