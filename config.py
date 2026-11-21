"""Project-wide paths and storage settings.

Every notebook and script reads its locations from here, so this is the only file to edit before running.
The data is confidential (NDA) and is not in this repository: point these at your own copies.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Local inputs
# read by 1-EDA_Cleaned.ipynb, 3-ManualFeatureBasedApproach/3.1-Dataset-Development.ipynb, 3-ManualFeatureBasedApproach/3.2-PromptFeatures.ipynb
TARGET_DIR = REPO_ROOT / 'TargetData'          # BI-Coverage-ALL + Endpoint-* parquet files
# read by 1-EDA_Cleaned.ipynb, 3-ManualFeatureBasedApproach/3.1-Dataset-Development.ipynb, 2-EmbeddingBasedApproach/2.2-Embedding/2.2.1-Note-Preparation.ipynb,
#         2-EmbeddingBasedApproach/2.1-DatasetDevelopment/2.1.1-build_keyword_registry.py
NOTES_DIR = REPO_ROOT.parent / 'BI_Notes'      # batch_*.csv adjuster-note files

# Local scratch space for caches, batch files and logs
# used by 3-ManualFeatureBasedApproach/3.1-Dataset-Development.ipynb, 2-EmbeddingBasedApproach/2.2-Embedding/2.2.1-Note-Preparation.ipynb,
#         2-EmbeddingBasedApproach/2.2-Embedding/2.2.2-Embedding-Inference.ipynb,
#         2-EmbeddingBasedApproach/2.1-DatasetDevelopment/2.1.1-build_keyword_registry.py and 2.1.3-build_cutoff_dataset.py
TMP_DIR = Path('/tmp')

# S3 storage
S3_BUCKET = 'bi-severity-capstone'

# S3 copies of the target tables (the same files as TARGET_DIR), read by
# 2-EmbeddingBasedApproach/2.1-DatasetDevelopment/2.1.2-assign_global_splits.py and 2.1.3-build_cutoff_dataset.py
TARGET_DATA_PREFIX = 'target-data'
COVERAGE_KEY = f'{TARGET_DATA_PREFIX}/BI-Coverage-ALL-2015_2022.parquet'
SEVERITY_KEY = f'{TARGET_DATA_PREFIX}/Endpoint-Severity-2015_2022.parquet'

# written by 3-ManualFeatureBasedApproach/3.1-Dataset-Development.ipynb, read by 3-ManualFeatureBasedApproach/3.2-PromptFeatures.ipynb
SNAPSHOT_PREFIX = 'adjuster-notes/manual-feature-eng/datasets'

# written by 2-EmbeddingBasedApproach/2.2-Embedding/2.2.1-Note-Preparation.ipynb (input/)
#     and 2-EmbeddingBasedApproach/2.2-Embedding/2.2.2-Embedding-Inference.ipynb (model/, embeddings/),
# read by 2-EmbeddingBasedApproach/2.1-DatasetDevelopment/2.1.3-build_cutoff_dataset.py (embeddings/)
EMBEDDING_PREFIX = 'adjuster-notes/embedding'

# written by 2-EmbeddingBasedApproach/2.1-DatasetDevelopment/*.py
#   (_keyword_registry.parquet, _SPLITS.parquet, {model}/{split}/{feature}/part-*.parquet, _DATASET.json),
# read by 2-EmbeddingBasedApproach/2.3-ModelTraining/2.3.2-tune.py (passed to the job by 2.3.1-launch_sagemaker.py)
CUTOFF_DATASET_PREFIX = f'{EMBEDDING_PREFIX}/cutoff_dataset'
# written by 2-EmbeddingBasedApproach/2.3-ModelTraining/2.3.2-tune.py (the Optuna tuning jobs):
#   {study}/checkpoints/, trials/, best/model.pt, study.db, _SUMMARY.json
CUTOFF_RUNS_PREFIX = f'{EMBEDDING_PREFIX}/cutoff_runs'

# AWS / SageMaker
# used by 2-EmbeddingBasedApproach/2.2-Embedding/2.2.2-Embedding-Inference.ipynb, 2-EmbeddingBasedApproach/2.1-DatasetDevelopment/*.py
#     and 2-EmbeddingBasedApproach/2.3-ModelTraining/2.3.1-launch_sagemaker.py
AWS_REGION = 'us-east-1'
# Your SageMaker execution role, e.g. 'arn:aws:iam::<12-digit-account-id>:role/<role-name>'.
# This is the only place the AWS account number appears. AWS credentials themselves are NOT stored here:
# boto3 / SageMaker pick them up from your environment (`aws configure`, an AWS profile, or the notebook's role).
SAGEMAKER_ROLE_ARN = '<your-sagemaker-execution-role-arn>'

# Claude Code CLI used by the Agent SDK (3-ManualFeatureBasedApproach/3.2-PromptFeatures.ipynb).
# Uses its existing login, so no API key is stored here.
CLAUDE_CODE_PATH = str(Path.home() / '.local' / 'bin' / 'claude')

# Best-model predictions for every train / val / holdout row (contain claim numbers: confidential, git-ignored),
# read by 2-EmbeddingBasedApproach/2.5-Results/2.5.1-Results-Visualization.ipynb
PREDICTIONS_DIR = REPO_ROOT / '2-EmbeddingBasedApproach' / '2.5-Results' / 'predictions'
