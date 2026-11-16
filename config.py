"""Project-wide paths and storage settings.

Every notebook reads its locations from here, so this is the only file to edit before running.
The data is confidential (NDA) and is not in this repository: point these at your own copies.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Local inputs
# read by 01-EDA_Cleaned.ipynb, 02-Dataset-Development.ipynb, ManualFeatureBasedApproach/06-PromptFeatures.ipynb
TARGET_DIR = REPO_ROOT / 'TargetData'          # BI-Coverage-ALL + Endpoint-* parquet files
# read by 01-EDA_Cleaned.ipynb, 02-Dataset-Development.ipynb, EmbeddingBasedApproach/03-Embedding-Dataset-Formation.ipynb
NOTES_DIR = REPO_ROOT.parent / 'BI_Notes'      # batch_*.csv adjuster-note files

# Local scratch space for caches, batch files and logs
# used by 02-Dataset-Development.ipynb, EmbeddingBasedApproach/03-Embedding-Dataset-Formation.ipynb,
#         EmbeddingBasedApproach/04-Embedding-Inference.ipynb, EmbeddingBasedApproach/tuning/launch_tuning.py
TMP_DIR = Path('/tmp')

# S3 storage
S3_BUCKET = 'bi-severity-capstone'

# written by 02-Dataset-Development.ipynb,
# read by EmbeddingBasedApproach/05-Model-Training.ipynb and ManualFeatureBasedApproach/06-PromptFeatures.ipynb
SNAPSHOT_PREFIX = 'adjuster-notes/manual-feature-eng/datasets'

# written by EmbeddingBasedApproach/03-Embedding-Dataset-Formation.ipynb (input/)
#         EmbeddingBasedApproach/04-Embedding-Inference.ipynb (model/, embeddings/),
#         EmbeddingBasedApproach/tuning/launch_tuning.py (tuning_data/)
#     and EmbeddingBasedApproach/tuning/train_optuna.py (runs/, the Optuna tuning job),
# read by EmbeddingBasedApproach/05-Model-Training.ipynb (embeddings/, runs/)
EMBEDDING_PREFIX = 'adjuster-notes/embedding'

# SageMaker
# used by EmbeddingBasedApproach/04-Embedding-Inference.ipynb and the tuning job in EmbeddingBasedApproach/05-Model-Training.ipynb
AWS_REGION = 'us-east-1'
# Your SageMaker execution role, e.g. 'arn:aws:iam::<12-digit-account-id>:role/<role-name>'.
# This is the only place the AWS account number appears. AWS credentials themselves are NOT stored here:
# boto3 / SageMaker pick them up from your environment (`aws configure`, an AWS profile, or the notebook's role).
SAGEMAKER_ROLE_ARN = '<your-sagemaker-execution-role-arn>'

# Claude Code CLI used by the Agent SDK (ManualFeatureBasedApproach/06-PromptFeatures.ipynb).
# Uses its existing login, so no API key is stored here.
CLAUDE_CODE_PATH = str(Path.home() / '.local' / 'bin' / 'claude')
