# Strategy A: Note-Embedding Approach

Embed every adjuster note **once** with a pretrained text-embedding model. Build each claim's running history
from those note vectors, then train a neural network to predict **attorney involvement** and **litigation**
at each snapshot cutoff defined in `../02-Dataset-Development.ipynb`.

**Why this approach:** a note's embedding does not depend on the target or the cutoff, so 4.65M notes are
embedded once and reused by every model and every snapshot. Running summaries (mean, max, change from first to
latest) capture both what the claim looks like now and how it has changed, without retraining a language model.

```
BI_Notes CSVs ──03──▶ notes_XXX.jsonl (20 shards) ──04──▶ 1,536-dim vectors per note ──05──▶ running summaries
                                                                                           │
                       02 snapshot rows (CLM_CVG_PK, N_NOTES, TARGET, split) ──────────────┘──▶ ClaimNet ──▶ Gini
```

Run in order: **03 → 04 → 05**. Notebook `02` must already have been run.

---

## `03-Embedding-Dataset-Formation.ipynb`: prepare notes for embedding

- Loads all 16 `BI_Notes` batches (4,645,819 notes, 138,233 claims).
- **Light cleaning only:** HTML tags removed, `<br>` turned into newlines, entities unescaped, repeated spaces
  collapsed. Shorthand, casing and typos are **kept on purpose**, because that is what the model must learn to read.
- One record per note: `{note_id, claim_num, date, text}`. `note_id = <claim>_<row seq>` keeps the original
  order for notes with the same date.
- **No split and no cutoff here.** Those are applied later in `05` from `02`'s rows, so both strategies share
  exactly the same split.
- Writes 20 round-robin JSON-Lines shards (about 232K notes each) so the endpoint load is even.
- **Output:** `{S3_PREFIX}/input/notes_000.jsonl … notes_019.jsonl`

## `04-Embedding-Inference.ipynb`: embed every note on SageMaker

- **Model:** `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, with 1.5B parameters, 1,536-dim L2-normalized output and
  an 8,192-token cap. It was chosen for its long context: a single note is short, but the cap means long notes
  are not truncated. Batches are padded only to the longest note in the batch, so short notes stay cheap.
- Writes a custom `code/inference.py` handler and a pinned `code/requirements.txt`, then packages them as
  `model.tar.gz`. The pins (`transformers==4.51.3`, `sentence-transformers==5.7.0`) keep the AWS Hugging Face
  container from breaking on import.
- Deploys a real-time endpoint on **4 × ml.g5.xlarge** (A10G) and streams each shard with 16 parallel requests
  of 64 notes each.
- **Cost guard:** a running cost estimate stops the job at **$100**, and the endpoint is **always deleted** in a
  `finally` block. Shards already in S3 are skipped, so the run can resume after a stop.
- Sanity check: output record has 1,536 dimensions with L2 norm 1.0.
- **Output:** `{S3_PREFIX}/embeddings/notes_XXX.jsonl.out` with `{note_id, claim_num, date, embedding[1536]}` per line.

## `05-Model-Training.ipynb`: features, network, tuning and evaluation

1. **Running features (CPU).** Notes are sorted per claim by `(date, seq)`. For every note position *n* it
   computes 8 element-wise summaries over notes 1..n, each 1,536-dim:
   `mean, max, min, sum, max÷min, max−min, first÷last, first−last` (division is guarded against zero).
2. **Snapshot selection.** Loads `02`'s snapshot keys (`CLM_CVG_PK, CLM_NUM, SNAPSHOT_NUM, SNAPSHOT_DAY, N_NOTES`,
   target, split) and takes each claim's summaries as of `N_NOTES`. Training rows, targets and split are therefore
   identical to Strategy B: Attorney 466,090 rows, Litigation 388,956, Severity 619,029.
3. **Network (`ClaimNet`).** One encoder per summary (Linear → LayerNorm → GELU → Dropout), outputs concatenated,
   then a fusion layer, an MLP head and one logit.
4. **Tuning.** Optuna, 50 trials per target, run as a SageMaker training job on `ml.g5.4xlarge`. Search space:
   learning rate, dropout, weight decay, input-noise std, batch size, fusion dim and per-summary dim. Fixed
   settings: `BCEWithLogits` with `pos_weight = neg/pos` for class imbalance, AdamW, cosine LR schedule, up to
   60 epochs, early stopping after 8 epochs without validation-Gini gain. **The best trial is picked on validation
   Gini only.** Step 4a launches the job (code in `tuning/`, below), step 4b lists every trial, and step 4c
   retrains the winner with `seed=42`.
5. **Evaluation.** Gini (= 2·AUC − 1) on train, val and holdout, both row-level (every snapshot) and on each
   coverage's last snapshot.
6. **Gini by cutoff day** on the holdout: how early a useful signal appears.

### Results (holdout)

| Target | Best val Gini | Holdout row-level | Holdout last snapshot |
|---|---|---|---|
| Attorney | 0.500 | 0.461 | 0.818 |
| Litigation | 0.449 | 0.465 | 0.762 |

Attorney holdout Gini by cutoff day: 0.31 (day 1) → 0.39 (day 30) → 0.45 (day 90).

All scores are on the coverage-level snapshot rows and split from `02`. Severity is not trained in this notebook.

## `tuning/`: Optuna tuning job

| File | What it does |
|---|---|
| `claimnet.py` | `ClaimNet`, `train_one`, `predict`, the Gini metric and the Optuna search space. These are the same definitions as Steps 3–4 of `05`, shared with the job. |
| `train_optuna.py` | Job entry point. Loads the train, val and holdout arrays and runs 50 TPE trials (seed 42). Failed trials, such as GPU out-of-memory, are skipped. Each trial is saved to S3 as soon as it finishes, so a stopped job loses nothing. The best trial is picked on **validation Gini**, then `all_trials.json` and `_SUMMARY.json` are written. |
| `launch_tuning.py` | `upload_arrays()` saves `X.npy` / `y.npy` / `cvg.npy` per split to S3. `launch()` starts a SageMaker PyTorch 2.6 job on `ml.g5.4xlarge` (24 h cap). |
| `requirements.txt` | Extra packages installed in the training container (`optuna`, `scikit-learn`, `pandas`). |

Run it from `05`, Step 4a:

```python
from tuning.launch_tuning import upload_arrays, launch
launch('Attorney', 'attorney_v2', upload_arrays(att_data, 'Attorney'))
launch('Litigation', 'litigation_v1', upload_arrays(lit_data, 'Litigation'))
```

---

### Outputs (all under `EMBEDDING_PREFIX` from `config.py`)

| Path | Written by | Content |
|---|---|---|
| `input/notes_XXX.jsonl` | 03 | cleaned notes, 20 shards |
| `model/model.tar.gz` | 04 | inference handler and pinned requirements |
| `embeddings/notes_XXX.jsonl.out` | 04 | one 1,536-dim vector per note |
| `tuning_data/<target>/<split>/*.npy` | `tuning/launch_tuning.py` | arrays the tuning job trains on |
| `runs/<study>/trials/`, `all_trials.json`, `_SUMMARY.json` | `tuning/train_optuna.py` | every trial, plus the best parameters |
