# Predicting Bodily Injury (BI) Severity, Litigation Risk & Attorney Involvement from Adjuster Notes

**Author:** Jay Jajoo · Northeastern University capstone (Fall 2026)
**Sponsor:** a New England auto insurer (data under NDA; not included in this repository)

---

## 1. The problem

When an auto accident causes an injury, the insurer has to judge early how expensive the claim will be and
whether it will turn into a lawsuit. Today that judgment comes from an adjuster reading free-text claim notes
by hand. This project predicts three outcomes **from the adjuster notes alone, as they stood on a given day**:

| # | Target | Question | Type | Source column |
|---|---|---|---|---|
| 1 | **Attorney** | Will the claimant hire a lawyer? | binary | `ATTNY_FLG` |
| 2 | **Litigation** | Will the claim become a lawsuit instead of a negotiated settlement? | binary | `LITIG_FLG` |
| 3 | **Severity** | Which cost band will the claim close in? `$0–2.5K`, `$2.5–10K`, `$10–25K`, `$25–100K`, `$100K+` | 5-class ordinal | `SEV_BUCKET` (0–4) |

Early, accurate flags let an insurer fast-track low-risk claims and send high-risk ones to legal review sooner.

**What this project adds.** Prior work uses either note embeddings (Fursov et al., 2019) or LLM-extracted
structure (Li et al., 2025) for insurance claims. This project compares **both strategies head-to-head on
the same rows, targets and split**, for attorney involvement, litigation and BI severity.

---

## 2. Data

| Item | Size |
|---|---|
| Adjuster notes (`BI_Notes`, 16 CSV batches) | **4,645,819 notes** across **138,233 claims** |
| BI coverages, claim years 2015–2022 (`BI-Coverage-ALL-2015_2022.parquet`) | 169,322 coverages (167,726 closed) |
| Endpoint tables (`Endpoint-Attorney`, `Endpoint-Litigation`, `Endpoint-Severity`) | one row per coverage (`CLM_CVG_PK`) |

- **Grain:** notes are keyed by claim (`CLAIM_NUM`); targets are keyed by coverage (`CLM_CVG_PK`). One claim can
  have several coverages (one per injured party), and they share the same notes.
- **Notes:** adjuster shorthand (`COVG`, `LIAB`, `RES`, `POA`, ...) mixed with free text and HTML. Median note is
  46 words, but a claim's full history is long (median 26 notes per claim, about 1.4 years from first to last note).
- **Base rates (coverage level):** attorney about 44%, litigation about 7%, severity buckets about
  60% / 16% / 12% / 8% / 3%.

> **Data is confidential.** No raw data, notes or outputs are in this repo. All paths in the notebooks
> come from `config.py` and are **placeholders**; set them to your own locations before running. Every claim, person and firm name shown in notebook outputs is redacted.

---

## 3. Overall approach

```mermaid
flowchart LR
    A[Raw adjuster notes<br/>+ target tables] --> B[01 EDA]
    A --> C[02 Snapshot dataset<br/>leakage-safe cutoffs<br/>70/15/15 split by claim]
    C --> D[Strategy A: Embeddings<br/>03 → 04 → 05]
    C --> E[Strategy B: LLM features<br/>06]
    D --> F[Evaluation on the same<br/>holdout: Gini overall<br/>and by cutoff day]
    E --> F
```

### 3.1 Framing: predict at a point in time, never after the event

A model is only useful if it predicts *before* the attorney is hired, the suit is filed or the claim closes.
So every training row is a **snapshot**: the claim's notes as they stood on day *d* after the claim opened.

- **Day grids** (days after `CLM_OPEN_DT`), denser early because attorney involvement happens fast
  (median 31 days) and litigation slowly (median 666 days):
  - Attorney: `1, 7, 15, 30, 60, 90, 180, 365, 730, 1095`
  - Litigation: `1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460`
  - Severity: `1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460, 1825`
- **Boundary per coverage:** the event date if the event happened (`ATTNY_DT` / `LITIG_DT`), otherwise the
  coverage close date; for severity, `LAST_CLOSE_DT`. Snapshots stop once a cutoff reaches the boundary.
- **Leakage margin:** only notes strictly before the boundary are kept. If no note falls exactly on the
  boundary, the last kept note is also dropped **if it mentions the event** (keyword check, e.g. "LOR",
  "attorney", "summons", "settlement"), since the event date in the system often lags the note.
- Snapshots with no new notes, and snapshots with 0 notes, are skipped.

### 3.2 Train / validation / holdout split

One **global, claim-level, stratified 70/15/15 split**, shared by all three targets and both strategies.
Stratified on `ORG_GROUP_NAME` + `POL_STATE` + `LOSS_YEAR` + `ATTNY_FLG` + `LITIG_FLG` + `SEV_BUCKET`.
Splitting by claim (not by coverage or snapshot) keeps shared note text from leaking across splits.

| Split | Claims | Attorney rows | Litigation rows | Severity rows |
|---|---|---|---|---|
| train / val / holdout | 81,212 / 17,418 / 17,267 | 466,090 total | 388,956 total | 619,029 total |

### 3.3 Two feature strategies

| | **A. Note embeddings** (`EmbeddingBasedApproach/`) | **B. LLM-extracted features** (`ManualFeatureBasedApproach/`) |
|---|---|---|
| Idea | Embed each note once, then summarize the running history numerically | Have an LLM read the snapshot like a senior analyst and answer fixed questions |
| Model | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (1,536-dim) on SageMaker | Claude Opus 5.5 through the Claude Agent SDK, structured JSON output |
| Features | 8 running summaries per claim (mean, max, min, sum, max÷min, max−min, first÷last, first−last) | 171 interpretable features (A01–A59 attorney, L01–L69 litigation, S01–S43 severity), each with an evidence quote |
| Downstream model | Multi-branch neural net (one encoder per summary → fusion → MLP), tuned with Optuna | To be trained on the extracted features (pilot stage) |
| Strengths | Cheap per note; scales to 4.65M notes | Interpretable, auditable features; domain-aware |

### 3.4 Evaluation

- **Primary metric: Gini = 2·AUC − 1**, the standard ranking metric in insurance. It is threshold-free and
  robust to class imbalance (litigation is about 7% positive), and it matches how the sponsor uses scores: to
  rank claims for triage.
- **Gini by cutoff day** on the holdout: shows how early a useful signal appears, which is the business question.
- **Row-level Gini** (every snapshot) and **last-snapshot Gini** (each coverage's latest snapshot).
- **Holdout is used once**, for final reporting. Model selection uses validation Gini only.
- For strategy B, also: **schema validity** (every value checked exactly against its allowed list),
  **run-to-run stability** (share of identical values across two runs), and **cost per snapshot**.

---

## 4. Results so far

**Strategy A, embeddings (holdout Gini):**

| Target | Row-level (all snapshots) | Last snapshot per coverage |
|---|---|---|
| Attorney | 0.461 | 0.818 |
| Litigation | 0.465 | 0.762 |

Holdout Gini for attorney rises with the cutoff day (0.31 at day 1 → 0.45 by day 90): the notes carry more
signal as the claim develops. Severity on embeddings is not trained yet.

All scores are on the coverage-level snapshot rows and split from `02`.

**Strategy B, LLM features (100-snapshot pilot from train):** 100 of 100 snapshots returned valid outputs on
the first try. Two independent runs agreed on **97.8%** of all feature values (none of the 171 features below
90%). Cost was **$0.34 per snapshot** ($34 for 100).

---

## 5. Repository layout

```
.
├── README.md                          ← this file
├── config.py                          ← all paths and storage settings (edit before running)
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← keeps confidential data and outputs out of git
├── 01-EDA_Cleaned.ipynb               ← exploratory data analysis
├── 02-Dataset-Development.ipynb       ← leakage-safe snapshot datasets + global split (feeds both strategies)
├── EmbeddingBasedApproach/            ← Strategy A: note embeddings → neural net
│   ├── README.md
│   ├── 03-Embedding-Dataset-Formation.ipynb
│   ├── 04-Embedding-Inference.ipynb
│   ├── 05-Model-Training.ipynb
│   └── tuning/                        ← Optuna tuning job (SageMaker) launched from 05
└── ManualFeatureBasedApproach/        ← Strategy B: LLM-extracted features
    ├── README.md
    └── 06-PromptFeatures.ipynb
```

### Top-level notebooks

**`01-EDA_Cleaned.ipynb`: exploratory data analysis**
- Two worked example claims (one with attorney and litigation), printed note by note with a plain-English
  walkthrough, to show what the notes contain and where the attorney and litigation dates fall.
- Corpus-wide statistics over all 4.65M notes: words and characters per note, notes per claim, claim duration,
  gaps between notes, note volume per year, top words and a word cloud.
- Target analysis: attorney and litigation rate and severity mix by claim-open year, by severity bucket, by
  `ORG_GROUP_NAME` (Direct vs. Independent) and by loss state.
- Time to event: days from claim open to attorney, to litigation and to close. These results set the day grids
  used in `02`.

**`02-Dataset-Development.ipynb`: snapshot dataset construction**
- Starts with high-level pseudocode, then implements the rules in §3.1: per-model boundaries, keyword-checked margin rule, day grids, dedup and 0-note skip.
- Unit check of the margin rule, a test on the EDA sample claim and on 12 edge cases (sibling coverages with
  different outcomes, long attorney and litigation lags, events that never happened).
- Global claim-level stratified split (§3.2).
- Full build: caches all notes once, then runs the three models in parallel processes and writes parquet batches
  to `{S3_PREFIX}/{Attorney|Litigation|Severity}/{train|val|holdout}/`.
- Output schema: `CLM_CVG_PK, CLM_NUM, SNAPSHOT_NUM, SNAPSHOT_DAY, N_NOTES, NOTE_TEXT, TARGET` (or
  `TARGET_BUCKET` for severity).

---

## 6. How to reproduce

**Run order:** `01` (optional) → `02` → **either** `03 → 04 → 05` (Strategy A) **or** `06` (Strategy B).
`02` must run first; both strategies read its snapshot rows and split.

**Environment**
- Python 3.12; `pandas`, `numpy`, `pyarrow`, `matplotlib`, `wordcloud`, `boto3`, `scikit-learn`
- Strategy A: `torch`, `optuna`, `sagemaker`. The endpoint pins `transformers==4.51.3` and `sentence-transformers==5.7.0`
- Strategy B: `claude-agent-sdk==0.2.180` and a logged-in Claude Code install
- Compute used: `02` on a 16-core, 54 GB RAM machine (about 15 min for all three models); `04` on 4 × `ml.g5.xlarge`
  (budget cap $100); `05` tuning on `ml.g5.4xlarge`

**Before running:** edit **`config.py`** (the only file with paths): local data folders, S3 bucket and prefixes,
AWS region and SageMaker role ARN. Every notebook imports its paths from it. Install packages with
`pip install -r requirements.txt`.

**Determinism:** all sampling and splitting uses `seed=42` (`numpy.random.default_rng`, `torch.manual_seed`).

---

## 7. Attribution

All code, dataset design, prompts and analysis in this repository are my own work. External components are
used as tools and are credited here:

- Embedding model: `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (Hugging Face), served with the AWS Hugging Face DLC.
- Feature extraction: Claude Opus 5.5 (Anthropic) through the Claude Agent SDK.
- Libraries: pandas, NumPy, PyTorch, Optuna, scikit-learn, sentence-transformers, boto3, SageMaker SDK.

**References**
- Fursov, I. et al. (2019). *Sequence embeddings help to identify fraudulent cases in healthcare insurance.* arXiv:1910.03072.
- Li, et al. (2025). Textual analysis of insurance claims with large language models. *Journal of Risk and Insurance*, 92, 505–537.
