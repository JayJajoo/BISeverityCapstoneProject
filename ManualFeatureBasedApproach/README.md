# Strategy B: LLM-Extracted Feature Approach

Instead of learning from raw vectors, have a large language model read a claim's snapshot the way a senior BI
claims analyst would, and answer a fixed set of **171 domain questions**. Each answer is one constrained value
plus an **evidence quote** from the notes. The answers become interpretable, auditable features for the three
models (attorney, litigation, severity).

**Why this approach:** adjusters already think in these terms (liability split, treatment type, attorney letter
received, reserve changes, and so on). Turning notes into those variables gives features a claims team can read
and challenge, and avoids the "black box" problem of embeddings.

## `06-PromptFeatures.ipynb`: feature-extraction pilot

| Section | What it does |
|---|---|
| **1. Setup** | Model `claude-opus-5-5` through the **Claude Agent SDK** (`claude-agent-sdk==0.2.180`), adaptive thinking at `high` effort, max $3 per call, 1 retry. |
| **2. Sample** | Builds a **100-snapshot stratified pilot from the TRAIN split only**, so the holdout stays untouched. Attorney 34 (17 yes / 17 no), Litigation 33 (17 / 16), Severity 33 (buckets 0–4: 7/7/7/6/6). Within each class, snapshots are spread over `ORG_GROUP_NAME × POL_STATE × LOSS_YEAR` by systematic sampling. One coverage per claim, and no claim is reused across models. Each coverage contributes its last snapshot. |
| **3. Prompt and schema** | One prompt (about 192K characters) defines all features with allowed values: **A01–A59** attorney, **L01–L69** litigation, **S01–S43** severity. Rules: ground every answer in the notes, treat the snapshot date as "today" (no future knowledge), copy allowed values exactly. The JSON schema is **parsed from the prompt itself**, so the definitions and validation cannot drift apart. The fixed part is the system prompt (cached); the claim facts and notes are the message. |
| **4. Token estimate** | Estimates input, thinking and answer tokens and cost for every snapshot **before** spending (about $34 for 100). |
| **5. Single test** | Runs one snapshot and checks validity, tokens and answers before the full run. |
| **6. Full run** | Warms the prompt cache with one call, then runs 10 in parallel. One JSON per snapshot in `output/`. Finished snapshots are skipped, so the run can be stopped and resumed. |
| **7. Tokens and cost** | Real per-call usage split into input, thinking and answer, checked against the SDK's own cost figure. |
| **8. Stability** | Compares this run with an earlier run (`PREVIOUS_DIR`) value by value. Features with low agreement point to definitions that need tightening. |

### How a call works

- No tools and no project settings are loaded (`tools=[]`, `setting_sources=[]`). The agent runs in an empty temp
  folder and only reads the notes and answers.
- The output is constrained by a JSON schema (`enum` per feature), then **every value is re-checked exactly**
  (case-sensitive) in Python. An invalid answer is retried once with the problems listed. Failures go to
  `output/_failed/`.

### Pilot results (100 snapshots)

| Check | Result |
|---|---|
| Valid on the first attempt | **100 / 100** |
| Run-to-run agreement (two independent runs) | **97.8%** of all values identical; median snapshot 98.2%; all 171 features ≥ 90% |
| Least stable features | S43, A48, A53, A08 (90–92%), marked for definition review |
| Cost | **$0.34 per snapshot** on average (max $0.74), $34 total; tokens per call about 92–96K in, 5–9K thinking, about 8.5K answer |

### Outputs (local and git-ignored, since they contain raw notes and evidence quotes)

| Path | Content |
|---|---|
| `data/pilot_sample_100.parquet` | the 100 sampled snapshots with notes |
| `data/pilot_sample_summary.csv` | counts per model and class only |
| `prompt_cache/system_prompt.md` | the fixed system prompt as sent |
| `output/<MODEL>_<CLM_CVG_PK>_snap<N>.json` | features, evidence, validity, tokens and cost per snapshot |

### Next steps

Scale extraction to the full train, val and holdout snapshot sets. Then train gradient-boosted or logistic models
on the 171 features, and compare holdout Gini by cutoff day against Strategy A on the same rows.
