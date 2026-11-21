# 2.2 Note embeddings

Embeds every adjuster note **once** with `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, which outputs 1,536-dim,
L2-normalized vectors. A note's embedding doesn't depend on the target or the cutoff, so 4.65M notes are
embedded once and reused by all three models and every snapshot.

Run in order: **2.2.1 → 2.2.2**. Paths come from `config.py` at the repo root.

## `2.2.1-Note-Preparation.ipynb`: prepare the notes

- Loads all 16 `BI_Notes` batches: 4,645,819 notes across 138,233 claims.
- **Light cleaning only:** HTML tags are removed, `<br>` becomes a newline, entities are unescaped and repeated
  spaces are collapsed. Shorthand, casing and typos are kept on purpose, because that's what the model has to read.
- One record per note: `{note_id, claim_num, date, text}`. The `note_id` is `<claim>_<seq>`, where `seq` is the
  global row order. `../2.1-DatasetDevelopment/2.1.1-build_keyword_registry.py` uses the same order.
- No split and no cutoff are applied here. Step 2.1 applies both.
- Writes 20 round-robin JSON-Lines shards of about 232K notes each, so the endpoint load is even.
- **Output:** `{EMBEDDING_PREFIX}/input/notes_000.jsonl … notes_019.jsonl`

## `2.2.2-Embedding-Inference.ipynb`: embed every note on SageMaker

- The model has 1.5B parameters and an 8,192-token cap. Each batch is padded only to its longest note, so short
  notes stay cheap.
- Writes a custom `code/inference.py` handler and a pinned `code/requirements.txt` (`transformers==4.51.3`,
  `sentence-transformers==5.7.0`), then packages them as `model.tar.gz`.
- Deploys a real-time endpoint on **4 × ml.g5.xlarge** and streams each shard with 16 parallel requests of 64
  notes each.
- **Cost guard:** a running cost estimate stops the job at **$100**, and a `finally` block always deletes the
  endpoint. Shards already in S3 are skipped, so a stopped run can resume.
- Sanity check: an output record has 1,536 dimensions and an L2 norm of 1.0.
- **Output:** `{EMBEDDING_PREFIX}/embeddings/notes_XXX.jsonl.out`, one `{note_id, claim_num, date, embedding[1536]}`
  per line. `../2.1-DatasetDevelopment/2.1.3-build_cutoff_dataset.py` reads these files.
