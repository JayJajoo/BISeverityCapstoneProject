"""
Coverage-level, day-grid-cutoff embedding datasets for Attorney / Litigation /
Severity, built directly from the per-note embeddings.

Uses the same boundary / margin-rule / day-grid logic as build_snapshots /
eligible_note_count in 3.1-Dataset-Development.ipynb. Instead of the cumulative
NOTE_TEXT, each row holds 8 elementwise summaries (mean, max, min, sum,
max_div_min, max_minus_min, first_div_last, first_minus_last) computed over
exactly the eligible notes at each surviving day-grid cutoff.

Margin rule (applied per note): strict `<` on the boundary date; if no note lands exactly
on the boundary, drop ONLY the single latest still-included note (by the
file's note_id seq order) if ITS OWN text matched an event keyword -- any
earlier notes on the same calendar date are unaffected and kept.

Grain: one row per (CLM_CVG_PK, snapshot). A claim with multiple coverages
gets its note history replicated per coverage, each with its own boundary,
day-grid walk, and target.

Input:  per-note embeddings written by 2.2.2-Embedding-Inference.ipynb
        (s3://{S3_BUCKET}/{EMBEDDING_PREFIX}/embeddings/notes_XXX.jsonl.out),
        _SPLITS.parquet (2.1.2-assign_global_splits.py), _keyword_registry.parquet (2.1.1-build_keyword_registry.py)
Output: s3://{S3_BUCKET}/{CUTOFF_DATASET_PREFIX}/{model}/{split}/{feature}/part-NNNNN.parquet
        + _DATASET.json manifest.   (all prefixes set in config.py)
"""
import argparse
import datetime
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root
from config import AWS_REGION as REGION, S3_BUCKET as BUCKET, COVERAGE_KEY, SEVERITY_KEY, TMP_DIR  # noqa: E402
from config import EMBEDDING_PREFIX, CUTOFF_DATASET_PREFIX as OUT_PREFIX  # noqa: E402

EMB_PREFIX = f"{EMBEDDING_PREFIX}/embeddings"      # written by 2.2.2-Embedding-Inference.ipynb
SPLITS_KEY = f"{OUT_PREFIX}/_SPLITS.parquet"
KW_REGISTRY_KEY = f"{OUT_PREFIX}/_keyword_registry.parquet"
EMB_DIM = 1536
EPS = 1e-6
FP16_MAX = 65504.0
CHUNK_ROWS = 100_000
TMP = str(TMP_DIR / "cutoffds")

MODELS = ["Attorney", "Litigation", "Severity"]
DAY_GRIDS = {
    "Attorney":   [1, 7, 15, 30, 60, 90, 180, 365, 730, 1095],
    "Litigation": [1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460],
    "Severity":   [1, 90, 180, 270, 365, 540, 730, 810, 900, 1095, 1460, 1825],
}
TARGET_SRC_COL = {"Attorney": "ATTNY_FLG", "Litigation": "LITIG_FLG", "Severity": "SEV_BUCKET"}
BOUNDARY_COL = {"Attorney": "BOUNDARY_ATTORNEY", "Litigation": "BOUNDARY_LITIGATION", "Severity": "BOUNDARY_SEVERITY"}
KW_COL = {"Attorney": "kw_attorney", "Litigation": "kw_litigation", "Severity": "kw_severity"}
FEATURES = ["mean", "max", "min", "sum", "max_div_min", "max_minus_min", "first_div_last", "first_minus_last"]
VARIATION_FORMULAS = {
    "mean": "elementwise mean of the eligible notes' embeddings as of this cutoff",
    "max": "elementwise max of the eligible notes' embeddings as of this cutoff",
    "min": "elementwise min of the eligible notes' embeddings as of this cutoff",
    "sum": "elementwise sum of the eligible notes' embeddings as of this cutoff",
    "max_div_min": "elementwise max / min (safe-divide eps=%g, clipped to +-%g)" % (EPS, FP16_MAX),
    "max_minus_min": "elementwise max - min (range)",
    "first_div_last": "elementwise first / last (safe-divide eps=%g, clipped to +-%g)" % (EPS, FP16_MAX),
    "first_minus_last": "elementwise first - last",
}


def safe_div(num, den):
    sign = np.where(den >= 0, 1.0, -1.0)
    d = np.where(np.abs(den) < EPS, sign * EPS, den)
    return np.clip(num / d, -FP16_MAX, FP16_MAX)


# --------------------------------------------------------------------------
# Phase 1: load boundary/target table + global split + keyword registry
# --------------------------------------------------------------------------
def load_boundary_table(s3):
    cov_b = s3.get_object(Bucket=BUCKET, Key=COVERAGE_KEY)["Body"].read()
    cov = pq.read_table(io.BytesIO(cov_b)).to_pandas()
    sev_b = s3.get_object(Bucket=BUCKET, Key=SEVERITY_KEY)["Body"].read()
    sev = pq.read_table(io.BytesIO(sev_b), columns=["CLM_CVG_PK", "SEV_BUCKET", "LAST_CLOSE_DT"]).to_pandas()
    sev["CLM_CVG_PK"] = sev["CLM_CVG_PK"].astype("int64")

    merged = cov.merge(sev, on="CLM_CVG_PK", how="left")
    closed = merged[merged["CVG_CLOSE_DT"].notna()].copy()
    closed["BOUNDARY_ATTORNEY"] = closed["ATTNY_DT"].where(closed["ATTNY_FLG"] == 1, closed["CVG_CLOSE_DT"])
    closed["BOUNDARY_LITIGATION"] = closed["LITIG_DT"].where(closed["LITIG_FLG"] == 1, closed["CVG_CLOSE_DT"])
    closed["BOUNDARY_SEVERITY"] = closed["LAST_CLOSE_DT"]
    return closed


def load_split_map(s3):
    b = s3.get_object(Bucket=BUCKET, Key=SPLITS_KEY)["Body"].read()
    df = pq.read_table(io.BytesIO(b), columns=["claim_num", "split"]).to_pandas()
    return pd.Series(df["split"].values, index=df["claim_num"].values)


def load_keyword_registry(s3):
    b = s3.get_object(Bucket=BUCKET, Key=KW_REGISTRY_KEY)["Body"].read()
    return pq.read_table(io.BytesIO(b)).to_pandas()


# --------------------------------------------------------------------------
# Phase 2: load + sort raw per-note embeddings (the notes_XXX.jsonl.out shards from 2.2.2)
# --------------------------------------------------------------------------
def list_shards(s3):
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=EMB_PREFIX + "/")
    return sorted(o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".out"))


def parse_shard(key, sample):
    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
    buf = io.TextIOWrapper(body, encoding="utf-8")
    tag = os.path.basename(key).replace(".jsonl.out", "")
    emb_path = os.path.join(TMP, f"{tag}.emb.f16")
    claims, dates, seqs = [], [], []
    n = 0
    with open(emb_path, "wb") as fh:
        for i, line in enumerate(buf):
            if sample and i >= sample:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:   # e.g. a bare NaN token; the standard json module accepts it
                r = json.loads(line)
            if not isinstance(r.get("date"), str):   # note without a readable date: it cannot be placed in time
                continue
            np.asarray(r["embedding"], dtype=np.float16).tofile(fh)
            claims.append(r["claim_num"])
            dates.append(r["date"])
            seqs.append(int(r["note_id"].rsplit("_", 1)[1]))
            n += 1
    paths = {"emb": emb_path}
    for name, arr, pk in (
        ("claim", np.asarray(claims, dtype=object), True),
        ("datei", np.asarray([int(d.replace("-", "")) for d in dates], dtype=np.int32), False),
        ("seq", np.asarray(seqs, dtype=np.int64), False)):
        p = os.path.join(TMP, f"{tag}.{name}.npy")
        np.save(p, arr, allow_pickle=pk)
        paths[name] = p
    return tag, paths, n


def load_sorted_embeddings(sample_per_shard, max_shards, workers):
    s3 = boto3.client("s3", region_name=REGION)
    shards = list_shards(s3)
    if max_shards:
        shards = shards[:max_shards]
    print(f"{len(shards)} embedding shards; sample_per_shard={sample_per_shard or 'FULL'}", flush=True)

    results, counts = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(parse_shard, k, sample_per_shard): k for k in shards}
        for f in as_completed(futs):
            tag, paths, n = f.result()
            results[tag] = paths
            counts[tag] = n
            print(f"  parsed {tag}: {n} notes", flush=True)

    order_tags = sorted(results)
    N = sum(counts[t] for t in order_tags)
    emb = np.empty((N, EMB_DIM), dtype=np.float16)
    off = 0
    for t in order_tags:
        n = counts[t]
        emb[off:off + n] = np.fromfile(results[t]["emb"], dtype=np.float16).reshape(n, EMB_DIM)
        off += n
        os.remove(results[t]["emb"])
    claim = np.concatenate([np.load(results[t]["claim"], allow_pickle=True) for t in order_tags])
    date_int = np.concatenate([np.load(results[t]["datei"]) for t in order_tags])
    seq = np.concatenate([np.load(results[t]["seq"]) for t in order_tags])
    print(f"loaded {N:,} notes, emb {emb.shape} {emb.dtype} ({emb.nbytes / 1e9:.1f} GB)", flush=True)

    claim_code = np.unique(claim, return_inverse=True)[1]
    order = np.lexsort((seq, date_int, claim_code))
    claim_sorted = claim[order]
    date_int_sorted = date_int[order]
    claim_code_sorted = claim_code[order]
    seq_sorted = seq[order]
    return emb, order, claim_sorted, date_int_sorted, claim_code_sorted, seq_sorted


# --------------------------------------------------------------------------
# Phase 3: per-claim cumulative arrays + per-(CVG,model) snapshot emission
# --------------------------------------------------------------------------
class PartWriter:
    def __init__(self, s3, model, split, name, out_prefix, local_tmp):
        self.s3, self.model, self.split, self.name, self.out_prefix, self.local_tmp = (
            s3, model, split, name, out_prefix, local_tmp)
        self.keys_cvg, self.keys_claim, self.keys_snum, self.keys_sday, self.keys_n, self.keys_label = (
            [], [], [], [], [], [])
        self.vecs = []
        self.rows = 0
        self.part = 0
        self.total = 0

    def add(self, cvg, claim, snum, sday, n_notes, label, vec):
        self.keys_cvg.append(cvg); self.keys_claim.append(claim)
        self.keys_snum.append(snum); self.keys_sday.append(sday)
        self.keys_n.append(n_notes); self.keys_label.append(label)
        self.vecs.append(vec)
        self.rows += 1
        if self.rows >= CHUNK_ROWS:
            self.flush()

    def flush(self):
        if self.rows == 0:
            return
        vecs = np.stack(self.vecs).astype(np.float16)
        flat = pa.array(vecs.reshape(-1), type=pa.float16())
        feat = pa.FixedSizeListArray.from_arrays(flat, EMB_DIM)
        tbl = pa.table({
            "clm_cvg_pk": pa.array(self.keys_cvg, type=pa.int64()),
            "claim_num": pa.array(self.keys_claim, type=pa.string()),
            "snapshot_num": pa.array(self.keys_snum, type=pa.int32()),
            "snapshot_day": pa.array(self.keys_sday, type=pa.int32()),
            "n_notes": pa.array(self.keys_n, type=pa.int32()),
            "label": pa.array(self.keys_label, type=pa.int8()),
            "features": feat,
        })
        local = os.path.join(self.local_tmp, f"{self.name}-part-{self.part:05d}.parquet")
        pq.write_table(tbl, local, compression="zstd")
        key = f"{self.out_prefix}/{self.model}/{self.split}/{self.name}/part-{self.part:05d}.parquet"
        self.s3.upload_file(local, BUCKET, key)
        os.remove(local)
        self.total += self.rows
        self.part += 1
        self.keys_cvg, self.keys_claim, self.keys_snum, self.keys_sday, self.keys_n, self.keys_label = (
            [], [], [], [], [], [])
        self.vecs = []
        self.rows = 0


def eligible_note_count(date_int_claim, boundary_int, kw_flags_claim):
    """Note-level margin rule. date_int_claim/kw_flags_claim: this claim's notes,
    sorted by (date,seq). boundary_int: YYYYMMDD int. Returns max eligible n."""
    n_before = int(np.searchsorted(date_int_claim, boundary_int, side="left"))
    if n_before == 0:
        return 0
    n_upto = int(np.searchsorted(date_int_claim, boundary_int, side="right"))
    has_exact = n_upto > n_before
    if has_exact:
        return n_before
    if kw_flags_claim[n_before - 1]:
        return n_before - 1
    return n_before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-per-shard", type=int, default=0, help="cap notes/shard for a quick test")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    out_prefix = args.out_prefix or (OUT_PREFIX + ("/_sample" if args.sample_per_shard else ""))
    os.makedirs(TMP, exist_ok=True)
    s3 = boto3.client("s3", region_name=REGION)
    t0 = time.time()

    print("loading boundary table / split map / keyword registry...", flush=True)
    closed = load_boundary_table(s3)
    split_map = load_split_map(s3)
    kw_df = load_keyword_registry(s3)
    kw_by_seq = {
        "Attorney": kw_df.set_index("seq")["kw_attorney"].to_dict(),
        "Litigation": kw_df.set_index("seq")["kw_litigation"].to_dict(),
        "Severity": kw_df.set_index("seq")["kw_severity"].to_dict(),
    }
    open_dt_by_claim = closed.groupby("CLM_NUM")["CLM_OPEN_DT"].first()
    print(f"  {len(closed):,} closed coverages, {closed.CLM_NUM.nunique():,} claims "
          f"({time.time() - t0:.0f}s)", flush=True)

    print("loading + sorting raw per-note embeddings...", flush=True)
    emb, order, claim_sorted, date_int_sorted, claim_code_sorted, seq_sorted = load_sorted_embeddings(
        args.sample_per_shard, args.max_shards, args.workers)
    N = len(claim_sorted)
    t1 = time.time()
    print(f"  sorted {N:,} notes ({t1 - t0:.0f}s)", flush=True)

    # per-model keyword flag array, aligned to the SAME sorted order
    kw_arrays = {}
    for model in MODELS:
        d = kw_by_seq[model]
        kw_arrays[model] = np.array([bool(d.get(int(s), False)) for s in seq_sorted], dtype=bool)

    writers = {
        (model, split, feat): PartWriter(s3, model, split, feat, out_prefix, TMP)
        for model in MODELS for split in ["train", "val", "holdout"] for feat in FEATURES
    }

    cov_by_claim = closed.groupby("CLM_NUM")
    bnd = np.flatnonzero(np.diff(claim_code_sorted)) + 1
    starts = np.concatenate(([0], bnd))
    ends = np.concatenate((bnd, [N]))

    stats = {model: {"cvgs_processed": 0, "rows": 0, "label_sum": 0} for model in MODELS}
    claims_done = 0
    for a, b in zip(starts, ends):
        claim_id = claim_sorted[a]
        try:
            cvg_rows = cov_by_claim.get_group(claim_id)
        except KeyError:
            continue
        split = split_map.get(claim_id)
        if split is None:
            continue
        open_dt = open_dt_by_claim.get(claim_id)
        if pd.isna(open_dt):
            continue

        seg = emb[order[a:b]].astype(np.float32)
        di = date_int_sorted[a:b]
        csum = np.cumsum(seg, axis=0)
        cmax = np.maximum.accumulate(seg, axis=0)
        cmin = np.minimum.accumulate(seg, axis=0)
        first = seg[0]

        for model in MODELS:
            for _, cvg_row in cvg_rows.iterrows():
                boundary = cvg_row[BOUNDARY_COL[model]]
                if pd.isna(boundary):
                    continue
                boundary_int = int(pd.Timestamp(boundary).strftime("%Y%m%d"))
                max_eligible_n = eligible_note_count(di, boundary_int, kw_arrays[model][a:b])
                if max_eligible_n == 0:
                    continue
                target_val = cvg_row[TARGET_SRC_COL[model]]
                if pd.isna(target_val):
                    continue
                label = int(target_val)
                cvg_pk = int(cvg_row["CLM_CVG_PK"])

                prev_n = 0
                snapshot_num = 0
                for day in DAY_GRIDS[model]:
                    cutoff = open_dt + pd.Timedelta(days=day)
                    if cutoff >= boundary:
                        break
                    cutoff_int = int(cutoff.strftime("%Y%m%d"))
                    n_this = int(np.searchsorted(di[:max_eligible_n], cutoff_int, side="left"))
                    if n_this == 0 or n_this == prev_n:
                        continue
                    snapshot_num += 1
                    idx = n_this - 1
                    s_sum = csum[idx]
                    s_mean = s_sum / n_this
                    s_max = cmax[idx]
                    s_min = cmin[idx]
                    s_last = seg[idx]
                    out_vecs = {
                        "mean": s_mean, "max": s_max, "min": s_min, "sum": s_sum,
                        "max_div_min": safe_div(s_max, s_min),
                        "max_minus_min": s_max - s_min,
                        "first_div_last": safe_div(first, s_last),
                        "first_minus_last": first - s_last,
                    }
                    for feat, vec in out_vecs.items():
                        writers[(model, split, feat)].add(
                            cvg_pk, claim_id, snapshot_num, day, n_this, label, vec.astype(np.float16))
                    stats[model]["rows"] += 1
                    stats[model]["label_sum"] += label
                    prev_n = n_this
                if snapshot_num > 0:
                    stats[model]["cvgs_processed"] += 1

        claims_done += 1
        if claims_done % 20000 == 0:
            print(f"  processed {claims_done:,} claims ({time.time() - t1:.0f}s)", flush=True)

    for w in writers.values():
        w.flush()
    print(f"\nDONE aggregating: {claims_done:,} claims ({time.time() - t0:.0f}s)", flush=True)
    for model in MODELS:
        print(f"  {model}: {stats[model]['cvgs_processed']:,} CVGs, "
              f"{stats[model]['rows']:,} rows, label_sum={stats[model]['label_sum']:,}", flush=True)

    manifest = {
        "purpose": "Coverage-level (CLM_CVG_PK), day-grid-cutoff embedding snapshot datasets for "
                   "Attorney/Litigation/Severity. Built directly from raw per-note embeddings "
                   "(not the date-deduped running-feature store), one row per (CLM_CVG_PK, snapshot).",
        "ported_from": "3.1-Dataset-Development.ipynb "
                       "(boundary/margin-rule/day-grid logic; NOTE_TEXT replaced with embedding aggregates)",
        "source_embeddings": f"s3://{BUCKET}/{EMB_PREFIX}/",
        "source_coverage_table": f"s3://{BUCKET}/{COVERAGE_KEY}",
        "source_severity_table": f"s3://{BUCKET}/{SEVERITY_KEY}",
        "split_table": f"s3://{BUCKET}/{SPLITS_KEY}",
        "keyword_registry": f"s3://{BUCKET}/{KW_REGISTRY_KEY}",
        "boundary_rule": {
            "Attorney": "ATTNY_DT if ATTNY_FLG==1 else CVG_CLOSE_DT",
            "Litigation": "LITIG_DT if LITIG_FLG==1 else CVG_CLOSE_DT",
            "Severity": "LAST_CLOSE_DT always",
        },
        "margin_rule": "strict '<' on boundary; if no note lands exactly on the boundary, drop ONLY "
                       "the single latest still-included note (by note_id seq order) if its own text "
                       "matched an event keyword; earlier same-date notes are never affected",
        "day_grids": DAY_GRIDS,
        "dedup_rule": "day-grid snapshot skipped if its eligible note count is unchanged from the "
                     "previous surviving snapshot, or if it would have 0 eligible notes",
        "label_col": "label (int8): ATTNY_FLG/LITIG_FLG (0/1) for Attorney/Litigation; "
                     "SEV_BUCKET (0-4) for Severity",
        "features": FEATURES, "embedding_dim": EMB_DIM, "dtype": "float16",
        "variations": [{"name": n, "formula": f} for n, f in VARIATION_FORMULAS.items()],
        "parquet_schema": {"clm_cvg_pk": "int64", "claim_num": "string", "snapshot_num": "int32",
                           "snapshot_day": "int32", "n_notes": "int32", "label": "int8",
                           "features": f"fixed_size_list<float16>[{EMB_DIM}]"},
        "split_rule": "global claim-level stratified 70/15/15 (see _SPLITS.json); split is encoded "
                     "by the {model}/{split}/ folder, not a column",
        "stats": stats,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    s3.put_object(Bucket=BUCKET, Key=f"{out_prefix}/_DATASET.json", Body=json.dumps(manifest, indent=2, default=str).encode())
    print(f"manifest -> s3://{BUCKET}/{out_prefix}/_DATASET.json", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
