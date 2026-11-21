"""
Global claim-level stratified train/val/holdout (70/15/15) split for the
embedding datasets. Same algorithm and seed as assign_global_claim_splits in
3.1-Dataset-Development.ipynb, applied to every closed coverage.

Computed ONCE per CLM_NUM (not per CLM_CVG_PK) over the full closed-coverage
universe, so every submodel (Attorney/Litigation/Severity) reuses the exact
same split -- a claim lands in the same bucket for all three. Stratifies on
ORG_GROUP_NAME + POL_STATE + LOSS_YEAR + max(ATTNY_FLG) + max(LITIG_FLG) +
max(SEV_BUCKET) per claim. Strata with <3 claims can't be split 3 ways -> all
forced to train. seed=42.

Output: s3://{S3_BUCKET}/{CUTOFF_DATASET_PREFIX}/_SPLITS.parquet (+ .json)   (both set in config.py)
"""
import datetime
import io
import json
import os
import sys

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root
from config import AWS_REGION as REGION, S3_BUCKET as BUCKET, COVERAGE_KEY, SEVERITY_KEY  # noqa: E402
from config import CUTOFF_DATASET_PREFIX as OUT_PREFIX  # noqa: E402

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def assign_global_claim_splits(closed_df, seed=SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    claim_attrs = closed_df.groupby("CLM_NUM").agg(
        ORG_GROUP_NAME=("ORG_GROUP_NAME", "first"),
        POL_STATE=("POL_STATE", "first"),
        LOSS_YEAR=("LOSS_YEAR", "first"),
        _ATTNY=("ATTNY_FLG", "max"),
        _LITIG=("LITIG_FLG", "max"),
        _SEV=("SEV_BUCKET", "max"),
    )
    claim_attrs["STRATUM"] = (
        claim_attrs["ORG_GROUP_NAME"].astype(str) + "|" +
        claim_attrs["POL_STATE"].astype(str) + "|" +
        claim_attrs["LOSS_YEAR"].astype(str) + "|" +
        claim_attrs["_ATTNY"].astype(str) + "|" +
        claim_attrs["_LITIG"].astype(str) + "|" +
        claim_attrs["_SEV"].astype(str)
    )

    rng = np.random.default_rng(seed)
    result = pd.Series(index=claim_attrs.index, dtype=object)

    for _, group in claim_attrs.groupby("STRATUM"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            result.loc[idx] = "train"
            continue
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        result.loc[idx[:n_train]] = "train"
        result.loc[idx[n_train:n_train + n_val]] = "val"
        result.loc[idx[n_train + n_val:]] = "holdout"

    return result, claim_attrs["STRATUM"]


def main():
    s3 = boto3.client("s3", region_name=REGION)

    cov_b = s3.get_object(Bucket=BUCKET, Key=COVERAGE_KEY)["Body"].read()
    cov = pq.read_table(io.BytesIO(cov_b)).to_pandas()

    sev_b = s3.get_object(Bucket=BUCKET, Key=SEVERITY_KEY)["Body"].read()
    sev = pq.read_table(io.BytesIO(sev_b), columns=["CLM_CVG_PK", "SEV_BUCKET", "LAST_CLOSE_DT"]).to_pandas()
    sev["CLM_CVG_PK"] = sev["CLM_CVG_PK"].astype("int64")

    merged = cov.merge(sev, on="CLM_CVG_PK", how="left")

    # closed coverages only (same rule as the notebook: CVG_CLOSE_DT not null)
    closed = merged[merged["CVG_CLOSE_DT"].notna()].copy()
    print(f"{len(merged):,} total coverages -> {len(closed):,} closed coverages in scope")
    print(f"missing LAST_CLOSE_DT (would drop from Severity only): {closed['LAST_CLOSE_DT'].isna().sum():,}")

    split_map, stratum = assign_global_claim_splits(closed)
    print("\nGlobal split sizes (claims):")
    print(split_map.value_counts())
    fracs = (split_map.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"\nfractions: {fracs}")
    n_small_strata = int((stratum.value_counts() < 3).sum())
    print(f"distinct strata: {stratum.nunique():,} (strata with <3 claims, forced to train: {n_small_strata:,})")

    out = pd.DataFrame({
        "claim_num": split_map.index.astype(str),
        "split": split_map.values,
        "stratum": stratum.loc[split_map.index].values,
    })
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="zstd")
    s3.put_object(Bucket=BUCKET, Key=f"{OUT_PREFIX}/_SPLITS.parquet", Body=buf.getvalue())

    meta = {
        "rule": "global claim-level stratified split, ported from "
                "3.1-Dataset-Development.ipynb::assign_global_claim_splits",
        "stratify_keys": ["ORG_GROUP_NAME", "POL_STATE", "LOSS_YEAR",
                          "max(ATTNY_FLG)", "max(LITIG_FLG)", "max(SEV_BUCKET)"],
        "seed": SEED, "fracs": {"train": TRAIN_FRAC, "val": VAL_FRAC, "holdout": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
        "small_stratum_rule": "strata with <3 claims forced entirely to train",
        "n_claims": len(out), "n_closed_coverages": len(closed),
        "n_strata": int(stratum.nunique()), "n_small_strata_forced_train": n_small_strata,
        "split_sizes_claims": split_map.value_counts().to_dict(),
        "source_coverage_table": f"s3://{BUCKET}/{COVERAGE_KEY}",
        "source_severity_table": f"s3://{BUCKET}/{SEVERITY_KEY}",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    s3.put_object(Bucket=BUCKET, Key=f"{OUT_PREFIX}/_SPLITS.json", Body=json.dumps(meta, indent=2, default=str).encode())
    print(f"\n-> s3://{BUCKET}/{OUT_PREFIX}/_SPLITS.parquet")
    print(f"-> s3://{BUCKET}/{OUT_PREFIX}/_SPLITS.json")


if __name__ == "__main__":
    main()
