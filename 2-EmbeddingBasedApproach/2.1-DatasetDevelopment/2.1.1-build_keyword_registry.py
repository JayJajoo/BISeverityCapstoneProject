"""
One-time, reusable registry: for every note (in the SAME global seq order that
2.2.1-Note-Preparation.ipynb used to build note_id = f"{claim}_{seq}"), whether its cleaned
text matches each model's event-keyword regex (Attorney/Litigation/Severity).

This lets the margin rule run later without re-reading the raw BI_Notes CSVs or
holding any note text in memory -- only (seq -> 3 booleans).

seq is the row number across all files read in sorted order, counted on the rows
pandas parses (the same read_csv call as 2.2.1), so it matches note_id even when a
note contains line breaks. Files are processed in parallel and then concatenated
in file order, and seq is assigned after that. Text is cleaned the same way as
clean_note in 3.1-Dataset-Development.ipynb (<br> to newline, tags removed, HTML
entities decoded). Keyword matching is vectorized with pandas .str.contains, so all
4.6M notes are flagged in a few minutes.

Output: {TMP_DIR}/_keyword_registry.parquet and s3://{S3_BUCKET}/{CUTOFF_DATASET_PREFIX}/_keyword_registry.parquet
(set in config.py), with columns [seq, claim_num, date, kw_attorney,
kw_litigation, kw_severity].
"""
import glob
import html
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import boto3
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root
from config import AWS_REGION, NOTES_DIR, S3_BUCKET as BUCKET, TMP_DIR, CUTOFF_DATASET_PREFIX  # noqa: E402

DATA_DIR = str(NOTES_DIR)
OUT_KEY = f"{CUTOFF_DATASET_PREFIX}/_keyword_registry.parquet"
LOCAL_OUT = str(TMP_DIR / "_keyword_registry.parquet")

BR_RE = re.compile(r"<br\s*/?>", re.I)
TAG_RE = re.compile(r"<[^>]+>")

KEYWORDS = {
    "attorney":   r"(?i)\batty\b|attorney|lawyer|\bLOR\b|letter of rep|represent|retain|\besq\b|counsel",
    "litigation": r"(?i)summons|complaint|lawsuit|litigation|\bS&C\b|docket|subpoena|\bserved\b|\bsuit\b",
    "severity":   r"(?i)settle|settlement|release|closing papers|payment issued|\bclosed\b|resolved|\bstip\b|stipulation",
}


def process_file(fpath):
    df = pd.read_csv(fpath, usecols=["CLAIM_NUM", "DATE", "TEXT_NOTE"], dtype={"CLAIM_NUM": str})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    text = df["TEXT_NOTE"].fillna("").astype(str)
    text = text.str.replace(BR_RE, "\n", regex=True)
    text = text.str.replace(TAG_RE, "", regex=True)
    text = text.map(html.unescape)          # e.g. 'S&amp;C' -> 'S&C', as in 3.1's clean_note

    out = pd.DataFrame({
        "claim_num": df["CLAIM_NUM"].values,
        "date": df["DATE"].values,
        "kw_attorney": text.str.contains(KEYWORDS["attorney"], regex=True).values,
        "kw_litigation": text.str.contains(KEYWORDS["litigation"], regex=True).values,
        "kw_severity": text.str.contains(KEYWORDS["severity"], regex=True).values,
    })
    return os.path.basename(fpath), out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.csv")))   # same files and order as 2.2.1
    print(f"{len(files)} BI_Notes files found", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for name, out in ex.map(process_file, files):
            results[name] = out
            print(f"  done {name}: {len(out):,} rows", flush=True)

    ordered = [results[os.path.basename(f)] for f in files]
    full = pd.concat(ordered, ignore_index=True)
    full.insert(0, "seq", range(len(full)))   # row number over all files = the seq in 2.2.1's note_id
    print(f"total rows: {len(full):,}", flush=True)

    tbl = pa.Table.from_pandas(full, preserve_index=False)
    pq.write_table(tbl, LOCAL_OUT, compression="zstd")
    print(f"wrote -> {LOCAL_OUT}", flush=True)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(LOCAL_OUT, BUCKET, OUT_KEY)
    print(f"uploaded -> s3://{BUCKET}/{OUT_KEY}", flush=True)


if __name__ == "__main__":
    main()
