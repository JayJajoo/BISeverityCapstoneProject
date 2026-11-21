"""
Shared data layer for the 3 cutoff-snapshot models.

Reads s3://{BUCKET}/{DS_PREFIX}/{model}/{split}/{feature}/part-*.parquet, the
output of ../2.1-DatasetDevelopment/2.1.3-build_cutoff_dataset.py. The split
(train/val/holdout) is already baked into the folder (computed once, globally,
by 2.1.2-assign_global_splits.py), so there is NO split-recomputation here.

Packs each split into its own (N,8,1536) float16 memmap (bounded RAM), plus a
small meta file with clm_cvg_pk / label / snapshot_num (needed to build the
"final snapshot per coverage" index).
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import boto3
import pyarrow.parquet as pq
import torch

# Set by 2.3.2-tune.py from its --bucket / --dataset-prefix arguments. 2.3.1-launch_sagemaker.py passes
# S3_BUCKET and CUTOFF_DATASET_PREFIX from config.py (the training job cannot import config.py).
BUCKET = None
DS_PREFIX = None
GROUPS = ["mean", "max", "min", "sum", "max_div_min", "max_minus_min",
          "first_div_last", "first_minus_last"]
EMB_DIM = 1536
COUNT_GROUP = "min"       # smallest/arbitrary group -> cheap pass for keys


def list_parts(s3, model, split):
    pref = f"{DS_PREFIX}/{model}/{split}/{COUNT_GROUP}/"
    return sorted(os.path.basename(o["Key"])
                  for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=pref).get("Contents", [])
                  if o["Key"].endswith(".parquet"))


def _feat(table):
    fl = table.column("features").combine_chunks()
    return fl.values.to_numpy(zero_copy_only=False).astype(np.float16).reshape(len(fl), EMB_DIM)


def build_memmap(model, split, out_dir, region="us-east-1", workers=8, log=print):
    """Pack one (model, split) combination into {out_dir}/{split}.X.f16 (N,8,1536)
    + {out_dir}/{split}.meta.npz (y, clm_cvg_pk, snapshot_num). Returns (N, pathX, pathM)."""
    os.makedirs(out_dir, exist_ok=True)
    s3 = boto3.client("s3", region_name=region)
    parts = list_parts(s3, model, split)
    t0 = time.time()

    counts, cvgs, snums, labels = [], [], [], []
    for pn in parts:
        b = s3.get_object(Bucket=BUCKET, Key=f"{DS_PREFIX}/{model}/{split}/{COUNT_GROUP}/{pn}")["Body"].read()
        t = pq.read_table(io.BytesIO(b), columns=["clm_cvg_pk", "snapshot_num", "label"])
        counts.append(t.num_rows)
        cvgs += t.column("clm_cvg_pk").to_pylist()
        snums += t.column("snapshot_num").to_pylist()
        labels += t.column("label").to_pylist()
    N = sum(counts)
    offsets = np.cumsum([0] + counts)
    pathX = os.path.join(out_dir, f"{split}.X.f16")
    X = np.memmap(pathX, dtype=np.float16, mode="w+", shape=(N, len(GROUPS), EMB_DIM))
    log(f"  [{model}/{split}] pack: N={N}, {len(parts)} parts ({time.time() - t0:.0f}s)")

    def dl(g, pn):
        b = s3.get_object(Bucket=BUCKET, Key=f"{DS_PREFIX}/{model}/{split}/{g}/{pn}")["Body"].read()
        return _feat(pq.read_table(io.BytesIO(b)))

    for pi, pn in enumerate(parts):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            mats = list(ex.map(lambda g: dl(g, pn), GROUPS))
        block = np.stack(mats, axis=1)
        a = offsets[pi]
        X[a:a + counts[pi]] = block
        if pi % 4 == 0 or pi == len(parts) - 1:
            X.flush()
            log(f"  [{model}/{split}] pack: part {pi + 1}/{len(parts)} ({time.time() - t0:.0f}s)")

    X.flush()
    pathM = os.path.join(out_dir, f"{split}.meta.npz")
    np.savez(pathM, y=np.asarray(labels, dtype=np.float32),
             clm_cvg_pk=np.asarray(cvgs, dtype=np.int64),
             snapshot_num=np.asarray(snums, dtype=np.int32), n=N, groups=np.asarray(GROUPS))
    del X
    log(f"  [{model}/{split}] pack: DONE N={N} ({time.time() - t0:.0f}s)")
    return N, pathX, pathM


def load_meta(out_dir, split):
    m = np.load(os.path.join(out_dir, f"{split}.meta.npz"), allow_pickle=True)
    return m["y"], m["clm_cvg_pk"], m["snapshot_num"], int(m["n"])


def open_memmap(out_dir, split, N):
    return np.memmap(os.path.join(out_dir, f"{split}.X.f16"), dtype=np.float16,
                      mode="r", shape=(N, len(GROUPS), EMB_DIM))


def last_snapshot_indices(clm_cvg_pk, snapshot_num):
    """Index of the row with the MAX snapshot_num within each clm_cvg_pk -- the
    final (most-informed) cutoff representation of that coverage. Rows for a
    given CVG are not guaranteed contiguous (part files interleave CVGs), so
    this groups explicitly rather than assuming a sorted block structure."""
    order = np.lexsort((snapshot_num, clm_cvg_pk))
    cvg_sorted = clm_cvg_pk[order]
    chg = np.flatnonzero(cvg_sorted[:-1] != cvg_sorted[1:])
    last_pos = np.append(chg, len(cvg_sorted) - 1)
    return order[last_pos]


class MemmapFeatures(torch.utils.data.Dataset):
    """Rows from an (N,8,1536) fp16 memmap, restricted to `indices`."""
    def __init__(self, X, y, indices):
        self.X, self.y, self.idx = X, y, indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.ascontiguousarray(self.X[j]))
        return x, torch.tensor(self.y[j], dtype=torch.float32)
