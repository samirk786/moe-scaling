#!/usr/bin/env python3
# Build a large unique-token pool from codelion/dclm-baseline-1B for the R=1
# (single-pass) MoE-vs-dense comparison.
#
# Shard-at-a-time: download one parquet, filter out anything whose text matches
# the fixed val set, append to the jsonl.gz, delete the parquet. /scratch is
# 20GB and the HF cache keeps a second copy of every blob, so holding all nine
# at once does not fit.
#
# Token estimate is chars/3.6, calibrated on the existing 92.47M pool. The real
# count comes from the .npy size after dolma runs.

import gzip
import json
import os
import pickle
import shutil
import sys
import hashlib

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "codelion/dclm-baseline-1B"
OUT = "/scratch/users/samir.kassam/dclm/raw/dclm1b.jsonl.gz"
CACHE = "/scratch/users/samir.kassam/hf_cache"
VAL_HASHES = "/scratch/users/samir.kassam/val_hashes.pkl"
TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 550_000_000
CHARS_PER_TOK = 3.6

val = pickle.load(open(VAL_HASHES, "rb"))
print(f"loaded {len(val)} val hashes; target ~{TARGET_TOKENS/1e6:.0f}M tokens")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
chars = docs = skipped = 0
doc_id = 0

with gzip.open(OUT, "wt") as out:
    for i in range(9):
        fn = f"data/train-{i:05d}-of-00009.parquet"
        print(f"\n[{i}] downloading {fn} ...", flush=True)
        p = hf_hub_download(REPO, fn, repo_type="dataset", cache_dir=CACHE)
        t = pq.read_table(p)
        col = "text" if "text" in t.column_names else t.column_names[0]
        print(f"[{i}] {t.num_rows:,} rows, columns={t.column_names}", flush=True)

        for txt in t.column(col).to_pylist():
            if not txt:
                continue
            if hashlib.md5(txt.encode()).hexdigest() in val:
                skipped += 1
                continue
            out.write(json.dumps({"id": str(doc_id), "text": txt}) + "\n")
            doc_id += 1
            docs += 1
            chars += len(txt)

        # drop both the symlink target and the blob, or the cache keeps growing
        real = os.path.realpath(p)
        for path in (real, p):
            try:
                os.remove(path)
            except OSError:
                pass
        est = chars / CHARS_PER_TOK
        print(f"[{i}] cumulative {docs:,} docs, ~{est/1e6:.1f}M tokens "
              f"({skipped} val dupes dropped)", flush=True)
        if est >= TARGET_TOKENS:
            print(f"[{i}] target reached, stopping", flush=True)
            break

shutil.rmtree(os.path.join(CACHE, "datasets"), ignore_errors=True)
print(f"\nDONE: {docs:,} docs, ~{chars/CHARS_PER_TOK/1e6:.1f}M est tokens -> {OUT}")
print(f"dropped {skipped} documents matching the fixed val set")
