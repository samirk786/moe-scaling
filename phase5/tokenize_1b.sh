#!/bin/bash
# Tokenize the large pool and register it as a DataMix.
# Run on shelob AFTER build_1b.py finishes. Each step verifies before the next.
set -euo pipefail

TOK=/scratch/users/samir.kassam/conda/envs/tok/bin
RAW=/scratch/users/samir.kassam/dclm/raw/dclm1b.jsonl.gz
DEST=/scratch/users/samir.kassam/dclm/tokenized/dclm1b_train
REPO=/accounts/projects/berkeleynlp/samir.kassam/EMO

# --- 1. tokenize -------------------------------------------------------------
# id field is required: this dolma build silently DROPS records without one.
# max_size high enough that the pool lands in a single part file, matching the
# 92M pool's layout (the mix .txt names one part explicitly).
mkdir -p "$DEST"
$TOK/dolma tokens \
  --documents "$RAW" \
  --destination "$DEST" \
  --tokenizer.name_or_path allenai/dolma2-tokenizer \
  --tokenizer.eos_token_id 100257 \
  --tokenizer.pad_token_id 100277 \
  --fields.id_field_name id \
  --dtype uint32 \
  --max_size 20_000_000_000 \
  --processes 16

echo "=== tokenizer output"
ls -la "$DEST"

# --- 2. verify ---------------------------------------------------------------
# Raw flat binary, headerless. np.load raises "pickled data" here; use memmap.
OPENBLAS_NUM_THREADS=1 $TOK/python - << 'PY'
import glob, numpy as np
parts = sorted(glob.glob("/scratch/users/samir.kassam/dclm/tokenized/dclm1b_train/*.npy"))
print(f"{len(parts)} part file(s)")
total = 0
for p in parts:
    a = np.memmap(p, dtype=np.uint32, mode="r")
    eos = int((a == 100257).sum())
    print(f"  {p.split('/')[-1]}: {len(a):,} tokens, max id {a.max()}, {eos:,} eos")
    assert a.max() < 100278, "token id out of range"
    total += len(a)
print(f"TOTAL {total:,} tokens ({total/1e6:.1f}M)")
print(f"steps at 262144 tok/step: {total // 262144}")
PY
