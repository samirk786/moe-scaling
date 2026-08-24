#!/usr/bin/env python3
# Carve smaller unique-token pools out of the 92.47M train pool, keeping whole
# documents. Same distribution as the parent by construction, and disjoint from
# the val set because val was drawn from dclm10m, not from this pool.

import numpy as np
import os

SRC = "/scratch/users/samir.kassam/dclm/tokenized/dclm100m_train/part-0-00000.npy"
VAL = "/scratch/users/samir.kassam/dclm/tokenized/dclm_fixed_val/part-0-00000.npy"
ROOT = "/scratch/users/samir.kassam/dclm/tokenized"
EOS = 100257
SEED = 20260823
TARGETS = [("dclm30msub", 30_000_000), ("dclm10msub", 10_000_000)]

src = np.memmap(SRC, dtype=np.uint32, mode="r")
eos = np.flatnonzero(np.asarray(src) == EOS)
print(f"source: {src.size:,} tokens, {eos.size:,} docs")

starts = np.concatenate([[0], eos[:-1] + 1])
ends = eos + 1                       # exclusive, keeps the eos with its doc
lens = ends - starts
assert lens.sum() == ends[-1], "doc spans do not tile the file"

rng = np.random.default_rng(SEED)
order = rng.permutation(len(starts))

val = np.memmap(VAL, dtype=np.uint32, mode="r")
probe = np.asarray(val[:64])

for name, target in TARGETS:
    take, total = [], 0
    for i in order:
        if total >= target:
            break
        take.append(i)
        total += lens[i]
    take.sort()                      # keep source order; loader shuffles anyway
    out = np.concatenate([np.asarray(src[starts[i]:ends[i]]) for i in take])

    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "part-0-00000.npy")
    out.astype(np.uint32).tofile(p)

    n_eos = int((out == EOS).sum())
    hits = np.flatnonzero(out[:out.size - 64] == probe[0])
    leak = any(np.array_equal(out[i:i + 64], probe) for i in hits[:20000])
    print(f"\n{name}")
    print(f"  {out.size:,} tokens, {len(take):,} docs")
    print(f"  eos count {n_eos:,} == doc count: {n_eos == len(take)}")
    print(f"  max token {int(out.max())} < 100278: {int(out.max()) < 100278}")
    print(f"  val leak: {leak}")
    print(f"  steps/epoch {round(out.size / 262144)}   params/token {1_484_916_736 / out.size:.1f}")
    print(f"  -> {p}")
