#!/usr/bin/env python3
"""
Build the fixed router probe: one held-out batch and one training batch.

Both are drawn once with a fixed seed and reused at every probe point, in every
arm, at every rung. The held-out batch comes from the seeded val set the model
never trains on; the training batch comes from the 100M pool the model sees
once per epoch. Routing consistency rising on the training batch while the
held-out batch stays fluid is the memorization signature; both rising together
is ordinary convergence and says nothing about repetition.

Token files are raw flat uint32, so memmap rather than np.load.
"""

import argparse
import glob
from typing import List

import numpy as np


def windows(paths: List[str], seq_len: int) -> List[tuple]:
    out = []
    for path in paths:
        toks = np.memmap(path, dtype=np.uint32, mode="r")
        out.extend((path, i * seq_len) for i in range(toks.size // seq_len))
    return out


def sample(paths: List[str], n_seq: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    slots = windows(paths, seq_len)
    if len(slots) < n_seq:
        raise ValueError(f"need {n_seq} windows of {seq_len}, found {len(slots)} in {paths}")
    picks = rng.choice(len(slots), size=n_seq, replace=False)
    rows = []
    for i in sorted(picks):
        path, start = slots[i]
        toks = np.memmap(path, dtype=np.uint32, mode="r")
        rows.append(np.asarray(toks[start : start + seq_len], dtype=np.int32))
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-glob", required=True, help="dclm-fixed-val token files")
    parser.add_argument("--train-glob", required=True, help="dclm-100m-train token files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-seq", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    heldout_paths = sorted(glob.glob(args.heldout_glob))
    train_paths = sorted(glob.glob(args.train_glob))
    if not heldout_paths or not train_paths:
        raise SystemExit("one of the globs matched nothing")

    rng = np.random.default_rng(args.seed)
    heldout = sample(heldout_paths, args.n_seq, args.seq_len, rng)
    train = sample(train_paths, args.n_seq, args.seq_len, rng)

    np.savez_compressed(
        args.out,
        heldout=heldout,
        train=train,
        seed=np.array(args.seed),
        heldout_sources=np.array(heldout_paths),
        train_sources=np.array(train_paths),
    )
    tokens = args.n_seq * args.seq_len
    print(f"wrote {args.out}")
    print(f"  {args.n_seq} x {args.seq_len} = {tokens:,} tokens per split, seed {args.seed}")
    for name, arr in (("heldout", heldout), ("train", train)):
        print(f"  {name}: shape {arr.shape}, min {arr.min()}, max {arr.max()}")
    # sizing check: at N=128 top_8 the uniform per-expert share is tokens*8/128.
    for n_experts in (16, 32, 64, 128):
        print(f"  uniform share at N={n_experts:3d} top_8: {tokens * 8 / n_experts:,.0f} tokens")


if __name__ == "__main__":
    main()
