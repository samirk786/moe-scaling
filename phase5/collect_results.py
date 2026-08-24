#!/usr/bin/env python3
# Pull val/CE and Hellaswag BPB out of the slurm logs into one CSV.
# Run name comes from the wandb name line, arm/scheduler from the run prefix.

import csv
import glob
import re
import sys

LOGDIR = "/scratch/users/samir.kassam/sched_logs"
OUT = "/scratch/users/samir.kassam/results.csv"
SPE = 353

NAME = re.compile(r"(cosine_ep16|trunk|fork_ep16_1-sqrt|moe32)_lr([0-9e.\-]+)_wd([0-9.]+)")
STEP = re.compile(r"console_logger:6[07].*step=(\d+)/(\d+)")
CE = re.compile(r"val/CE loss=([0-9.]+)")
BPB = re.compile(r"hellaswag \(BPB\)=([0-9.]+)")


def parse(path):
    name = arm = lr = wd = None
    step = None
    rows = []
    pending_ce = None
    with open(path, errors="ignore") as f:
        for line in f:
            if name is None:
                m = NAME.search(line)
                if m:
                    name, lr, wd = m.group(0), m.group(2), m.group(3)
                    arm = m.group(1)
            m = STEP.search(line)
            if m:
                step = int(m.group(1))
                continue
            m = CE.search(line)
            if m:
                pending_ce = float(m.group(1))
                rows.append([step, pending_ce, None])
                continue
            m = BPB.search(line)
            if m and rows:
                rows[-1][2] = float(m.group(1))
    if name is None:
        return []
    out = []
    seen = set()
    for step, ce, bpb in rows:
        if step is None:
            continue
        # only boundary evals are valid datapoints; skip post-requeue startup evals
        if step % SPE != 0:
            continue
        if step in seen:
            continue
        seen.add(step)
        out.append(dict(run=name, arm=arm, lr=lr, wd=wd,
                        step=step, epoch=step // SPE, val_ce=ce, hs_bpb=bpb,
                        log=path.split("/")[-1]))
    return out


rows = []
for p in sorted(glob.glob(f"{LOGDIR}/*.out")):
    rows.extend(parse(p))

rows.sort(key=lambda r: (r["arm"], r["lr"], r["wd"], r["epoch"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "arm", "lr", "wd", "step", "epoch",
                                      "val_ce", "hs_bpb", "log"])
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} rows -> {OUT}")
runs = sorted({(r["arm"], r["lr"], r["wd"]) for r in rows})
print(f"{len(runs)} runs")
for arm in sorted({r[0] for r in runs}):
    n = sum(1 for r in runs if r[0] == arm)
    print(f"  {arm:20s} {n}")
