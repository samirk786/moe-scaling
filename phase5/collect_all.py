#!/usr/bin/env python3
# Pull val/CE and Hellaswag BPB from the slurm logs.
# Replaces collect_results.py. Two things that one got wrong:
#   - run names with a size or pool tag (trunk_n761m_, trunk_u10m_) never matched
#   - epoch was step//353 everywhere, but steps per epoch is U/262144 and U varies
# Config is read out of each log rather than hardcoded, so a new pool needs no edit.

import csv
import glob
import re
from collections import defaultdict

LOGDIR = "/scratch/users/samir.kassam/sched_logs"
OUT = "/scratch/users/samir.kassam/results_all.csv"
TOK_PER_STEP = 262144

# token counts verified on disk
POOLS = {
    "dclm-100m-train": 92_469_693,
    "dclm-10m-sub": 10_004_022,
    "dclm-30m-sub": 30_001_113,
}

SAVE = re.compile(r"save_folder='[^']*/models/([^']+)'")
DUR = re.compile(r"max_duration=Duration\(value=(\d+)")
MIX = re.compile(r"mix='(dclm-[0-9a-z-]+)'")
DMODEL = re.compile(r"d_model=(\d+)")
NLAYERS = re.compile(r"n_layers=(\d+)")
STEP = re.compile(r"step=(\d+)/(\d+)")
CE = re.compile(r"val/CE loss=([0-9.]+)")
BPB = re.compile(r"hellaswag \(BPB\)=([0-9.]+)")

# name -> arm, size tag, pool tag, lr, wd
NAME = re.compile(r"^(cosine_ep16|trunk|fork_ep16_1-sqrt|moe32)"
                  r"(?:_(n\d+m|u\d+m\d*e?p?|thid\d+m))?"
                  r"_lr([0-9e.\-]+)_wd([0-9.]+)$")


def scan(path):
    head = open(path, errors="ignore").read(200_000)
    m = SAVE.search(head)
    if not m or "smoke" in m.group(1):
        return None
    name = m.group(1)
    nm = NAME.match(name)
    if not nm:
        print(f"  UNPARSED NAME: {name}  ({path.split('/')[-1]})")
        return None
    mix = MIX.search(head)
    dur = DUR.search(head)
    if not mix or not dur or mix.group(1) not in POOLS:
        print(f"  no pool/duration: {name}")
        return None
    return dict(name=name, arm=nm.group(1), tag=nm.group(2) or "",
                lr=nm.group(3), wd=nm.group(4),
                pool=mix.group(1), u=POOLS[mix.group(1)],
                total=int(dur.group(1)) // TOK_PER_STEP,
                d_model=int(DMODEL.search(head).group(1)) if DMODEL.search(head) else None,
                n_layers=int(NLAYERS.search(head).group(1)) if NLAYERS.search(head) else None)


def curve(path, total):
    # anchor the step counter on the run's own denominator; the LM evaluator
    # prints its own step=N/862 progress and would otherwise be picked up
    step = None
    pts = []
    for line in open(path, errors="ignore"):
        m = STEP.search(line)
        if m and int(m.group(2)) == total:
            step = int(m.group(1))
            continue
        m = CE.search(line)
        if m and step is not None:
            pts.append([step, float(m.group(1)), None])
            continue
        m = BPB.search(line)
        if m and pts:
            pts[-1][2] = float(m.group(1))
    return pts


runs = defaultdict(lambda: {"meta": None, "pts": {}, "logs": []})
for p in sorted(glob.glob(f"{LOGDIR}/*.out")):
    meta = scan(p)
    if not meta:
        continue
    r = runs[meta["name"]]
    r["meta"] = meta
    r["logs"].append(p.split("/")[-1])
    for step, ce, bpb in curve(p, meta["total"]):
        old = r["pts"].get(step)
        # requeue can replay a step; keep whichever row carries bpb
        if old is None or (old[1] is None and bpb is not None):
            r["pts"][step] = (ce, bpb)

rows = []
for name, r in runs.items():
    m = r["meta"]
    spe = m["u"] / TOK_PER_STEP
    for step in sorted(r["pts"]):
        ce, bpb = r["pts"][step]
        rows.append(dict(run=name, arm=m["arm"], tag=m["tag"], pool=m["pool"],
                         u=m["u"], d_model=m["d_model"], n_layers=m["n_layers"],
                         lr=m["lr"], wd=m["wd"], step=step,
                         epoch=round(step / spe, 3),
                         tokens=step * TOK_PER_STEP,
                         val_ce=ce, hs_bpb=bpb, logs="|".join(r["logs"])))

rows.sort(key=lambda x: (x["arm"], x["tag"], x["lr"], x["wd"], x["step"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} rows, {len(runs)} runs -> {OUT}\n")
groups = defaultdict(list)
for name, r in runs.items():
    m = r["meta"]
    groups[(m["arm"], m["tag"], m["pool"], m["d_model"])].append(name)
print(f"{'arm':16s} {'tag':12s} {'pool':18s} {'d_model':>7s} {'cells':>5s}")
for (arm, tag, pool, dm), names in sorted(groups.items(), key=lambda x: str(x[0])):
    print(f"{arm:16s} {tag:12s} {pool:18s} {str(dm):>7s} {len(names):5d}")
