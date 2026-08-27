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
DFF = re.compile(r"feed_forward=FeedForwardConfig\(hidden_size=(\d+)")
STEP = re.compile(r"step=(\d+)/(\d+)")
CE = re.compile(r"val/CE loss=([0-9.]+)")
BPB = re.compile(r"hellaswag \(BPB\)=([0-9.]+)")

# trunk / cosine / moe: arm, optional size-or-pool tag, lr, wd
NAME = re.compile(r"^(cosine_ep16|trunk|moe32)"
                  r"(?:_(n\d+m|u\d+m\d*e?p?|thid\d+m))?"
                  r"_lr([0-9e.\-]+)_wd([0-9.]+)$")

# forks: fork[SUFFIX]_(ep16|sSTEP)_SHAPE_lr..._wd...
# suffix names the arm branched from; ep16 is the original naming for step 4518.
FORK = re.compile(r"^fork(761|30m|C_n761m|C|moe)?_(?:ep16|s(\d+))_([0-9a-z\-]+)"
                  r"_lr([0-9e.\-]+)_wd([0-9.]+)$")
FORK_TAG = {None: "", "761": "n761m", "30m": "u30m20ep",
            "C": "", "C_n761m": "n761m", "moe": "moe32"}


def scan(path):
    head = open(path, errors="ignore").read(200_000)
    m = SAVE.search(head)
    if not m or "smoke" in m.group(1):
        return None
    name = m.group(1)
    mix = MIX.search(head)
    dur = DUR.search(head)
    if not mix or not dur or mix.group(1) not in POOLS:
        return None

    nm = NAME.match(name)
    fk = FORK.match(name)
    if nm:
        arm, tag, lr, wd, fstep, shape = nm.group(1), nm.group(2) or "", \
            nm.group(3), nm.group(4), "", ""
    elif fk:
        arm, tag = "fork", FORK_TAG[fk.group(1)]
        fstep = fk.group(2) or "4518"      # ep16 forks branch from step 4518
        shape, lr, wd = fk.group(3), fk.group(4), fk.group(5)
    else:
        print(f"  UNPARSED NAME: {name}  ({path.split('/')[-1]})")
        return None

    dm = DMODEL.search(head)
    nl = NLAYERS.search(head)
    dff = DFF.search(head)
    return dict(name=name, arm=arm, tag=tag, lr=lr, wd=wd,
                fork_step=fstep, shape=shape,
                pool=mix.group(1), u=POOLS[mix.group(1)],
                total=int(dur.group(1)) // TOK_PER_STEP,
                d_model=int(dm.group(1)) if dm else None,
                n_layers=int(nl.group(1)) if nl else None,
                d_ff=int(dff.group(1)) if dff else None)


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
    pts = curve(p, meta["total"])
    if not pts:
        continue          # crashed before any eval (e.g. the failed 761M forks)
    r = runs[meta["name"]]
    r["meta"] = meta
    r["logs"].append(p.split("/")[-1])
    for step, ce, bpb in pts:
        old = r["pts"].get(step)
        # requeue can replay a step; keep whichever row carries bpb
        if old is None or (old[1] is None and bpb is not None):
            r["pts"][step] = (ce, bpb)

rows = []
for name, r in runs.items():
    m = r["meta"]
    spe = m["u"] / TOK_PER_STEP
    steps = sorted(r["pts"])
    for step in steps:
        ce, bpb = r["pts"][step]
        rows.append(dict(run=name, arm=m["arm"], tag=m["tag"], pool=m["pool"],
                         u=m["u"], d_model=m["d_model"], n_layers=m["n_layers"],
                         d_ff=m["d_ff"], lr=m["lr"], wd=m["wd"],
                         fork_step=m["fork_step"],
                         decay=m["total"] - int(m["fork_step"] or 0), step=step,
                         epoch=round(step / spe, 3),
                         tokens=step * TOK_PER_STEP,
                         # only a fork's final eval is fully annealed
                         annealed=int(m["arm"] == "fork" and step == max(steps)),
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
print(f"{'arm':10s} {'tag':12s} {'pool':18s} {'d_model':>7s} {'cells':>5s}")
for (arm, tag, pool, dm), names in sorted(groups.items(), key=lambda x: str(x[0])):
    print(f"{arm:10s} {tag:12s} {pool:18s} {str(dm):>7s} {len(names):5d}")

print("\nANNEALED FORK ENDPOINTS")
print(f"{'run':40s} {'d_model':>7s} {'epoch':>7s} {'val/CE':>7s}")
for r in sorted(rows, key=lambda x: (x["d_model"] or 0, x["epoch"])):
    if r["annealed"]:
        print(f"{r['run']:40s} {str(r['d_model']):>7s} {r['epoch']:7.2f} {r['val_ce']:7.3f}")
