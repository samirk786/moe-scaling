#!/usr/bin/env python3
# Pull router probe metrics out of wandb into a long CSV.
# The probe records via record_metric, so these never reach the slurm logs.
# Long format: one row per (run, step, split, layer, metric).

import csv
import os
import re
import sys

import wandb

ENTITY = "samirkassam"
PROJECT = "dclm-scaling"
OUT = "/scratch/users/samir.kassam/router.csv"

KEY = re.compile(r"^router_probe/(train|heldout)/(?:layer(\d+)/)?(.+)$")

api = wandb.Api()
runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if "moe" in r.tags]
if not runs:
    sys.exit("no runs tagged moe")

rows = []
for run in runs:
    keys = [k for k in run.summary.keys() if k.startswith("router_probe/")]
    if not keys:
        print(f"  {run.name}: no probe keys, skipping")
        continue
    hist = run.history(keys=keys, samples=10000, pandas=False)
    n = 0
    for rec in hist:
        step = rec.get("_step")
        for k, v in rec.items():
            m = KEY.match(k)
            if not m or v is None:
                continue
            split, layer, metric = m.groups()
            rows.append(dict(run=run.name, step=step, split=split,
                             layer=int(layer) if layer else -1,
                             metric=metric, value=v))
            n += 1
    print(f"  {run.name}: {n} values over {len(hist)} probes")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "step", "split", "layer", "metric", "value"])
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} rows -> {OUT}")
print("layer=-1 is the layer-mean aggregate")
mets = sorted({r["metric"] for r in rows})
print(f"{len(mets)} metrics: {', '.join(mets)}")
