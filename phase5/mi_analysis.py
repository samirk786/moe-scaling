#!/usr/bin/env python3
# Does any router metric predict where the loss turns?
# Three MoE cells turn at different epochs, so if a metric drives the turn its
# trajectory should differ across cells in the same order.

import csv
from collections import defaultdict

RES = "/scratch/users/samir.kassam/results.csv"
ROUT = "/scratch/users/samir.kassam/router.csv"
SPE = 353

loss = defaultdict(dict)
with open(RES) as f:
    for r in csv.DictReader(f):
        if r["arm"] != "moe32" or not r["val_ce"]:
            continue
        loss[r["run"]][int(r["epoch"])] = float(r["val_ce"])

turn = {}
for run, d in loss.items():
    ep = min(d, key=d.get)
    turn[run] = (ep, d[ep])

print("=" * 70)
print("TURN POINTS")
print("=" * 70)
for run in sorted(turn, key=lambda r: turn[r][0]):
    ep, v = turn[run]
    print(f"  {run:28s} min {v:.3f} at epoch {ep}")

rout = defaultdict(lambda: defaultdict(dict))   # run -> (split,metric) -> step -> value
with open(ROUT) as f:
    for r in csv.DictReader(f):
        if int(r["layer"]) != -1:
            continue
        base = r["run"].rsplit("_j", 1)[0]
        rout[base][(r["split"], r["metric"])][int(r["step"])] = float(r["value"])

rout = {k: v for k, v in rout.items() if k in turn}
runs = sorted(rout, key=lambda r: turn[r][0])
metrics = sorted({m for d in rout.values() for m in d})

print()
print("=" * 70)
print("LAYER-MEAN METRIC vs EPOCH   (runs ordered by turn epoch)")
print("=" * 70)
for split, metric in metrics:
    if split != "heldout":
        continue
    print(f"\n--- heldout/{metric} ---")
    for run in runs:
        series = rout[run].get((split, metric), {})
        if not series:
            continue
        pts = [(s // SPE, v) for s, v in sorted(series.items()) if s % SPE == 0 and s > 0]
        t = turn.get(run, (None,))[0]
        vals = " ".join(f"{v:6.3f}" for _, v in pts[:12])
        print(f"  turn@{t:>2}  {run:26s} {vals}")

print()
print("=" * 70)
print("TRAIN MINUS HELDOUT  (memorization = train pulling ahead)")
print("=" * 70)
for _, metric in [m for m in metrics if m[0] == "heldout"]:
    if ("train", metric) not in metrics:
        continue
    print(f"\n--- {metric} ---")
    for run in runs:
        tr = rout[run].get(("train", metric), {})
        ho = rout[run].get(("heldout", metric), {})
        common = sorted(set(tr) & set(ho))
        pts = [(s // SPE, tr[s] - ho[s]) for s in common if s % SPE == 0 and s > 0]
        if not pts:
            continue
        t = turn.get(run, (None,))[0]
        vals = " ".join(f"{v:+6.3f}" for _, v in pts[:12])
        print(f"  turn@{t:>2}  {run:26s} {vals}")

print()
print("=" * 70)
print("VALUE AT EACH RUN'S OWN TURN EPOCH")
print("=" * 70)
print("if a metric drives the turn, these should agree across cells")
for split, metric in metrics:
    if split != "heldout":
        continue
    row = []
    for run in runs:
        t = turn.get(run, (None,))[0]
        v = rout[run].get((split, metric), {}).get(t * SPE if t else -1)
        row.append(f"{v:.3f}" if v is not None else "  -  ")
    print(f"  {metric:28s} " + "  ".join(row))
print("  " + " " * 28 + "  ".join(f"ep{turn[r][0]:<3}" for r in runs))
