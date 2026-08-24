#!/usr/bin/env python3
# Dense vs MoE at matched lr/wd: where does each turn, and how fast does it
# degrade after? Both arms are constant-LR trunks, so no scheduler confound.

import csv
from collections import defaultdict

RES = "/scratch/users/samir.kassam/results.csv"
U = 92_469_693
P = {"trunk": (1_484_916_736, 1_484_916_736), "moe32": (3_901_884_416, 1_485_965_312)}

curves = defaultdict(dict)
with open(RES) as f:
    for r in csv.DictReader(f):
        if r["arm"] not in P or not r["val_ce"]:
            continue
        curves[(r["arm"], r["lr"], r["wd"])][int(r["epoch"])] = float(r["val_ce"])

print("=" * 78)
print("PARAMS PER UNIQUE TOKEN")
print("=" * 78)
for arm, (tot, act) in P.items():
    print(f"  {arm:8s} total {tot/1e9:5.2f}B -> {tot/U:5.1f} params/token   "
          f"active {act/1e9:5.2f}B -> {act/U:5.1f}")

def summarise(c):
    ep = min(c, key=c.get)
    last = max(c)
    # degradation over the 4 epochs after the minimum, if available
    later = [e for e in c if e > ep]
    if not later:
        return ep, c[ep], 0.0, ep
    end = min(later, key=lambda e: abs(e - (ep + 4)))
    return ep, c[ep], c[end] - c[ep], end

print()
print("=" * 78)
print("MATCHED PAIRS  (same lr, same wd, both constant-LR trunks)")
print("=" * 78)
print(f"{'lr':>6} {'wd':>5} {'arm':>7} {'min':>7} {'@ep':>4} {'+4ep':>8} {'ep12':>7}")
cells = sorted({(lr, wd) for arm, lr, wd in curves if arm == "moe32"})
for lr, wd in cells:
    for arm in ("trunk", "moe32"):
        c = curves.get((arm, lr, wd))
        if not c:
            print(f"{lr:>6} {wd:>5} {arm:>7}   (missing)")
            continue
        ep, best, deg, end = summarise(c)
        e12 = c.get(12, c[max(c)])
        print(f"{lr:>6} {wd:>5} {arm:>7} {best:7.3f} {ep:4d} {deg:+8.3f} {e12:7.3f}")
    print()

print("=" * 78)
print("DENSE GRID: turn epoch vs hyperparameters  (params/token is CONSTANT here)")
print("=" * 78)
print("if capacity drove the turn, these would all be the same. they are not.")
rows = []
for (arm, lr, wd), c in curves.items():
    if arm != "trunk":
        continue
    ep, best, deg, _ = summarise(c)
    rows.append((lr, wd, ep, best, deg))
for lr, wd, ep, best, deg in sorted(rows, key=lambda r: r[2]):
    print(f"  lr {lr:>5}  wd {wd:>4}   turn@{ep:<3} min {best:.3f}   +4ep {deg:+.3f}")
eps = [r[2] for r in rows]
print(f"\n  turn epoch ranges {min(eps)} to {max(eps)} at fixed 16.1 params/token")
