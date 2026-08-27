#!/usr/bin/env python3
# Turn epoch and post-turn degradation across the size and pool axes.
# Reads results_all.csv. Each (arm, tag) is a 3x3 lr/wd grid; the turn is read
# at each grid's own best cell, with the matched lr1e-3/wd0.8 cell alongside.

import csv
from collections import defaultdict

RES = "/scratch/users/samir.kassam/results_all.csv"
TOK = 262144
# non-embedding-inclusive counts from the repo's own convention
PARAMS = {2048: 1_484_916_736, 1536: 761_341_440}

cells = defaultdict(dict)
meta = {}
with open(RES) as f:
    for r in csv.DictReader(f):
        if not r["val_ce"]:
            continue
        k = (r["arm"], r["tag"], r["lr"], r["wd"])
        cells[k][float(r["epoch"])] = float(r["val_ce"])
        meta[k] = (int(r["u"]), int(r["d_model"]), int(r["step"]) / float(r["epoch"]))


def turn(c, window=4.0):
    ep = min(c, key=c.get)
    last = max(c)
    later = [e for e in c if e > ep]
    if not later:
        return ep, c[ep], None, None, last, True
    end = min(later, key=lambda e: abs(e - (ep + window)))
    return ep, c[ep], c[end] - c[ep], end, last, False


print("=" * 96)
print("PER-GRID: every cell, turn epoch and degradation over the ~4 epochs after")
print("=" * 96)
grids = defaultdict(list)
for k in cells:
    grids[(k[0], k[1])].append(k)

order = sorted(grids, key=lambda g: (g[0], g[1]))
best = {}
for g in order:
    u, dm, spe = meta[grids[g][0]]
    print(f"\n--- {g[0]}{('/' + g[1]) if g[1] else '':<12}  U={u:,}  d_model={dm}  "
          f"{PARAMS[dm]/u:.1f} params/token  ({spe:.1f} steps/epoch)")
    print(f"    {'lr':>6} {'wd':>5} {'min':>7} {'@ep':>6} {'+4ep':>8} {'(to ep)':>8} {'last':>6}")
    rows = []
    for k in sorted(grids[g], key=lambda k: (float(k[2]), float(k[3]))):
        ep, lo, deg, end, last, cens = turn(cells[k])
        rows.append((k, lo, ep, deg, end, last, cens))
        d = "  CENSORED" if cens else f"{deg:+8.3f} {end:8.2f}"
        print(f"    {k[2]:>6} {k[3]:>5} {lo:7.3f} {ep:6.2f} {d} {last:6.2f}")
    b = min(rows, key=lambda r: r[1])
    best[g] = b
    print(f"    best cell: lr {b[0][2]} wd {b[0][3]}  min {b[1]:.3f} @ep {b[2]:.2f}")

print()
print("=" * 96)
print("CROSS-GRID SUMMARY")
print("=" * 96)
hdr = (f"{'grid':22s} {'p/tok':>7} {'min':>7} {'turn':>6} {'tok@turn':>10} "
       f"{'+4ep':>8} {'per-ep':>8}")
for label, pick in (("AT EACH GRID'S BEST CELL", None), ("AT MATCHED lr1e-3 wd0.8", ("1e-3", "0.8"))):
    print(f"\n{label}")
    print(hdr)
    for g in order:
        if pick:
            k = (g[0], g[1], pick[0], pick[1])
            if k not in cells:
                print(f"{g[0] + '/' + g[1]:22s}   (cell missing)")
                continue
            ep, lo, deg, end, last, cens = turn(cells[k])
        else:
            k, lo, ep, deg, end, last, cens = best[g]
        u, dm, spe = meta[k]
        tok = ep * u
        rate = "" if cens else f"{deg/(end-ep):+8.3f}"
        dcol = "CENSORED" if cens else f"{deg:+8.3f}"
        print(f"{(g[0] + ('/' + g[1] if g[1] else '')):22s} {PARAMS[dm]/u:7.1f} {lo:7.3f} "
              f"{ep:6.2f} {tok/1e6:9.0f}M {dcol:>8} {rate:>8}")
