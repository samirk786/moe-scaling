#!/usr/bin/env python3
# Router metrics for the moe32 trunks, read against the loss curve.
# Question: does the faster degradation of the MoE arm under repetition show up
# in routing at all, or is routing healthy the whole way through?
#
# Collapse and freezing predict different things and must not be conflated:
#   collapse  -> effective_experts falls, dead_expert_frac rises, MI -> 0
#   freezing  -> capacity holds, but TRAIN consistency pulls ahead of HELDOUT
# Read the train-heldout GAP, never the level: under collapse the argmax is
# near token-independent, so it is trivially stable on both splits.

import csv
import math
from collections import defaultdict

ROUTER = "/scratch/users/samir.kassam/router.csv"
RESULTS = "/scratch/users/samir.kassam/results_all.csv"
SPE = 352.75          # 92.47M pool / 262144
N_EXPERTS = 32
AGG = "-1"            # layer-mean aggregate

R = defaultdict(dict)          # (run, step, split, layer, metric) -> value
runs = set()
for row in csv.DictReader(open(ROUTER)):
    if not row["run"].startswith("moe32"):
        continue
    runs.add(row["run"])
    R[(row["run"], int(row["step"]), row["split"], row["layer"])][row["metric"]] = float(row["value"])

# cell label from the run name, minus the job id
def cell(run):
    return "_".join(run.split("_")[1:3])

loss = defaultdict(dict)       # cell -> step -> val_ce
for row in csv.DictReader(open(RESULTS)):
    if row["arm"] == "moe32" and row["val_ce"]:
        loss[f"lr{row['lr']}_wd{row['wd']}"][int(row["step"])] = float(row["val_ce"])

print("=" * 94)
print("ROUTING vs LOSS, moe32 trunks, layer-mean, heldout split")
print("=" * 94)
print("eff_exp = exp(H_marginal), ceiling 32.  MI separates specialization from collapse.")
print("dloss = val/CE minus this cell's own minimum.\n")

for run in sorted(runs):
    c = cell(run)
    steps = sorted({s for (r, s, sp, l) in R if r == run})
    lo = min(loss[c].values()) if loss[c] else None
    print(f"--- {run}")
    print(f"    {'ep':>5} {'val/CE':>7} {'dloss':>7} {'eff_exp':>8} {'MI':>7} "
          f"{'dead':>6} {'maxload':>8} {'margin':>7}")
    for s in steps:
        h = R.get((run, s, "heldout", AGG), {})
        ce = loss[c].get(s)
        d = f"{ce - lo:+7.3f}" if (ce is not None and lo is not None) else "      -"
        print(f"    {s/SPE:5.1f} {ce if ce else float('nan'):7.3f} {d} "
              f"{h.get('effective_experts', float('nan')):8.2f} "
              f"{h.get('mutual_information', float('nan')):7.3f} "
              f"{h.get('dead_expert_frac', float('nan')):6.3f} "
              f"{h.get('max_load_frac', float('nan')):8.3f} "
              f"{h.get('margin_mean', float('nan')):7.3f}")
    print()

print("=" * 94)
print("MEMORIZATION TEST: train minus heldout consistency")
print("=" * 94)
print("Rising train consistency with a WIDENING gap = routing memorizes the repeated data.")
print("Both rising together with a flat gap = ordinary convergence, not memorization.")
print("Epoch 1 is uninterpretable: nothing has been repeated yet.\n")

for run in sorted(runs):
    steps = sorted({s for (r, s, sp, l) in R if r == run})
    print(f"--- {run}")
    print(f"    {'ep':>5} {'top1 tr':>8} {'top1 ho':>8} {'gap':>7} "
          f"{'jac tr':>7} {'jac ho':>7} {'gap':>7} {'churn ho':>9}")
    for s in steps:
        t = R.get((run, s, "train", AGG), {})
        h = R.get((run, s, "heldout", AGG), {})
        if not t or not h:
            continue
        t1t, t1h = t.get("top1_agreement"), h.get("top1_agreement")
        jt, jh = t.get("topk_jaccard"), h.get("topk_jaccard")
        if None in (t1t, t1h, jt, jh):
            continue
        print(f"    {s/SPE:5.1f} {t1t:8.3f} {t1h:8.3f} {t1t - t1h:+7.3f} "
              f"{jt:7.3f} {jh:7.3f} {jt - jh:+7.3f} "
              f"{h.get('churn_rate', float('nan')):9.3f}")
    print()

print("=" * 94)
print("DEPTH PROFILE: effective experts by layer, first vs last probe (heldout)")
print("=" * 94)
for run in sorted(runs):
    steps = sorted({s for (r, s, sp, l) in R if r == run})
    first, last = steps[0], steps[-1]
    layers = sorted({int(l) for (r, s, sp, l) in R if r == run and l != AGG})
    a = [R.get((run, first, "heldout", str(l)), {}).get("effective_experts") for l in layers]
    b = [R.get((run, last, "heldout", str(l)), {}).get("effective_experts") for l in layers]
    fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"
    print(f"--- {run}")
    print(f"    ep {first/SPE:4.1f}: " + " ".join(fmt(v) for v in a))
    print(f"    ep {last/SPE:4.1f}: " + " ".join(fmt(v) for v in b))
    live = [v for v in b if v is not None]
    if live:
        print(f"    last probe: min {min(live):.2f}  max {max(live):.2f}  "
              f"layers under 2.0: {sum(v < 2.0 for v in live)}/{len(live)}")
    print()
