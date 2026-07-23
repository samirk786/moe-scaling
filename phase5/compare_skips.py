#!/usr/bin/env python3
"""
Compare optimizer skip traces between runs, to test whether SkipStepAdamW is the
mechanism behind the ~0.035 nat run-to-run floor.

SkipStepAdamW drops an update when the loss falls outside a rolling window, so a
last-bit difference on some step can flip a borderline skip decision, and a
skipped step is a whole missing update rather than a small perturbation. If two
nominally identical runs skip on different steps, that is the mechanism. If the
traces match, the floor is ordinary GPU nondeterminism.

Usage:
    python compare_skips.py --runs nondet_wsds2ep p4_wsds_8ep --max-step 706
"""

import argparse

import wandb

KEY = "optim/step skipped"


def trace(api, entity, project, name, max_step):
    runs = [r for r in api.runs(f"{entity}/{project}") if r.name == name]
    if not runs:
        raise SystemExit(f"no run named {name}")
    # a fixed --trainer.callbacks.wandb.name means resubmits pile up under one
    # display name, so take the one that actually got the furthest.
    runs.sort(key=lambda r: r.summary.get("_step", 0) or 0, reverse=True)
    run = runs[0]
    print(f"{name}: id={run.id} state={run.state} last_step={run.summary.get('_step')}"
          f"  ({len(runs)} runs share this name)")

    steps = {}
    for row in run.scan_history(keys=["_step", KEY]):
        s, v = row.get("_step"), row.get(KEY)
        if s is None or v is None or s > max_step:
            continue
        steps[s] = float(v)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="samirkassam")
    parser.add_argument("--project", default="dclm-scaling")
    parser.add_argument("--runs", nargs=2, required=True)
    parser.add_argument("--max-step", type=int, default=706)
    args = parser.parse_args()

    api = wandb.Api()
    a = trace(api, args.entity, args.project, args.runs[0], args.max_step)
    b = trace(api, args.entity, args.project, args.runs[1], args.max_step)

    for name, t in ((args.runs[0], a), (args.runs[1], b)):
        nonzero = {s: v for s, v in t.items() if v > 0}
        print(f"\n{name}: {len(t)} logged points, sum={sum(t.values()):.3f}, "
              f"{len(nonzero)} nonzero")
        if nonzero:
            print(f"  steps: {sorted(nonzero)[:40]}")

    shared = sorted(set(a) & set(b))
    diffs = [s for s in shared if a[s] != b[s]]
    print(f"\n{len(shared)} shared logged steps, {len(diffs)} disagree")
    if diffs:
        print(f"  first disagreements: {diffs[:20]}")
        print("  => skip decisions diverge; SkipStepAdamW is a live mechanism.")
    elif sum(a.values()) == sum(b.values()) == 0:
        print("  => no skips in either run. Floor is not skip-driven.")
    else:
        print("  => traces identical. Floor is not skip-driven.")


if __name__ == "__main__":
    main()
