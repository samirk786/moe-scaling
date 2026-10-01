# Data-constrained scaling: dense vs. MoE under repeated data

Research codebase for studying whether mixture-of-experts language models keep their
advantage over dense models when the pool of unique training data is fixed and extra
training tokens are repetitions. Built on [allenai/EMO](https://github.com/allenai/EMO)
(OLMo-core). Everything under "What I built" is my work on top of that base.

**Status:** paused, Sept 2026. Infrastructure and validation phases complete; full sweep not yet run.

## The question

Fix a unique-token pool U and repeat it R times. Compare dense and MoE models at matched
active parameters (embeddings included), varying the number of routed experts while holding
top-k and expert granularity fixed. Measure held-out loss and Hellaswag BPB per epoch, and
instrument the router to see what happens to expert usage as data repeats.

## What I built

- **Scheduler suite** (`src/scripts/train/olmo2-1B.py`). Constant, cosine, WSD, and WSDS
  (cyclic warmup-stable-decay), selectable from the CLI. WSDS periods align to the epoch
  length, so one N-epoch run yields a fully annealed checkpoint at every epoch 1..N, where
  cosine needs a separate run per target epoch. Save and eval intervals snap to period
  boundaries.
- **Dense-to-MoE graft.** MoE arms are built by passing an MoE FFN config into the same dense
  preset, so dense and MoE arms share an identical backbone by construction and differ only
  in the FFN. Router config is set explicitly (the library default routes top-1).
- **Router probe** (`src/olmo_core/train/callbacks/router_probe.py`). Eval-mode callback that
  hooks every router, recomputes the full routing distribution, and reports marginal entropy
  (balance), conditional entropy (decisiveness), their mutual information, effective experts,
  dead-expert fraction, decision margins, and routing consistency between probes on held-out
  vs. training data. Normalized so arms with different expert counts share an axis;
  preemption-safe. Validated against a synthetic three-regime test, against upstream's
  built-in metric, and by per-layer alignment. Surfaced a normalization inconsistency in the
  upstream entropy metric.
- **Data pipeline.** Randomly sampled DCLM subsets tokenized with the dolma2 tokenizer; a
  seeded, disjoint ~4.2M-token held-out validation set.
- **Cluster tooling** (`phase4/`, `phase5/`). Slurm array scripts for single-GPU H200 runs on
  a preemptible queue: requeue-resume from checkpoints, per-job port assignment, append-mode
  logs, and an LR x WD grid launcher.
- **Planning tools** (`phase5/capacity.py`, `phase5/check_moe_params.py`). Parameter and
  memory calculator calibrated to the repo's exact parameter count, an FSDP sharding table,
  and a GPU-hour budget model built from measured step times.
- **Experiments run.** WSDS vs. cosine scheduler comparison (dense 1B, up to 8 epochs);
  run-to-run nondeterminism characterization; router probe sweeps over load-balance weight
  and learning rate.

## Layout

| Path | What's there |
|---|---|
| `src/scripts/train/olmo2-1B.py` | Training entry point with the scheduler suite and MoE graft |
| `src/olmo_core/train/callbacks/router_probe.py` | Router probe callback |
| `phase4/` | Slurm scripts for the scheduler and nondeterminism experiments |
| `phase5/` | Probe tests, planning calculators, data prep, result collectors |
| `tools/` | Helper scripts |
| `archive/` | Raw run logs, collected metrics, and data snapshots |

Everything else is the upstream EMO / OLMo-core codebase.

## Acknowledgments

Project direction and advising: Prof. Sewon Min (UC Berkeley, BAIR), summer 2026.
Built on [allenai/EMO](https://github.com/allenai/EMO) and OLMo-core; upstream Apache-2.0
license retained. Findings have not been written up or published.
