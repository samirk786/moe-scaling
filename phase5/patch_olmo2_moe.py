#!/usr/bin/env python3
"""
Apply the MoE graft branch and the router probe wiring to olmo2-1B.py, and export
RouterProbeCallback from the callbacks package.

Every edit is anchored on an exact string. If any anchor is missing or ambiguous
the script writes nothing and reports which one failed, so a partial patch is not
a reachable state. Stdlib only.

Usage:
    python patch_olmo2_moe.py --repo /accounts/projects/berkeleynlp/samir.kassam/EMO
    python patch_olmo2_moe.py --repo ... --apply
"""

import argparse
import sys
from pathlib import Path

TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"
CALLBACKS_INIT = "src/olmo_core/train/callbacks/__init__.py"

EDITS = {}

EDITS[TRAIN_SCRIPT] = [
    (
        "transformer import",
        "from olmo_core.nn.transformer import TransformerConfig\n",
        "from olmo_core.nn.moe import (\n"
        "    MoEConfig,\n"
        "    MoERouterConfig,\n"
        "    MoERouterGatingFunction,\n"
        "    MoEType,\n"
        ")\n"
        "from olmo_core.nn.transformer import (\n"
        "    TransformerBlockType,\n"
        "    TransformerConfig,\n"
        "    TransformerType,\n"
        ")\n",
    ),
    (
        "callbacks import",
        "    ProfilerCallback,\n    WandBCallback,\n)\n",
        "    ProfilerCallback,\n    RouterProbeCallback,\n    WandBCallback,\n)\n",
    ),
    (
        "moe constants",
        "GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH  # 64 seqs, matches infinite-compute batch=64\n",
        "GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH  # 64 seqs, matches infinite-compute batch=64\n"
        "\n"
        "MOE_TOP_K = 8\n"
        "MOE_GRANULARITY = 8  # expert width = dense d_ff / G, so top_k experts match the dense FFN\n",
    ),
    (
        "build_model_config",
        "def build_config(opts, overrides: List[str]) -> ExperimentConfig:\n",
        '''def build_model_config(opts, tokenizer_config) -> TransformerConfig:
    vocab_size = tokenizer_config.padded_vocab_size()
    dense = TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size)
    if not opts.num_experts:
        return dense

    # read the built hidden size rather than recomputing it: llama_like truncates twice
    # then rounds up to a multiple of 256, so 4 * d_model is an output of that pipeline
    # and not a rule it follows.
    d_ff = dense.block.feed_forward.hidden_size

    # same backbone as the dense arm, MoE feed-forward. Routing through olmo2_1B_v2
    # rather than llama_like_moe inherits qk_norm, rope_theta, layer_norm_eps and
    # hidden_size_multiplier instead of restating them, which is what keeps the arms
    # differing in the feed-forward and nothing else.
    return TransformerConfig.olmo2_1B_v2(
        vocab_size=vocab_size,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_reordered_norm,
        feed_forward_moe=MoEConfig(
            name=MoEType.dropless,
            num_experts=opts.num_experts,
            hidden_size=d_ff // MOE_GRANULARITY,
            router=MoERouterConfig(
                top_k=MOE_TOP_K,
                gating_function=MoERouterGatingFunction.softmax,
                jitter_eps=None,
                normalize_expert_weights=None,
                bias_gamma=None,
                uniform_expert_assignment=False,
            ),
            # total across layers: MoEBase divides by n_layers when scale_loss_by_num_layers
            # is set, so holding this fixed keeps the aux pressure constant across rungs.
            lb_loss_weight=opts.lb_loss_weight,
            z_loss_weight=0.001,
        ),
    )


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
''',
    ),
    (
        "model config call",
        "    model_config = TransformerConfig.olmo2_1B_v2(\n"
        "        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128\n"
        "    )\n",
        "    model_config = build_model_config(opts, tokenizer_config)\n",
    ),
    (
        "router probe callback",
        "    config = ExperimentConfig(\n        model=model_config,\n",
        "    # steps_per_epoch is 0 without --unique-tokens, and interval 0 divides by zero\n"
        "    # in post_step. Kept out of the chain above because the chain has no conditional form.\n"
        "    if opts.router_probe and steps_per_epoch > 0:\n"
        "        trainer_config = trainer_config.with_callback(\n"
        '            "router_probe",\n'
        "            RouterProbeCallback(\n"
        "                probe_file=opts.router_probe,\n"
        '                dump_dir=f"{save_folder}/router_probes",\n'
        "                interval=steps_per_epoch,\n"
        "            ),\n"
        "        )\n"
        "\n"
        "    config = ExperimentConfig(\n        model=model_config,\n",
    ),
    (
        "argparse options",
        '    parser.add_argument("run_name", type=str, help="""The name of the run.""")\n',
        '    parser.add_argument("run_name", type=str, help="""The name of the run.""")\n'
        "    parser.add_argument(\n"
        '        "--num-experts",\n'
        "        type=int,\n"
        "        default=0,\n"
        '        help="""Number of routed experts. 0 builds the dense arm.""",\n'
        "    )\n"
        "    parser.add_argument(\n"
        '        "--lb-loss-weight",\n'
        "        type=float,\n"
        "        default=0.01,\n"
        '        help="""MoE load-balancing loss weight, as a total across layers.""",\n'
        "    )\n"
        "    parser.add_argument(\n"
        '        "--router-probe",\n'
        "        type=str,\n"
        '        help="""Path to the router probe .npz. Enables router instrumentation.""",\n'
        "    )\n",
    ),
]

EDITS[CALLBACKS_INIT] = [
    (
        "router probe import",
        "from .wandb import WandBCallback\n",
        "from .router_probe import RouterProbeCallback\nfrom .wandb import WandBCallback\n",
    ),
    (
        "router probe __all__",
        '    "WandBCallback",\n',
        '    "RouterProbeCallback",\n    "WandBCallback",\n',
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    staged = {}
    failures = []

    for rel, edits in EDITS.items():
        path = repo / rel
        if not path.exists():
            failures.append(f"{rel}: file not found")
            continue
        text = path.read_text()
        for name, old, new in edits:
            count = text.count(old)
            if count == 0:
                failures.append(f"{rel}: anchor not found: {name}")
            elif count > 1:
                failures.append(f"{rel}: anchor matches {count} times, ambiguous: {name}")
            elif new in text:
                failures.append(f"{rel}: already patched: {name}")
            else:
                text = text.replace(old, new, 1)
        staged[path] = text

    if failures:
        print("NO CHANGES WRITTEN")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)

    for rel, edits in EDITS.items():
        for name, _, _ in edits:
            print(f"  ok  {rel}: {name}")

    if not args.apply:
        print("\nall anchors resolved. re-run with --apply to write.")
        return

    for path, text in staged.items():
        path.write_text(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
