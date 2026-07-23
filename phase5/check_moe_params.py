#!/usr/bin/env python3
"""
Param-count regression check. Zero GPU.

Two separate things are checked and they are not the same configuration:

1. Released reproduction. The 2.52x and 8.60x figures were calibrated with the
   released model's one shared expert, so reproducing them requires it. This is
   the accounting unit test, not a ladder constraint.

2. Ladder arms. No shared expert, so active FFN capacity is exactly top_k
   experts. The whole-model ratio is reported drift, not a target.
"""

from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoERouterGatingFunction, MoEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig, TransformerType

VOCAB = 100352
TOP_K = 8
GRANULARITY = 8


def moe_config(num_experts, expert_hidden, shared_hidden=None):
    return TransformerConfig.olmo2_1B_v2(
        vocab_size=VOCAB,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_reordered_norm,
        feed_forward_moe=MoEConfig(
            name=MoEType.dropless,
            num_experts=num_experts,
            hidden_size=expert_hidden,
            router=MoERouterConfig(
                top_k=TOP_K,
                gating_function=MoERouterGatingFunction.softmax,
                jitter_eps=None,
                normalize_expert_weights=None,
                bias_gamma=None,
                uniform_expert_assignment=False,
            ),
            shared_mlp=None if shared_hidden is None else FeedForwardConfig(hidden_size=shared_hidden),
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
        ),
    )


def main():
    dense = TransformerConfig.olmo2_1B_v2(vocab_size=VOCAB)
    d_ff = dense.block.feed_forward.hidden_size
    print(f"dense: total={dense.num_params:,}  d_model={dense.d_model}  d_ff={d_ff}")
    assert dense.num_params == 1_484_916_736, "dense count drifted from the Phase 4 anchor"
    assert d_ff == 4 * dense.d_model, "d_ff is no longer 4*d_model; moe_ladder.py assumes it"
    print("  dense anchor OK\n")

    print("released reproduction (expert width 1024, 1 shared expert):")
    for n, target in ((32, 2.52), (128, 8.60)):
        c = moe_config(n, 1024, shared_hidden=1024)
        ratio = c.num_params / c.num_active_params
        flag = "OK" if abs(ratio - target) < 0.02 else "MISMATCH"
        print(
            f"  N={n:3d}  total={c.num_params/1e9:6.2f}B  active={c.num_active_params/1e9:.2f}B"
            f"  ratio={ratio:.2f} (target {target})  {flag}"
        )

    print(f"\nladder arms (expert width d_ff/{GRANULARITY} = {d_ff // GRANULARITY}, no shared expert):")
    for n in (16, 32, 64, 128):
        c = moe_config(n, d_ff // GRANULARITY)
        ratio = c.num_params / c.num_active_params
        active_gap = c.num_active_params - dense.num_params
        print(
            f"  N={n:3d}  total={c.num_params/1e9:6.2f}B  active={c.num_active_params/1e9:.2f}B"
            f"  ratio={ratio:5.2f}  active-vs-dense={active_gap/1e6:+.1f}M"
        )

    # iso-active: top_k experts at d_ff/G must reproduce the dense FFN, so active
    # params should differ from dense only by the router. A large gap means top_k
    # or the expert width is wrong, which is the failure this file exists to catch.
    c = moe_config(128, d_ff // GRANULARITY)
    router_params = 128 * dense.d_model * dense.n_layers
    gap = c.num_active_params - dense.num_params
    print(f"\niso-active check at N=128: gap={gap:,}  expected ~router={router_params:,}")
    assert abs(gap - router_params) < 1_000_000, "active params are not iso-active with dense"
    print("  iso-active OK")


if __name__ == "__main__":
    main()
