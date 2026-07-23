#!/usr/bin/env python3
"""
Unit test for the router probe metric definitions. No model, no GPU, no repo
imports beyond the callback module itself.

The point is the three-regime table: healthy specialization, collapse, and
indecision. Conditional entropy alone cannot tell specialization from collapse,
marginal entropy alone cannot tell specialization from indecision, and the
mutual information separates all three. If a future edit breaks that, these
assertions fail before any GPU time is spent.

Run:  python test_router_probe_math.py
"""

import numpy as np
import torch

from olmo_core.train.callbacks.router_probe import RouterProbeCallback

N_EXPERTS = 8
TOP_K = 1
N_TOKENS = 4096
STATS = RouterProbeCallback._layer_stats


def summarize(scores: torch.Tensor, indices: torch.Tensor):
    """Mirror of the accumulation in _run_split, on a single batch."""
    p_bar = scores.sum(dim=0).double()
    p_bar = p_bar / p_bar.sum()
    h_cond = float(RouterProbeCallback._entropy(scores, dim=-1).mean())
    pair = scores.topk(2, dim=-1).values
    return STATS(
        p_bar=p_bar,
        h_cond=h_cond,
        counts=torch.bincount(indices.view(-1), minlength=N_EXPERTS),
        margins=pair[:, 0] - pair[:, 1],
        n_tokens=N_TOKENS,
        top_k=TOP_K,
        dead_expert_frac=0.1,
    )


def one_hot(assign: torch.Tensor) -> torch.Tensor:
    scores = torch.full((N_TOKENS, N_EXPERTS), 1e-6)
    scores[torch.arange(N_TOKENS), assign] = 1.0
    return scores / scores.sum(dim=-1, keepdim=True)


def regimes():
    tok = torch.arange(N_TOKENS)

    # A: healthy specialization. Each token goes decisively to one expert, and
    # the experts are used evenly.
    assign = tok % N_EXPERTS
    a = summarize(one_hot(assign), assign[:, None])

    # B: collapse. Every token goes decisively to expert 3.
    assign = torch.full((N_TOKENS,), 3)
    b = summarize(one_hot(assign), assign[:, None])

    # C: indecision. Gates are uniform, so routing carries no information even
    # though the load looks perfectly balanced.
    scores = torch.full((N_TOKENS, N_EXPERTS), 1.0 / N_EXPERTS)
    c = summarize(scores, (tok % N_EXPERTS)[:, None])

    return a, b, c


def close(x, y, tol=0.02):
    return abs(x - y) < tol


def test_regimes():
    a, b, c = regimes()

    # the blind spots, asserted as blind spots
    assert close(a["entropy_conditional"], b["entropy_conditional"]), "A/B should tie on conditional"
    assert close(a["entropy_marginal"], c["entropy_marginal"]), "A/C should tie on marginal"

    # what each metric does see
    assert close(a["entropy_marginal"], 1.0)
    assert close(b["entropy_marginal"], 0.0)
    assert close(a["entropy_conditional"], 0.0)
    assert close(c["entropy_conditional"], 1.0)

    # mutual information separates all three
    assert close(a["mutual_information"], 1.0)
    assert close(b["mutual_information"], 0.0)
    assert close(c["mutual_information"], 0.0)
    assert a["mutual_information"] - max(b["mutual_information"], c["mutual_information"]) > 0.9

    # effective experts is the readable version of marginal entropy
    assert close(a["effective_experts"], 8.0, tol=0.1)
    assert close(b["effective_experts"], 1.0, tol=0.1)

    # dead experts: relative threshold, and B is the only regime with any
    assert a["dead_expert_frac"] == 0.0
    assert close(b["dead_expert_frac"], 7 / 8)
    assert c["dead_expert_frac"] == 0.0

    # margin distinguishes locked-in from coincidentally-stable
    assert a["margin_mean"] > 0.9 and b["margin_mean"] > 0.9
    assert close(c["margin_mean"], 0.0)
    print("regimes OK")


def test_normalization_across_expert_counts():
    """A fully collapsed router reads 0 and a uniform one reads 1 at every N."""
    for n in (16, 32, 64, 128):
        uniform = torch.full((n,), 1.0 / n, dtype=torch.float64)
        collapsed = torch.zeros(n, dtype=torch.float64)
        collapsed[0] = 1.0
        common = dict(
            h_cond=0.0,
            counts=torch.zeros(n, dtype=torch.long),
            margins=torch.zeros(4),
            n_tokens=N_TOKENS,
            top_k=8,
            dead_expert_frac=0.1,
        )
        assert close(STATS(p_bar=uniform, **common)["entropy_marginal"], 1.0)
        assert close(STATS(p_bar=collapsed, **common)["entropy_marginal"], 0.0)
    print("normalization OK")


def test_consistency():
    rng = np.random.default_rng(0)
    cur = rng.integers(0, N_EXPERTS, size=(3, 256, 1 + 4)).astype(np.int16)

    same = RouterProbeCallback._consistency(cur, cur)
    assert close(same["top1_agreement"], 1.0)
    assert close(same["topk_jaccard"], 1.0)
    assert close(same["churn_rate"], 0.0)

    # disjoint expert pools: no overlap possible
    lo = rng.integers(0, 4, size=(3, 256, 1 + 4)).astype(np.int16)
    hi = rng.integers(4, 8, size=(3, 256, 1 + 4)).astype(np.int16)
    none = RouterProbeCallback._consistency(lo, hi)
    assert none["top1_agreement"] == 0.0
    assert none["topk_jaccard"] == 0.0
    assert close(none["churn_rate"], 1.0)
    print("consistency OK")


if __name__ == "__main__":
    test_regimes()
    test_normalization_across_expert_counts()
    test_consistency()
    print("all router probe math checks passed")
