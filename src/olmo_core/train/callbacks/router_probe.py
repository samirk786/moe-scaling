import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from olmo_core.nn.moe.router import MoERouter, MoERouterGatingFunction

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class RouterProbeCallback(Callback):
    """
    Router diagnostics on fixed probe batches, read only at annealed points.

    Separates the two ways sparsity can stop paying off under data repetition:
    collapse (usable expert count shrinks, so the model drifts toward a smaller
    dense one) and freezing (expert count holds but routing locks onto the
    repeated training tokens). Marginal and conditional entropy are reported
    separately because either one alone is blind to one of the two.

    The router's own ``compute_metrics`` is deliberately not reused: it is
    token-level entropy only, gated on ``self.training``, accumulated over
    training batches, and summed across layers.

    No-op on dense runs, so the same launch script serves every arm.
    """

    probe_file: Optional[str] = None
    dump_dir: Optional[str] = None
    interval: int = 353
    micro_batch_size: int = 4
    dead_expert_frac: float = 0.1
    probe_on_start: bool = True

    # NOTE: omegaconf can't use these annotations, same as Callback._trainer. They are
    # plain class attributes, not dataclass fields, so the config walk skips them.
    #  _probes: Dict[str, torch.Tensor] = {}
    #  _prev_step: int = -1
    _probes = None
    _prev_step = -1

    def state_dict(self) -> Dict[str, Any]:
        # only the pointer: the arrays themselves are on disk so a requeue in the
        # middle of an epoch does not blank out the next consistency reading.
        return {"prev_step": self._prev_step}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._prev_step = state_dict.get("prev_step", -1)

    def pre_train(self):
        if self.probe_file is None or self.dump_dir is None:
            raise ValueError("RouterProbeCallback needs probe_file and dump_dir")

        self._probes = {}
        blob = np.load(self.probe_file)
        for split in ("heldout", "train"):
            self._probes[split] = torch.from_numpy(blob[split].astype(np.int64))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

        # step 0 is the random-init reference every later consistency delta is read against.
        if self.probe_on_start and self.step == 0:
            self._probe()

    def post_step(self):
        if self.step % self.interval == 0:
            self._probe()

    def _probe(self):
        model = self.trainer.train_module.model
        routers: List[Tuple[str, MoERouter]] = [
            (name, mod) for name, mod in model.named_modules() if isinstance(mod, MoERouter)
        ]
        if not routers:
            return

        dumps: Dict[str, np.ndarray] = {}
        for split, seqs in self._probes.items():
            stats, indices = self._run_split(model, routers, seqs)
            dumps[split] = indices
            prev = self._load_prev(split)
            if prev is not None and prev.shape == indices.shape:
                stats.update(self._consistency(indices, prev))
            for key, value in stats.items():
                self.trainer.record_metric(f"router_probe/{split}/{key}", value)

        step = self.step
        np.savez_compressed(Path(self.dump_dir) / f"probe_{step:07d}.npz", **dumps)
        self._prev_step = step

    def _run_split(self, model, routers, seqs) -> Tuple[Dict[str, float], np.ndarray]:
        device = next(model.parameters()).device
        n_layers = len(routers)
        n_experts = routers[0][1].num_experts
        top_k = routers[0][1].top_k

        captured: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            # forward (not pre-) hook: args[0] is the router input, which is all the
            # full score distribution needs, and the output carries the routed indices.
            def hook(module, args, output):
                captured[name] = args[0].detach()

            return hook

        p_sum = [torch.zeros(n_experts, dtype=torch.float64) for _ in range(n_layers)]
        h_cond_sum = [0.0] * n_layers
        margins: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        top1: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        topk: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

        was_training = model.training
        handles = [mod.register_forward_hook(make_hook(name)) for name, mod in routers]
        try:
            model.eval()  # also makes router jitter the identity (router.py:406)
            # temporary hooks change dynamo's guards; force eager so the probe does not
            # trigger a recompile of the training graph.
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                for start in range(0, seqs.shape[0], self.micro_batch_size):
                    batch = seqs[start : start + self.micro_batch_size].to(device)
                    captured.clear()
                    # logits_to_keep=1 skips the lm_head over the whole sequence; the
                    # routers have already fired by then. The default 0 means keep all,
                    # which is 6.6GB of logits per micro-batch at vocab 100352.
                    model(input_ids=batch, logits_to_keep=1)
                    for i, (name, mod) in enumerate(routers):
                        x = captured[name]
                        scores = self._scores(mod, x)
                        flat = scores.view(-1, n_experts)
                        p_sum[i] += flat.sum(dim=0).double().cpu()
                        h_cond_sum[i] += self._entropy(flat, dim=-1).sum().item()
                        pair = flat.topk(2, dim=-1).values
                        margins[i].append((pair[:, 0] - pair[:, 1]).float().cpu())
                        # argmax rather than column 0 of the routed indices: top-k
                        # ordering is a subclass detail, argmax is not.
                        top1[i].append(flat.argmax(dim=-1).cpu())
                        idx = mod.get_top_k(scores)[1].reshape(-1, top_k)
                        topk[i].append(idx.cpu())
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        n_tokens = int(seqs.numel())
        stats: Dict[str, float] = {}
        per_layer: Dict[str, List[float]] = {}
        for i in range(n_layers):
            p_bar = p_sum[i] / p_sum[i].sum()
            counts = torch.bincount(torch.cat(topk[i]).view(-1), minlength=n_experts)
            layer = self._layer_stats(
                p_bar=p_bar,
                h_cond=h_cond_sum[i] / n_tokens,
                counts=counts,
                margins=torch.cat(margins[i]),
                n_tokens=n_tokens,
                top_k=top_k,
                dead_expert_frac=self.dead_expert_frac,
            )
            for key, value in layer.items():
                per_layer.setdefault(key, []).append(value)
                stats[f"layer{i:02d}/{key}"] = value
        for key, values in per_layer.items():
            stats[key] = float(np.mean(values))

        indices = np.stack([torch.cat(topk[i]).numpy().astype(np.int16) for i in range(n_layers)])
        first = np.stack([torch.cat(top1[i]).numpy().astype(np.int16) for i in range(n_layers)])
        return stats, np.concatenate([first[:, :, None], indices], axis=2)

    @staticmethod
    def _layer_stats(
        *,
        p_bar: torch.Tensor,
        h_cond: float,
        counts: torch.Tensor,
        margins: torch.Tensor,
        n_tokens: int,
        top_k: int,
        dead_expert_frac: float,
    ) -> Dict[str, float]:
        """
        One layer's router summary. Pure, so the entropy definitions are testable
        without a model.

        Entropies are in units of ``log(num_experts)`` so arms with different
        expert counts share an axis. Their difference is the mutual information
        between token and expert, which is the only one of the three that tells
        healthy specialization apart from both collapse and indecision.
        """
        n_experts = p_bar.numel()
        ln_n = float(np.log(n_experts))
        h_marg = float(RouterProbeCallback._entropy(p_bar, dim=0))
        uniform = n_tokens * top_k / n_experts
        return {
            "entropy_marginal": h_marg / ln_n,
            "entropy_conditional": h_cond / ln_n,
            "mutual_information": (h_marg - h_cond) / ln_n,
            "effective_experts": float(np.exp(h_marg)),
            # relative threshold: exact zero never fires at N=16 and always fires at N=128.
            "dead_expert_frac": float((counts < dead_expert_frac * uniform).double().mean()),
            "max_load_frac": float(counts.max()) / (n_tokens * top_k),
            "margin_mean": float(margins.mean()),
            "margin_p10": float(margins.quantile(0.10)),
        }

    @staticmethod
    def _scores(router: MoERouter, x: torch.Tensor) -> torch.Tensor:
        logits = router.get_expert_logits(x).float()
        if router.gating_function == MoERouterGatingFunction.softmax:
            return logits.softmax(dim=-1)
        # sigmoid gates do not sum to one; normalize so the entropies stay comparable.
        scores = torch.sigmoid(logits) + 1e-7
        return scores / scores.sum(dim=-1, keepdim=True)

    @staticmethod
    def _entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
        return -(p * torch.log(p.clamp_min(1e-10))).sum(dim=dim)

    def _load_prev(self, split: str) -> Optional[np.ndarray]:
        if self._prev_step < 0:
            return None
        path = Path(self.dump_dir) / f"probe_{self._prev_step:07d}.npz"
        if not path.exists():
            log.warning("router probe: previous dump %s missing, skipping consistency", path)
            return None
        return np.load(path)[split]

    @staticmethod
    def _consistency(cur: np.ndarray, prev: np.ndarray) -> Dict[str, float]:
        # column 0 is argmax, columns 1: are the routed top-k set.
        agree = (cur[:, :, 0] == prev[:, :, 0]).mean(axis=1)
        a, b = cur[:, :, 1:], prev[:, :, 1:]
        inter = (a[:, :, :, None] == b[:, :, None, :]).any(axis=3).sum(axis=2)
        k = a.shape[2]
        jaccard = (inter / (2 * k - inter)).mean(axis=1)

        stats = {
            "top1_agreement": float(agree.mean()),
            "topk_jaccard": float(jaccard.mean()),
            "churn_rate": float(1.0 - agree.mean()),
        }
        for i in range(cur.shape[0]):
            stats[f"layer{i:02d}/top1_agreement"] = float(agree[i])
            stats[f"layer{i:02d}/topk_jaccard"] = float(jaccard[i])
        return stats
