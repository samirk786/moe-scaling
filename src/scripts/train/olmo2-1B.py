"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llm/train.py run_name [OVERRIDES...]
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.mixes import DataMix
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe import (
    MoEConfig,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import (
    ConstantWithWarmup,
    CosWithWarmup,
    WSD,
    WSDS,
    WSDSqrt,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    RouterProbeCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# from data_mixes import CustomDataMix

log = logging.getLogger(__name__)


C4_VALIDATION_PATH = [
    "/weka/oe-training-default/ai2-llm/examples/c4-en/gpt2/c4-validation.00000-00008.npy"
]

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH  # 64 seqs, matches infinite-compute batch=64

MOE_TOP_K = 8
MOE_GRANULARITY = 8  # expert width = dense d_ff / G, so top_k experts match the dense FFN


# docs: start-define-config
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""
    # docs: end-define-config


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # docs: start-build-components
    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    # docs: end-build-components

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # docs: start-load-path
    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
    # docs: end-load-path

    # Train.
    trainer.fit()


def build_model_config(opts, tokenizer_config) -> TransformerConfig:
    vocab_size = tokenizer_config.padded_vocab_size()
    dense = TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size)
    if not opts.num_experts:
        return dense

    # read the built size, do not recompute it: llama_like rounds up to a multiple of 256,
    # so d_ff == 4 * d_model is an output of that and not a rule
    d_ff = dense.block.feed_forward.hidden_size

    # olmo2_1B_v2 rather than llama_like_moe: it inherits qk_norm, rope_theta, layer_norm_eps
    # and hidden_size_multiplier, so the arms differ in the feed-forward and nothing else
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
            # total across layers: MoEBase divides by n_layers, so holding it fixed keeps
            # aux pressure constant as depth changes
            lb_loss_weight=opts.lb_loss_weight,
            z_loss_weight=0.001,
        ),
    )


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = build_model_config(opts, tokenizer_config)
    # docs: end-model-config

    log.info(f"Using data root: {opts.data_root}")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
        generate_doc_lengths=False,
        instance_filter_config=None,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,  # NOTE: this is specified in tokens, not instances
        seed=0,
        num_workers=4,
    )

    steps_per_epoch = round(opts.unique_tokens / GLOBAL_BATCH_SIZE) if opts.unique_tokens else 0
    if opts.scheduler == "constant":
        scheduler = ConstantWithWarmup(warmup=20)
    elif opts.scheduler == "cosine":
        scheduler = CosWithWarmup(warmup=20, alpha_f=opts.alpha_f)
    elif opts.scheduler == "wsd":
        scheduler = WSD(warmup=20, decay_fraction=0.1)
    elif opts.scheduler == "wsds":
        assert steps_per_epoch > 0, "wsds needs --unique-tokens"
        scheduler = WSDS(period_lengths=[steps_per_epoch] * opts.epochs, warmup=20, decay_fraction=0.1)
    elif opts.scheduler == "trunk":
        scheduler = ConstantWithWarmup(warmup=20)
    elif opts.scheduler == "fork":
        assert opts.decay_steps > 0, "fork needs --decay-steps"
        cls = WSDSqrt if opts.decay_shape == "1-sqrt" else WSD
        scheduler = cls(warmup=20, decay=opts.decay_steps, decay_fraction=None)
    else:
        raise ValueError(f"unknown scheduler {opts.scheduler}")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4
        * SEQUENCE_LENGTH,  # NOTE: this is specified in tokens, not instances
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=opts.wd,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=(steps_per_epoch if opts.scheduler in ("wsds", "trunk", "fork") else 5000),
                ephemeral_save_interval=(max(1, steps_per_epoch // 4) if opts.scheduler in ("wsds", "trunk", "fork") else 100),
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("beaker", BeakerCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.dclm_fixed_val,
                    mix_base_dir=opts.data_root,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=(steps_per_epoch if opts.scheduler in ("wsds", "trunk", "fork") else 25),
                eval_on_startup=True,
                eval_on_finish=True,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            # https://github.com/allenai/OLMo-in-loop-evals/blob/main/src/olmo_eval/tasks.py#L1752
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "hellaswag",
                    "arc_challenge",
                    "piqa",
                    "copa",
                    "mmlu_stem",
                    "mmlu_humanities",
                    "mmlu_social_sciences",
                    "mmlu_other",
                ],
                tokenizer=tokenizer_config,
                eval_interval=(steps_per_epoch if opts.scheduler in ("wsds", "trunk", "fork") else 250),
            ),
        )
    )

    if opts.router_probe:
        if steps_per_epoch <= 0:
            raise ValueError("--router-probe needs --unique-tokens to set the probe interval")
        trainer_config = trainer_config.with_callback(
            "router_probe",
            RouterProbeCallback(
                probe_file=opts.router_probe,
                dump_dir=f"{save_folder}/router_probes",
                interval=steps_per_epoch,
            ),
        )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    # Apply overrides.
    # docs: start-config-merge
    config = config.merge(overrides)
    # docs: end-config-merge

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train a transformer language model on c4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help="""Number of routed experts. 0 builds the dense arm.""",
    )
    parser.add_argument(
        "--lb-loss-weight",
        type=float,
        default=0.1,
        help="""MoE load-balancing loss weight, as a total across layers.
        0.1 is the released EMO value; the repo preset 0.01 collapses deep-layer routing.""",
    )
    parser.add_argument(
        "--router-probe",
        type=str,
        help="""Path to the router probe .npz. Enables router instrumentation.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/weka/oe-training-default/ai2-llm",
        help="Root directory for the data mix (mix_base_dir).",
    )
    parser.add_argument("--wd", type=float, default=0.033, help="""Weight decay.""")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        choices=["constant", "cosine", "wsd", "wsds", "trunk", "fork"],
        help="""LR schedule. wsds runs one decay period per epoch.""",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="""Number of wsds periods. Ignored by the other schedulers.""",
    )
    parser.add_argument(
        "--unique-tokens",
        type=int,
        default=0,
        help="""Size of the unique data pool in tokens. Sets steps per epoch.""",
    )
    parser.add_argument(
        "--alpha-f",
        type=float,
        default=0.0,
        help="""Final LR as a fraction of peak. Cosine only."""
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=0,
        help="""Length of the fork decay in steps."""
    )
    parser.add_argument(
        "--decay-shape",
        type=str,
        default="1-sqrt",
        choices=["1-sqrt", "linear"],
        help="""Fork decay shape."""
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    # note that this function basically initializes all the classes but does not actually call build on them
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
