import pathlib

p = pathlib.Path("src/scripts/train/olmo2-1B.py")
src = p.read_text()

def replace_once(s, old, new):
    assert s.count(old) == 1, f"want 1 match, got {s.count(old)} for: {old!r}"
    return s.replace(old, new)

def insert_before(s, anchor_stripped, block):
    lines = s.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.strip() == anchor_stripped:
            ind = ln[:len(ln) - len(ln.lstrip())]
            ins = [(ind + b + "\n") if b else "\n" for b in block]
            return "".join(lines[:i] + ins + lines[i:])
    raise AssertionError(f"anchor not found: {anchor_stripped!r}")

# 1. imports
src = replace_once(src,
    "from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig",
    "from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, WSD, WSDS, OptimGroupOverride, SkipStepAdamWConfig")

# 2. new opts before parse_known_args
src = insert_before(src, "opts, overrides = parser.parse_known_args()", [
    'parser.add_argument("--wd", type=float, default=0.033)',
    'parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine", "wsd", "wsds"])',
    'parser.add_argument("--epochs", type=int, default=1)',
    'parser.add_argument("--unique-tokens", type=int, default=0)',
    '',
])

# 3. scheduler selection before train_module_config
src = insert_before(src, "train_module_config = TransformerTrainModuleConfig(", [
    "# steps per epoch from the seed budget (GLOBAL_BATCH_SIZE is in tokens)",
    "steps_per_epoch = round(opts.unique_tokens / GLOBAL_BATCH_SIZE) if opts.unique_tokens else 0",
    'if opts.scheduler == "constant":',
    "    scheduler = ConstantWithWarmup(warmup=20)",
    'elif opts.scheduler == "cosine":',
    "    scheduler = CosWithWarmup(warmup_steps=2000)",
    'elif opts.scheduler == "wsd":',
    "    scheduler = WSD(warmup=20, decay_fraction=0.1)",
    'elif opts.scheduler == "wsds":',
    '    assert steps_per_epoch > 0, "wsds needs --unique-tokens"',
    "    scheduler = WSDS(period_lengths=[steps_per_epoch] * opts.epochs, warmup=20, decay_fraction=0.1)",
    "",
])

# 4. use opts.wd + the selected scheduler
src = replace_once(src, "weight_decay=0.033,", "weight_decay=opts.wd,")
src = replace_once(src, "scheduler=ConstantWithWarmup(warmup=20),", "scheduler=scheduler,")

# 5. align save + eval intervals to epoch boundaries under wsds
src = replace_once(src, "save_interval=5000,", 'save_interval=(steps_per_epoch if opts.scheduler == "wsds" else 5000),')
src = replace_once(src, "eval_interval=25,", 'eval_interval=(steps_per_epoch if opts.scheduler == "wsds" else 25),')
src = replace_once(src, "eval_interval=250,", 'eval_interval=(steps_per_epoch if opts.scheduler == "wsds" else 250),')

p.write_text(src)
print("Part A applied. Review: git diff src/scripts/train/olmo2-1B.py")
