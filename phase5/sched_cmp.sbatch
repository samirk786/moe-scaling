#!/bin/bash
#SBATCH --job-name=sched_cmp
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton   # override with sbatch --nodelist=
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-8%3
#SBATCH --output=/scratch/users/samir.kassam/sched_logs/%A_%a.out

set -euo pipefail

# .bashrc conda hook reads NFS miniforge and wedges when NFS is degraded
export BASH_ENV=/dev/null
unset -f conda 2>/dev/null || true
unset CONDA_EXE CONDA_SHLVL CONDA_PREFIX 2>/dev/null || true

export PATH=/data/samir.kassam/envs/emo/bin:$PATH
export HF_HOME=/scratch/users/samir.kassam/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME

# TCPStore binds even at world size 1; job id is unique and survives requeue
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 NUM_NODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

cd /data/samir.kassam/EMO

: "${MODE:?MODE must be set: trunk, fork, cosine, or moe}"
: "${SHAPE:=1-sqrt}"

U_TOKENS=${POOL_TOKENS:-92469693}
SPE=${POOL_SPE:-353}
EPOCHS=${POOL_EPOCHS:-16}
TOTAL=$(( SPE * EPOCHS ))          # 5648
DECAY=$(( (TOTAL * 20 + 50) / 100 ))   # 1130
FORKPT=$(( TOTAL - DECAY ))        # 4518
TOKSTEP=262144

# empty = the default 1.5B rung
SIZE_OVERRIDES=${SIZE_OVERRIDES:-}

LRS=(1e-3 3e-3 1e-2)
WDS=(${POOL_WDS:-0.1 0.8 3.2})
i=$SLURM_ARRAY_TASK_ID
LR=${LRS[$(( i / 3 ))]}
WD=${WDS[$(( i % 3 ))]}
LR=${OVERRIDE_LR:-$LR}
WD=${OVERRIDE_WD:-$WD}
CELL=lr${LR}_wd${WD}

MODEL=/data/samir.kassam/models
COMMON=(
  --data-root=/scratch/users/samir.kassam/dclm
  --dataset.mix=${POOL_MIX:-dclm-100m-train}
  --lr="$LR" --wd="$WD" --unique-tokens=$U_TOKENS
  --trainer.callbacks.wandb.entity=samirkassam
  --trainer.callbacks.wandb.project=dclm-scaling
  --trainer.callbacks.downstream_evaluator.tasks='[hellaswag]'
  $SIZE_OVERRIDES
)

case "$MODE" in
  trunk)
    RUN=trunk_${POOL_TAG:+${POOL_TAG}_}$CELL
    python src/scripts/train/olmo2-1B.py "$RUN" \
      --scheduler=trunk \
      --save-folder="$MODEL/$RUN" \
      --trainer.max_duration="{value: $(( FORKPT * TOKSTEP )), unit: tokens}" \
      --trainer.callbacks.checkpointer.save_interval=${SAVE_EVERY:-1500} \
      --trainer.callbacks.wandb.name="${RUN}_j${SLURM_JOB_ID}" \
      --trainer.callbacks.wandb.tags="[sched_cmp,trunk]" \
      "${COMMON[@]}"
    ;;
  fork)
    # source run, fork step and decay are settable so one script covers every
    # arm; defaults reproduce the original ep16 fork exactly.
    SRC=${FORK_SRC:-trunk_${POOL_TAG:+${POOL_TAG}_}$CELL}
    FORKPT=${FORK_STEP:-$FORKPT}
    DECAY=${FORK_DECAY:-$DECAY}
    RUN=${FORK_NAME:-fork_ep${EPOCHS}_${SHAPE}_${POOL_TAG:+${POOL_TAG}_}$CELL}
    END=$(( FORKPT + DECAY ))
    echo "FORK: $SRC/step$FORKPT -> $RUN, decay $DECAY, end $END"
    python src/scripts/train/olmo2-1B.py "$RUN" \
      --scheduler=fork --decay-steps=$DECAY --decay-shape="$SHAPE" \
      --save-folder="$MODEL/$RUN" \
      --load_path="$MODEL/$SRC/step$FORKPT" \
      --load_trainer_state=true \
      --trainer.max_duration="{value: $(( END * TOKSTEP )), unit: tokens}" \
      --trainer.callbacks.wandb.name="${RUN}_j${SLURM_JOB_ID}" \
      --trainer.callbacks.wandb.tags="[sched_cmp,fork,ep${EPOCHS}]" \
      ${FORK_EXTRA:-} \
      "${COMMON[@]}"
    ;;
  cosine)
    RUN=cosine_ep${EPOCHS}_$CELL
    python src/scripts/train/olmo2-1B.py "$RUN" \
      --scheduler=cosine --alpha-f=0.0 \
      --save-folder="$MODEL/$RUN" \
      --trainer.max_duration="{value: $(( TOTAL * TOKSTEP )), unit: tokens}" \
      --trainer.callbacks.lm_evaluator.eval_interval=$SPE \
      --trainer.callbacks.downstream_evaluator.eval_interval=$SPE \
      --trainer.callbacks.wandb.name="${RUN}_j${SLURM_JOB_ID}" \
      --trainer.callbacks.wandb.tags="[sched_cmp,cosine,ep${EPOCHS}]" \
      "${COMMON[@]}"
    ;;
  moe)
    MOE_LRS=(3e-4 1e-3 3e-3)
    LR=${MOE_LRS[$i]}
    WD=${MOE_WD:-0.8}
    N_EXPERTS=${N_EXPERTS:-32}
    MOE_EPOCHS=12
    MOE_STEPS=$(( SPE * MOE_EPOCHS ))
    RUN=moe${N_EXPERTS}_${POOL_TAG:+${POOL_TAG}_}lr${LR}_wd${WD}
    python src/scripts/train/olmo2-1B.py "$RUN" \
      --scheduler=trunk \
      --num-experts=$N_EXPERTS \
      --lb-loss-weight=0.1 \
      --router-probe=/scratch/users/samir.kassam/router_probe_v1.npz \
      --save-folder="$MODEL/$RUN" \
      --trainer.max_duration="{value: $(( MOE_STEPS * TOKSTEP )), unit: tokens}" \
      --trainer.callbacks.checkpointer.save_interval=1128 \
      --trainer.callbacks.wandb.name="${RUN}_j${SLURM_JOB_ID}" \
      --trainer.callbacks.wandb.tags="[sched_cmp,moe,N${N_EXPERTS}]" \
      --lr="$LR" --wd="$WD" \
      --data-root=/scratch/users/samir.kassam/dclm \
      --dataset.mix=${POOL_MIX:-dclm-100m-train} \
      --unique-tokens=$U_TOKENS \
      --trainer.callbacks.wandb.entity=samirkassam \
      --trainer.callbacks.wandb.project=dclm-scaling \
      --trainer.callbacks.downstream_evaluator.tasks='[hellaswag]'
    ;;
  *) echo "MODE must be trunk, fork, cosine, or moe" >&2; exit 1 ;;
esac
