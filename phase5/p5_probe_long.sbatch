#!/bin/bash
#SBATCH --job-name=p5_probe
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/samir.kassam/phase5_logs/probe_%j.out

set -euo pipefail

export BASH_ENV=/dev/null
unset -f conda 2>/dev/null || true
unset CONDA_EXE CONDA_SHLVL CONDA_PREFIX 2>/dev/null || true

export PATH=/data/samir.kassam/envs/emo/bin:$PATH
export HF_HOME=/scratch/users/samir.kassam/hf_cache
export HF_DATASETS_CACHE=/scratch/users/samir.kassam/hf_cache
export HF_HUB_CACHE=/scratch/users/samir.kassam/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/samir.kassam/hf_cache

export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 NUM_NODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

cd /data/samir.kassam/EMO

# 144 steps per epoch: warmup 20 is now 14% of a period rather than 50%, so router
# behaviour past warmup is observable. 3 periods = 432 steps.
NUM_EXPERTS=${NUM_EXPERTS:-16}
U_TOKENS=37748736
STEPS_PER_EPOCH=144
EPOCHS=3
MAXTOK=$(( STEPS_PER_EPOCH * EPOCHS * 262144 ))

RUN=p5_probe_n${NUM_EXPERTS}_j${SLURM_JOB_ID}
SAVE=/data/samir.kassam/models/$RUN

python src/scripts/train/olmo2-1B.py "$RUN" \
  --save-folder="$SAVE" \
  --data-root=/scratch/users/samir.kassam/dclm \
  --dataset.mix=dclm-100m-train \
  --trainer.max_duration="{value: $MAXTOK, unit: tokens}" \
  --lr=3e-3 --wd=0.8 --scheduler=wsds --epochs=$EPOCHS --unique-tokens=$U_TOKENS \
  --num-experts=$NUM_EXPERTS \
  --trainer.callbacks.checkpointer.save_interval=$STEPS_PER_EPOCH \
  --trainer.callbacks.checkpointer.ephemeral_save_interval=$(( STEPS_PER_EPOCH - 1 )) \
  --router-probe=/scratch/users/samir.kassam/router_probe_v1.npz \
  --trainer.callbacks.lm_evaluator.eval_interval=$STEPS_PER_EPOCH \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.callbacks.wandb.entity=samirkassam \
  --trainer.callbacks.wandb.project=dclm-scaling \
  --trainer.callbacks.wandb.name="$RUN" \
  --trainer.callbacks.wandb.tags="[probe_long,moe,n${NUM_EXPERTS}]"
