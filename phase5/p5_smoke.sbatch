#!/bin/bash
#SBATCH --job-name=p5_smoke
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/samir.kassam/phase5_logs/smoke_%j.out
#SBATCH --open-mode=append

set -euo pipefail

# skip conda init (.bashrc hook reads NFS miniforge; wedges when NFS degraded)
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

# 40 steps per epoch: WSDS needs warmup 20 + decay 4 to fit inside a period (P >= 23).
# 3 periods = 120 steps, so four probes fire (start, 40, 80, 120), three consistency reads.
U_TOKENS=10485760
STEPS_PER_EPOCH=40
EPOCHS=3
MAXTOK=$(( STEPS_PER_EPOCH * EPOCHS * 262144 ))

RUN=p5_smoke_n16_j${SLURM_JOB_ID}
SAVE=/data/samir.kassam/models/$RUN

# evaluators off: this run tests the MoE branch and the probe, and a startup LM eval
# is 862 batches before anything interesting happens.
python src/scripts/train/olmo2-1B.py "$RUN" \
  --save-folder="$SAVE" \
  --data-root=/scratch/users/samir.kassam/dclm \
  --dataset.mix=dclm-100m-train \
  --trainer.max_duration="{value: $MAXTOK, unit: tokens}" \
  --lr=3e-3 --wd=0.8 --scheduler=wsds --epochs=$EPOCHS --unique-tokens=$U_TOKENS \
  --num-experts=16 \
  --trainer.callbacks.checkpointer.save_interval=120 \
  --trainer.callbacks.checkpointer.ephemeral_save_interval=119 \
  --router-probe=/scratch/users/samir.kassam/router_probe_v1.npz \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.callbacks.wandb.entity=samirkassam \
  --trainer.callbacks.wandb.project=dclm-scaling \
  --trainer.callbacks.wandb.name="$RUN" \
  --trainer.callbacks.wandb.tags="[smoke,moe,router_probe]"
