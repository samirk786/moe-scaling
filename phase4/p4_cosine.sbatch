#!/bin/bash
#SBATCH --job-name=p4_cosine
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/samir.kassam/phase4_logs/p4_cosine_%j.out

set -euo pipefail

# skip conda init (its .bashrc hook reads NFS miniforge and can wedge); PATH-pin instead
export BASH_ENV=/dev/null
unset -f conda 2>/dev/null || true
unset CONDA_EXE CONDA_SHLVL CONDA_PREFIX 2>/dev/null || true

# env AND repo are node-local to horton, off the degraded /accounts mount.
export PATH=/data/samir.kassam/envs/emo/bin:$PATH

# sbatch does not source .bashrc, set caches explicitly (scratch reads confirmed OK on horton).
export HF_HOME=/scratch/users/samir.kassam/hf_cache
export HF_DATASETS_CACHE=/scratch/users/samir.kassam/hf_cache
export HF_HUB_CACHE=/scratch/users/samir.kassam/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/samir.kassam/hf_cache

# single-GPU single-process dist env (plain python, not torchrun).
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 NUM_NODES=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

cd /data/samir.kassam/EMO

python src/scripts/train/olmo2-1B.py p4_cosine_100m_4ep \
  --save-folder=/data/samir.kassam/models/p4_cosine_100m_4ep \
  --data-root=/scratch/users/samir.kassam/dclm \
  --dataset.mix=dclm-100m-train \
  --trainer.max_duration='{value: 369878772, unit: tokens}' \
  --lr=3e-3 --wd=0.8 --scheduler=cosine \
  --trainer.callbacks.lm_evaluator.eval_interval=353 \
  --trainer.callbacks.wandb.entity=samirkassam \
  --trainer.callbacks.wandb.project=dclm-scaling \
  --trainer.callbacks.wandb.name=p4_cosine_100m_4ep \
  --trainer.callbacks.wandb.tags='[phase4,cosine]' \
  --trainer.callbacks.downstream_evaluator.eval_interval=353 \
  --trainer.callbacks.downstream_evaluator.tasks='[hellaswag]'
