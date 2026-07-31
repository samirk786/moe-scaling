#!/bin/bash
#SBATCH --job-name=copy_emo_env
#SBATCH --partition=berkeleynlp
#SBATCH --qos=normal
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/copy_env_%j.out

set -euo pipefail
echo "host: $(hostname)  start: $(date)"
mkdir -p /data/samir.kassam/envs
time tar -C /scratch/users/samir.kassam/conda/envs -cf - emo \
  | tar -C /data/samir.kassam/envs -xf -
echo "tar done: $(date)"
du -sh /data/samir.kassam/envs/emo
/data/samir.kassam/envs/emo/bin/python -c "import time;t=time.time();import torch;print(f'torch ok in {time.time()-t:.1f}s')"
echo "finished: $(date)"
