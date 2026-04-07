#!/bin/bash
#SBATCH --job-name=glucose_pred_t4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12gb
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

echo "Running on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

source ~/software/init-conda
conda activate pytorch

nvidia-smi

JOB_SCRATCH="/scratch/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"
mkdir -p "${SLURM_SUBMIT_DIR}/artifacts"

python main.py \
  --device-profile slurm-cuda \
  --batch-size 128 \
  --max-epochs 1 \
  --output-dir "${SLURM_SUBMIT_DIR}/artifacts/glucose_${SLURM_JOB_ID}"
