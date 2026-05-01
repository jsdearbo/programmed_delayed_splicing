#!/usr/bin/env bash

#SBATCH --job-name=borzoi_inference
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_ANY
#SBATCH --error=SLURM/%x_%j.err

set -euo pipefail

REPO_DIR="/netfiles/dmlabshare1/rstagg/programmed_delayed_splicing"

module purge
module load cuda/12.6.2 cudnn/9.12.0-cuda12

eval "$(conda shell.bash hook)"
conda activate elife-delay-splice

cd "${REPO_DIR}"

echo "[$(date)] Host:          ${HOSTNAME}"
echo "[$(date)] Working dir:   ${PWD}"
echo "[$(date)] Conda env:     ${CONDA_PREFIX}"
echo "[$(date)] Python:        $(python --version)"
echo "[$(date)] PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
echo "[$(date)] CUDA avail:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[$(date)] GPUs visible:  ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

# ── Run inference ─────────────────────────────────────────────────
python "${REPO_DIR}/scripts/inference.py"

echo "[$(date)] Inference complete."
