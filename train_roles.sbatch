#!/bin/bash
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_roles/slurm_%x_%j.out
#SBATCH --exclusive

module purge
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.run_training "$@"
