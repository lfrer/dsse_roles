#!/bin/bash
#SBATCH --job-name=dsse_train
#SBATCH --partition=longgpu
#SBATCH --time=5-00:00:00   
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

module purge
module add anaconda

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.baselines.tune_params  "$@"
