# Extending DSSE-Based MARL Algorithms with Emergent Roles for Maritime Search and Rescue

This repository contains the code accompanying the research project *"Extending
DSSE-Based MARL Algorithms with Emergent Roles for Maritime Search and
Rescue."* It extends the CNN baseline from the
[DSSE (Drone Swarm Search Environment)](https://pettingzoo.farama.org/)
repository with the R3DM emergent-role mechanism (Goel et al., 2025) and
compares it against role-free CNN and CNN+LSTM baselines.

The central research question is whether emergent role specialization among
cooperating drones improves search performance in a multi-agent reinforcement
learning (MARL) setting for search and rescue. Six scenarios (S1–S6) vary grid
size (20×20 to 40×40), drone count (2, 4, or 6), and the dispersion of the
target probability map.

Models are trained with PPO and evaluated on metrics
such as success rate, time to find, search efficiency, and role-specific
diagnostics (e.g. marginal role entropy, co-occupancy, and behavioral
divergence between roles).

## Repository structure

- `src/models/` – CNN, CNN+LSTM, and R3DM role-augmented model variants
- `src/policies/` – custom PPO policies implementing the R3DM role mechanism
- `src/baselines/` – hyperparameter tuning and baseline training scripts
- `src/tests/` – evaluation scripts for trained checkpoints
- `logs_train/`, `logs_roles/` – training visualization utilities (`vis_training.py`, `vis_training_roles.py`)
- `logs_tune/` – output logs from hyperparameter tuning jobs
- `plotting/` – result plotting and comparison scripts
- `testresults/` – evaluation results
- `thesis_figures/` – figures used in the thesis
- `configs/` – scenario and training configuration files

## Setup

Tested on Python 3.12.

```bash
conda create -n dsse312 python=3.12 pip -y
conda activate dsse312
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage

### Hyperparameter tuning

Run directly:

```bash
python -m src.baselines.tune_params --algo cnn --scenario configs/scenarios/S4_medium_30x30_4drones_highDisp.json --train_cfg configs/training/ppo_centralized_baseline.json --outdir configs/training/slurm/cnn --num_samples 25 --max_env_steps 900000 --max_concurrent 3

python -m src.baselines.tune_params --algo lstm --scenario configs/scenarios/S4_medium_30x30_4drones_highDisp.json --train_cfg configs/training/ppo_lstm_baseline.json --outdir configs/training/slurm/lstm --num_samples 25 --max_env_steps 900000 --max_concurrent 3
```

Or as a Slurm job:

```bash
sbatch --job-name=cnn_S4 --output=logs_tune/cnn_S4_%j.out tune.sbatch --algo cnn --scenario configs/scenarios/S4_medium_30x30_4drones_highDisp.json --train_cfg configs/training/ppo_centralized_baseline.json --outdir configs/training/slurm/cnn --num_samples 25 --max_env_steps 900000 --max_concurrent 3

sbatch --job-name=lstm_S4 --output=logs_tune/lstm_S4_%j.out tune.sbatch --algo lstm --scenario configs/scenarios/S4_medium_30x30_4drones_highDisp.json --train_cfg configs/training/ppo_lstm_baseline.json --outdir configs/training/slurm/lstm --num_samples 25 --max_env_steps 900000 --max_concurrent 3
```

### Training

```bash
# Baseline CNN / CNN+LSTM (example: S1)
sbatch --job-name=train_cnn_S1 train_final.sbatch --algorithm ppo_centralized --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json --train_config configs/training/slurm/cnn/best_configs/best_train_config_cnn.json --outdir logs_train/ppo_centralized_S1

sbatch --job-name=train_lstm_S1 train_final.sbatch --algorithm ppo_lstm --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json --train_config configs/training/slurm/lstm/best_configs/best_train_config_lstm.json --outdir logs_train/ppo_lstm_S1

# CNN + R3DM roles (example: S1)
sbatch --job-name=train_cnn_roles_S1 train_roles.sbatch --algorithm ppo_cnn_roles --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json --train_config configs/training/cnn_roles_baseline.json --outdir logs_roles/ppo_cnn_roles_s1
```

Equivalent commands for scenarios S2–S6 follow the same pattern, substituting
the corresponding scenario file and output directory.

### Visualizing training progress

```bash
python logs_train/vis_training.py --run-dir logs_train/ppo_cnn_S2/ray_res/DSSE_PPO_Baseline/PPO_CNN/<TRIAL_DIR>/ --run-name cnn_S2

python logs_roles/vis_training_roles.py --run-dir logs_roles/ppo_cnn_roles_s1/ray_res/DSSE_PPO_CNN_R3DM/PPO_CNN_R3DM/<TRIAL_DIR>/ --run-name ppo_cnn_roles_s1

python plotting/plot_compare_two_training_runs.py --run-dir-a logs_train/... --run-dir-b logs_roles/... --label-a "CNN S2" --label-b "CNN + Roles S2" --comparison-name cnn_vs_roles_s2 --logs-root logs_roles/comparison
```

### Evaluating trained checkpoints

```bash
python -m src.tests.test_trained_search --checkpoint logs_train/ppo_centralized_S1/ray_res/DSSE_PPO_Baseline/PPO_CNN/<TRIAL_DIR>/ --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json

python -m src.tests.test_trained_lstm --checkpoint logs_train/ppo_lstm_S1/ray_res/DSSE_PPO_LSTM/PPO_LSTM/<TRIAL_DIR>/ --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json
```

To change the selection metric used to pick the best checkpoint:

```bash
python -m src.tests.test_trained_search --checkpoint logs_train/ppo_centralized_S1/ray_res/DSSE_PPO_Baseline/PPO_CNN/<TRIAL_DIR>/ --scenario configs/scenarios/S1_small_20x20_2drones_lowDisp.json --select_metric success_rate --mode max
```

Evaluation can also be run as a Slurm job:

```bash
sbatch --job-name=lstm_S1 --output=testresults/logs/%x_%j.out eval.sbatch lstm logs_train/ppo_lstm_S1/ray_res/DSSE_PPO_LSTM/PPO_LSTM/<TRIAL_DIR>/ configs/scenarios/S1_small_20x20_2drones_lowDisp.json

sbatch --job-name=cnn_roles_S2 --output=testresults/logs/%x_%j.out eval.sbatch cnn_roles logs_roles/ppo_cnn_S2/ray_res/DSSE_PPO_CNN_R3DM/PPO_CNN_R3DM/<TRIAL_DIR>/ configs/scenarios/S2_small_20x20_2drones_highDisp.json
```

### Visualizing results

```bash
python plotting/visualization.py testresults/CNN/S1/eval_metrics_cnn_<BEST_TAG>.json

python plotting/more_vis.py testresults/CNN_R3DM/S1/eval_metrics_cnn_r3dm_<BEST_TAG>.jsonl

python plotting/visualization_comparison.py testresults/CNN/S3/eval_metrics_cnn_<BEST_TAG>.jsonl testresults/CNN_R3DM/S3/eval_metrics_cnn_r3dm_<BEST_TAG>.jsonl --label-a "CNN S3" --label-b "CNN+Roles S3" --output-dir testresults/comparison/CNN/S3

python plotting/plot_role_profile.py testresults/CNN_R3DM/S1/eval_metrics_cnn_r3dm_summary_<BEST_TAG>.json testresults/CNN_R3DM/S2/eval_metrics_cnn_r3dm_summary_<BEST_TAG>.json ... --out-dir testresults/comparison/role_profile/ --labels S1 S2 S3 S4 S5 S6

python plotting/plot_role_layout_gallery.py testresults/CNN_R3DM/S1/eval_metrics_cnn_r3dm_summary_<BEST_TAG>.json ... --labels S1 S2 S3 S4 S5 S6 --out-dir testresults/comparison/role_profile/layout
```

## Acknowledgements

Built on top of the [DSSE](https://github.com/pfeinsper/drone-swarm-search)
environment and the R3DM role mechanism (Goel et al., 2025).
