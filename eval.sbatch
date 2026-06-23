#!/bin/bash
#SBATCH --job-name=test_trained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=testresults/logs/test_trained_%j.out

set -e

MODEL="$1"
CHECKPOINT="$2"
SCENARIO="$3"

if [ -z "$MODEL" ] || [ -z "$CHECKPOINT" ] || [ -z "$SCENARIO" ]; then
    echo "Usage: sbatch test_trained.sbatch <model> <checkpoint> <scenario>"
    echo "model must be: cnn or lstm"
    echo "Example:"
    echo "  sbatch test_trained.sbatch cnn logs_train/.../checkpoint_000100 configs/scenarios/S3_medium_30x30_4drones_lowDisp.json"
    exit 1
fi

case "$MODEL" in
    cnn)
        TEST_MODULE="src.tests.test_trained_search"
        ;;
    lstm)
        TEST_MODULE="src.tests.test_trained_lstm"
        ;;
    cnn_roles)
        TEST_MODULE="src.tests.test_trained_cnn_roles"
        ;;
    lstm_roles)
        TEST_MODULE="src.tests.test_trained_lstm_roles"
        ;;
    *)
        echo "Error: model must be either 'cnn', 'lstm', 'cnn_roles' or 'lstm_roles', got '$MODEL'"
        exit 1
        ;;
esac

module purge

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -m "$TEST_MODULE" \
    --checkpoint "$CHECKPOINT" \
    --scenario "$SCENARIO"
