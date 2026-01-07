#!/bin/bash
# Parallel HPO launcher for Neural Lyapunov
#
# Usage: ./launch_parallel.sh <config_file> [n_trials] [n_workers]
# Example: ./launch_parallel.sh ../configs/sta.yaml 50 4

set -e

CONFIG_FILE=${1:-"../configs/sta.yaml"}
N_TRIALS=${2:-50}
N_WORKERS=${3:-4}

# Extract controller name from config
CTRL_NAME=$(grep -A1 "^controller:" "$CONFIG_FILE" | grep "name:" | awk '{print $2}' | tr -d '"')
STUDY_NAME="${CTRL_NAME}_hpo"
DB_PATH="${CTRL_NAME}_hpo.db"

echo "=============================================="
echo "Parallel HPO for $CTRL_NAME"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Trials per worker: $N_TRIALS"
echo "Workers: $N_WORKERS"
echo "Study: $STUDY_NAME"
echo "Database: $DB_PATH"
echo "=============================================="

# Create output directory
mkdir -p outputs_hpo

# Launch workers in parallel
for i in $(seq 0 $((N_WORKERS - 1))); do
    echo "Starting worker $i..."
    python tune.py \
        --config "$CONFIG_FILE" \
        --n_trials "$N_TRIALS" \
        --polish \
        --study_name "$STUDY_NAME" \
        --storage "sqlite:///$DB_PATH" \
        > "outputs_hpo/worker_${i}.log" 2>&1 &
done

echo ""
echo "All $N_WORKERS workers started."
echo "Monitor progress with: tail -f outputs_hpo/worker_*.log"
echo "View results with: optuna-dashboard sqlite:///$DB_PATH"
echo ""
echo "To stop all workers: pkill -f 'tune.py.*$STUDY_NAME'"
