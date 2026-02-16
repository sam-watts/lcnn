#!/usr/bin/env bash
# run_next.sh -- Wait for the current training run to finish, then start the next one.
#
# Usage:
#   nohup bash run_next.sh &> run_next.log &
#   # or inside tmux:
#   bash run_next.sh
#
# What it does:
#   1. Waits for PID $WAIT_PID (the current train.py) to exit
#   2. Launches a new training run with the current config/wireframe.yaml
#
# You can safely close your terminal / go to sleep after launching this.

set -euo pipefail

# --------------- CONFIG ---------------
WAIT_PID=92848                          # PID of the currently running train.py
WORK_DIR=/home/swatts/lcnn
VENV_DIR="$WORK_DIR/.venv"
CONFIG="$WORK_DIR/config/wireframe.yaml"
DEVICES="0"                             # CUDA_VISIBLE_DEVICES
# --------------------------------------

echo "=== run_next.sh ==="
echo "Waiting for PID $WAIT_PID to finish..."
echo "  (current train.py run)"
echo "  Started at: $(date)"
echo ""

# Poll until the process exits (works even if we're not the parent)
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done

echo "PID $WAIT_PID has exited at $(date)"
echo "Sleeping 10s before starting next run..."
sleep 10

# Activate venv
cd "$WORK_DIR"
source "$VENV_DIR/bin/activate"

echo ""
echo "=== Starting next training run ==="
echo "  Config: $CONFIG"
echo "  Time:   $(date)"
echo ""

python train.py -d "$DEVICES" "$CONFIG"

echo ""
echo "=== Training complete at $(date) ==="
