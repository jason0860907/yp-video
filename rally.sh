#!/bin/bash
# Run VLM rally detection in a tmux session
#
# Usage:
#   ./rally.sh              # Process default games (G1-G10)
#   ./rally.sh G11 G12 G13  # Process specific games

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_NAME="rally"

# Use arguments if provided, otherwise default list
if [ $# -gt 0 ]; then
    GAMES="$*"
else
    GAMES="G4 G5 G6 G7 G8 G9 G10"
fi

# Kill existing session if running
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Killing existing tmux session '${SESSION_NAME}'..."
    tmux kill-session -t "${SESSION_NAME}"
fi

echo "----------------------------------------"
echo "Starting rally detection in tmux session '${SESSION_NAME}'..."
echo "Games: ${GAMES}"
echo "----------------------------------------"
echo "Attach with: tmux attach -t ${SESSION_NAME}"

# Start rally detection in a new tmux session
tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SCRIPT_DIR} && source .venv/bin/activate && \
    for game in ${GAMES}; do \
        for video in ~/videos/cuts/*_\${game}_*.mp4; do \
            if [ -f \"\$video\" ]; then \
                python vlm_segment.py --video \"\$video\"; \
            fi; \
        done; \
    done; \
    echo '========================================'; \
    echo 'Rally detection finished.'; \
    echo '========================================'; \
    exec bash"
