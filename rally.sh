#!/bin/bash
# Run VLM rally detection in a tmux session
#
# Usage:
#   ./rally.sh G1 G2 G3     # Process specific games

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_NAME="rally"

# Require at least one game argument
if [ $# -eq 0 ]; then
    echo "Usage: ./rally.sh G1 G2 G3 ..."
    exit 1
fi
GAMES="$*"

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
                python -m yp_video.core.vlm_segment --video \"\$video\"; \
            fi; \
        done; \
    done; \
    echo '========================================'; \
    echo 'Rally detection finished.'; \
    echo '========================================'; \
    exec bash"
