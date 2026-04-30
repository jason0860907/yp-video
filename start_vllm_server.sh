#!/bin/bash
# Start vLLM server in a tmux session (config from vllm.env)
#
# Usage:
#   ./start_vllm_server.sh [MODEL_NAME] [PORT]
#
# Examples:
#   ./start_vllm_server.sh                                    # Use defaults from vllm.env
#   ./start_vllm_server.sh Qwen/Qwen3-VL-4B-Instruct 8001    # Override model and port

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_NAME="vllm"

# Load shared config
source "${SCRIPT_DIR}/vllm.env"

MODEL_NAME="${1:-$VLLM_MODEL}"
PORT="${2:-$VLLM_PORT}"

# Kill existing session if running
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Killing existing tmux session '${SESSION_NAME}'..."
    tmux kill-session -t "${SESSION_NAME}"
fi

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/vllm.log"

echo "----------------------------------------"
echo "Starting vLLM server in tmux session '${SESSION_NAME}'..."
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Log: ${LOG_FILE}"
echo "----------------------------------------"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Tail log with: tail -f ${LOG_FILE}"

# Start vLLM server in a new tmux session
tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SCRIPT_DIR} && source .venv/bin/activate && export GLOO_SOCKET_IFNAME=lo && python -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_NAME}' \
    --port '${PORT}' \
    --max-model-len '${VLLM_MAX_MODEL_LEN}' \
    --gpu-memory-utilization '${VLLM_GPU_MEMORY_UTILIZATION}' \
    --max-num-seqs '${VLLM_MAX_NUM_SEQS}' \
    ${VLLM_MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens '${VLLM_MAX_NUM_BATCHED_TOKENS}'} \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --no-enable-log-requests \
    --allowed-local-media-path /home/jason_yp_wang \
    2>&1 | tee '${LOG_FILE}'"
