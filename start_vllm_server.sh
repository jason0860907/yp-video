#!/bin/bash
# Start vLLM server (config from vllm.env)
#
# Usage:
#   ./start_vllm_server.sh [MODEL_NAME] [PORT]
#
# Examples:
#   ./start_vllm_server.sh                                    # Use defaults from vllm.env
#   ./start_vllm_server.sh Qwen/Qwen3-VL-4B-Instruct 8001    # Override model and port

# Load shared config
source "$(dirname "$0")/vllm.env"

MODEL_NAME="${1:-$VLLM_MODEL}"
PORT="${2:-$VLLM_PORT}"

echo "Starting vLLM server..."
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "----------------------------------------"

# Start vLLM server with OpenAI-compatible API
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --disable-log-requests \
    --allowed-local-media-path /home/jason_yp_wang
