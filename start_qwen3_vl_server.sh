#!/bin/bash
# Start vLLM server for Qwen3-VL model
#
# Usage:
#   ./start_qwen3_vl_server.sh [MODEL_NAME] [PORT]
#
# Examples:
#   ./start_qwen3_vl_server.sh                                    # Use default 7B model on port 8000
#   ./start_qwen3_vl_server.sh Qwen/Qwen3-VL-4B-Instruct 8001     # Use 4B model on port 8001

MODEL_NAME="${1:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${2:-8000}"

echo "Starting vLLM server..."
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "----------------------------------------"

# Install vllm if not available
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vllm..."
    pip install vllm>=0.8.0
fi

# Start vLLM server with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path /home/jason_yp_wang
