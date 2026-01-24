#!/bin/bash
set -e

echo "=== Setting up InternVideo-Next environment ==="

# Step 1: Sync dependencies with uv
echo "[1/3] Syncing dependencies with uv..."
uv sync --no-install-project

# Step 2: Install flash-attn (requires compilation)
echo "[2/3] Installing flash-attn (this may take 10-20 minutes)..."
uv pip install flash-attn --no-build-isolation

# Step 3: Verify installation
echo "[3/3] Verifying installation..."
uv run python -c "
import torch
import transformers
import timm
import einops
import decord
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
try:
    from flash_attn import flash_attn_func
    print('flash-attn: OK')
except ImportError as e:
    print('flash-attn: FAILED -', e)
print('All dependencies installed successfully!')
"

echo "=== Setup complete ==="
