#!/usr/bin/env bash
set -euo pipefail

echo "=== llm-quant-profiler WSL2 setup ==="

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "ERROR: this project must be set up inside WSL2."
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable inside WSL2."
    exit 1
fi

VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-quant-profiler}"

if command -v uv >/dev/null 2>&1; then
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        uv venv "$VENV_DIR" --python 3.11
    fi
    uv pip install --python "$VENV_DIR/bin/python" \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
    uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ or uv is required")
PY
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/python" -m ensurepip --upgrade
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    "$VENV_DIR/bin/python" -m pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
fi

echo "=== Verification ==="
"$VENV_DIR/bin/python" - <<'PY'
import accelerate
import bitsandbytes
import torch
import transformers
import triton

print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.version.cuda}")
print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"Transformers: {transformers.__version__}")
print(f"Accelerate:   {accelerate.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")
print(f"Triton:       {triton.__version__}")
PY

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
echo "Then run: python scripts/run_phase3.py --local-files-only"
