#!/bin/bash

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Trap for cleanup on script exit
trap cleanup EXIT

# Cleanup function
cleanup() {
    local exit_code=$?
    
    # Always clean up temporary lock file
    if [ -f "uv.lock" ]; then
        rm -f uv.lock
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Installation failed with exit code $exit_code"
        echo ""
        echo "📞 For help:"
        echo "• Check README.md for detailed installation instructions"
        echo "• Review the error messages above"
        echo "• Consider manual installation steps"
        
        if [ -d ".venv" ]; then
            echo ""
            echo "🧹 Partial installation detected. To clean up:"
            echo "   rm -rf .venv"
        fi
    fi
}

# Basic system detection
OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "🚀 Installing SSM Analysis ($OS_TYPE-$ARCH)..."

# Check Python 3.12
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 required. Install it first."
    exit 1
fi

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    pip install uv
fi

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# CUDA detection
CUDA_AVAILABLE=false

echo "🔍 Checking CUDA..."

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        CUDA_AVAILABLE=true
        echo "✅ Found $GPU_COUNT GPU(s)"
    fi
    elif command -v nvcc &> /dev/null; then
    # Check nvcc and get CUDA version
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    CUDA_AVAILABLE=true
    echo "✅ NVCC found - CUDA $NVCC_VERSION"
else
    # Check CUDA installation
    for cuda_dir in "/usr/local/cuda" "/opt/cuda"; do
        if [ -d "$cuda_dir" ]; then
            CUDA_AVAILABLE=true
            echo "✅ CUDA installation: $cuda_dir"
            break
        fi
    done
fi

# Set installation mode
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Installing with GPU support..."
    export UV_TORCH_BACKEND=auto
    cp uv.cuda.lock uv.lock 2>/dev/null || true
else
    echo "💻 Installing CPU-only mode..."
    export UV_TORCH_BACKEND=cpu
    cp uv.cpu.lock uv.lock 2>/dev/null || true
fi

# Install dependencies
if [ "$CUDA_AVAILABLE" = true ]; then
    uv sync --extra typing --extra streamlit --extra dev --extra gpu
else
    uv sync --extra typing --extra streamlit --extra dev --extra cpu
fi

# Apply patches
echo "🔧 Applying patches..."

echo "🔧 streamlit-pydantic patch"
STREAMLIT_PYDANTIC_FILE=".venv/lib/python3.12/site-packages/streamlit_pydantic/settings.py"
if [ -f "$STREAMLIT_PYDANTIC_FILE" ]; then
    if [ "$OS_TYPE" = "darwin" ]; then
        # macOS (BSD) sed
        sed -i '' 's/from pydantic import BaseSettings/from pydantic_settings import BaseSettings/' "$STREAMLIT_PYDANTIC_FILE"
    else
        # Linux (GNU) sed
        sed -i 's/from pydantic import BaseSettings/from pydantic_settings import BaseSettings/' "$STREAMLIT_PYDANTIC_FILE"
    fi
fi

echo "🔧 causal_conv1d patch"
CAUSAL_CONV1D_FILE=".venv/lib/python3.12/site-packages/causal_conv1d/__init__.py"
if [ -f "$CAUSAL_CONV1D_FILE" ]; then
    cat > "$CAUSAL_CONV1D_FILE" << 'EOF'
__version__ = "1.5.0.post8"
import torch
try:
    if torch.cuda.is_available() and (torch.cuda.get_device_capability() > (6,1)):
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update
    else:
        causal_conv1d_fn, causal_conv1d_update = None, None
except (ImportError, RuntimeError):
    causal_conv1d_fn, causal_conv1d_update = None, None
EOF
fi

echo "🔍 Verifying successful installation..."

# Test installation
python -c "
import torch
from src.core.types import Float
print('✅ Installation successful!')
print(f'Mode: $CUDA_AVAILABLE')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import mamba_ssm
    print('✅ mamba-ssm available')
except ImportError:
    print('ℹ️  mamba-ssm not available (CPU mode)')
"

echo "🎉 Done! Activate with: source .venv/bin/activate"
