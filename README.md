# Mamba Knockout for Unraveling Factual Information Flow

This repository contains the research code for the paper **"Mamba Knockout for Unraveling Factual Information Flow"** accepted at ACL 2025.

📄 **Paper**: [arXiv:2505.24244](https://arxiv.org/abs/2505.24244)  
🔗 **GitHub**: [nirendy/mamba-knockout](https://github.com/nirendy/mamba-knockout)

## Abstract

This paper investigates the flow of factual information in Mamba State-Space Model (SSM)-based language models. We rely on theoretical and empirical connections to Transformer-based architectures and their attention mechanisms. Exploiting this relationship, we adapt attentional interpretability techniques originally developed for Transformers--specifically, the Attention Knockout methodology--to both Mamba-1 and Mamba-2. Using them we trace how information is transmitted and localized across tokens and layers, revealing patterns of subject-token information emergence and layer-wise dynamics. Notably, some phenomena vary between mamba models and Transformer based models, while others appear universally across all models inspected--hinting that these may be inherent to LLMs in general. By further leveraging Mamba's structured factorization, we disentangle how distinct "features" either enable token-to-token information exchange or enrich individual tokens, thus offering a unified lens to understand Mamba internal operations.

## Requirements

- **Python 3.12** (required - this project was developed and tested with Python 3.12)
- **Linux** (required for mamba-ssm compilation)
- **NVIDIA GPU** (required for mamba-ssm)
- **PyTorch 1.12+** (will be installed automatically)
- **CUDA 11.6+** (required for mamba-ssm)
- UV package manager (recommended) or pip

## Installation

> **Note**: This project has complex dependencies including CUDA-enabled packages. If you encounter import errors after installation, you may need to install additional dependencies. See the [Troubleshooting](#troubleshooting) section for common solutions.

### Option 1: Auto-installation Script (Recommended)

For convenience, use the provided auto-installation script:

```bash
# Make the script executable
chmod +x scripts/install.sh

# Run the installation script
./scripts/install.sh
```

The script will automatically:

- Create a virtual environment
- Install all dependencies in the correct order
- Apply the required patches
- Verify the installation

### Option 2: Manual Installation

If you prefer manual installation or need to customize the setup:

> **Note**: You can use either UV (recommended) or pip directly. The commands are the same, just replace `uv add` with `pip install`.

#### Using UV (Recommended)

1. **Install UV** (if not already installed):

   You can adjust the commands to use pip directly if you prefer.

   ```bash
   pip install uv
   ```

2. **Create virtual environment and install dependencies**:

   ```bash
   # Create virtual environment with Python 3.12
   uv venv --python 3.12

   # Activate the virtual environment
   source .venv/bin/activate
   ```

   **Install PyTorch**:

   See [PyTorch Installation](#pytorch-installation) for more details.

   ```bash
   # Install PyTorch with the correct CUDA version (replace *** with the correct CUDA version)
   uv add "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu***
   # or without the cuda
   # uv add "torch==2.5.1"

   # Install mamba-ssm with causal-conv1d
   uv add "mamba-ssm[causal-conv1d]==2.2.4" --no-build-isolation

   # Install the project in editable mode
   uv add -e .
   ```

   **Install additional optional dependencies as needed**:

   See [Optional Dependencies](#optional-dependencies) for more details.

   ```bash
   # - For type checking and development:
   uv add -e .[typing]
   # - For Streamlit web interface:
   uv add -e .[streamlit]
   # - For development tools:
   uv add -e .[dev]
   ```

3. **Apply required patches** (see [Known Issues and Workarounds](#known-issues-and-workarounds))

## Optional Dependencies

This project includes several optional dependency groups that you can install based on your needs:

- **`[typing]`**: Type checking and development tools. Includes `mypy`, `beartype`, and `jaxtyping` for enhanced type safety.
- **`[streamlit]`**: Web interface components for interactive analysis and visualization.
- **`[dev]`**: Development tools including `pre-commit`, `ruff`, and `pytest` for code quality and testing.

### When to Install Each Group

- **Basic usage**: Install only the core dependencies (default)
- **Development**: Add `[typing]` and `[dev]` for type checking and testing
- **Web interface**: Add `[streamlit]` for interactive visualizations
- **Full setup**: Install all groups for complete functionality

## Project Structure

```
ssm_analysis_public/
├── src/                    # Main source code
│   ├── analysis/          # Analysis modules
│   ├── core/              # Core functionality
│   ├── data_ingestion/    # Data processing
│   ├── experiments/       # Experimental code
│   ├── utils/             # Utility functions
│   └── app/               # Application components
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test files
├── data/                  # Data directory
├── final_plots/           # Generated plots and figures
└── setup.py              # Package configuration
```

## Usage

### Running Experiments

The main experimental code is located in `src/experiments/`. Key components include:

- **Knockout Analysis**: Implementation of attention knockout methodology adapted for Mamba models
- **Information Flow Analysis**: Tools for tracing factual information across tokens and layers
- **Model Analysis**: Utilities for analyzing Mamba-1 and Mamba-2 architectures

### Notebooks

The `notebooks/` directory contains Jupyter notebooks for:

- Interactive analysis and visualization
- Experiment reproduction
- Model exploration and debugging

## Troubleshooting

### Known Issues and Workarounds

#### Alternative Installation Methods (Advanced)

The automatic `uv sync` approach is preferred as it handles all dependencies automatically.
But if you encounter issues with the automatic installation, you can try alternative approaches:

**Manual pip-based installation (not recommended):**

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install PyTorch first (required for mamba-ssm)
# Adjust the CUDA version in the URL based on your system:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm
pip install "mamba-ssm[causal-conv1d]==2.2.4" --no-build-isolation

# Install the project
pip install -e .

# Install optional dependencies
pip install -e .[typing]
pip install -e .[streamlit]
pip install -e .[dev]
```

#### CUDA Compatibility for `causal_conv1d`

If you encounter an error like:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

This means your GPU is not supported by the prebuilt CUDA kernels in `causal_conv1d`. To work around this, patch the file:

```
.venv/lib/python3.12/site-packages/causal_conv1d/__init__.py
```

Replace its contents with:

```python
__version__ = "1.5.0.post8"
import torch
try:
    if (torch.cuda.get_device_capability() > (6,1)):
        print("Using causal_conv1d")
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update
    else:
        print("Not using causal_conv1d")
        causal_conv1d_fn, causal_conv1d_update = None, None
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
```

This will prevent the package from trying to use unsupported CUDA kernels on older GPUs.

#### Streamlit-Pydantic Patch

If you encounter an error like:

```
pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package
```

This is a known issue with `streamlit-pydantic` and Pydantic v2 (see [GitHub issue #69](https://github.com/lukasmasuch/streamlit-pydantic/issues/69)). To work around it, patch the file:

```
.venv/lib/python3.12/site-packages/streamlit_pydantic/settings.py
```

Replace the line:

```python
from pydantic import BaseSettings
```

With:

```python
from pydantic_settings import BaseSettings
```

This fixes the import error for Pydantic v2 compatibility.

#### HuggingFace Token Authentication

Some models or datasets used in this project may require authentication with the HuggingFace Hub. If you need to access private or gated models, you should set the `HUGGINGFACE_TOKEN` environment variable with your personal access token from HuggingFace.

- If `HUGGINGFACE_TOKEN` is set, the code will automatically use it to authenticate with the HuggingFace Hub when loading models or tokenizers.
- If the variable is not set, the code will print a message and attempt to proceed without authentication (which may fail for private/gated models).
- For more information on obtaining and using a HuggingFace token, see the official HuggingFace documentation: <https://huggingface.co/docs/hub/security-tokens>

No further setup is required in the code—just set the environment variable before running your scripts or notebooks.

### PyTorch Installation: Automatic Backend Detection

This project uses `uv`'s automatic PyTorch backend detection to install the optimal version for your system. The installation script uses:

```bash
UV_TORCH_BACKEND=auto uv sync
```

This automatically:

1. **Detects your CUDA version** using system queries
2. **Selects the appropriate PyTorch backend** (CPU, CUDA, ROCm, etc.)
3. **Installs all dependencies** in the correct order
4. **Handles the PyTorch → mamba-ssm dependency chain** automatically

**Why this approach is better:**

- **Automatic detection**: No manual CUDA version detection needed
- **Proper dependency resolution**: `uv` handles all dependencies together
- **Cleaner installation**: No step-by-step workarounds required
- **Cross-platform**: Works on Linux, macOS, and Windows

**What you'll see:**

- If you have CUDA: PyTorch will be installed with CUDA support (e.g., `+cu121`, `+cu118`)
- If no CUDA: PyTorch will be installed as CPU-only
- The exact version depends on your system's CUDA driver version

### Reproducible Lock Files

The project uses **dual lock files** for reproducible installations:

- **`uv.cuda.lock`**: Optimized for GPU systems with CUDA dependencies
- **`uv.cpu.lock`**: Optimized for CPU-only systems

The installation script automatically:

1. **Detects your system capabilities** (CPU vs GPU/CUDA)
2. **Copies the appropriate lock file** to `uv.lock` for exact dependency reproduction
3. **Cleans up the temporary lock file** when finished

This ensures faster, more reliable installations while maintaining reproducibility across different environments.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{endy2025mamba,
  title={Mamba Knockout for Unraveling Factual Information Flow},
  author={Endy, Nir and Grosbard, Idan Daniel and Ran-Milo, Yuval and Slutzky, Yonatan and Tshuva, Itay and Giryes, Raja},
  journal={arXiv preprint arXiv:2505.24244},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the research community and builds upon previous work in attention interpretability and state-space models.
