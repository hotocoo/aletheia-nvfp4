#!/usr/bin/env bash
# Provision the WSL2 side of the FP4 path: CUDA torch, Transformer Engine, notebook deps.
# Idempotent -- safe to re-run. Run as root inside the Ubuntu distro:
#   wsl -d Ubuntu -u root -- bash /mnt/c/Users/<you>/.../tools/wsl_setup.sh
set -u
VENV=/opt/ale
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "=== base packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev build-essential git curl ninja-build cmake pkg-config >/dev/null

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PIP" install -q --upgrade pip setuptools wheel

echo "=== torch (cu130) ==="
"$PIP" install -q torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
EOF

echo "=== notebook dependencies ==="
"$PIP" install -q numpy tokenizers transformers datasets safetensors matplotlib tqdm \
    zstandard papermill ipykernel huggingface_hub ipywidgets

echo "=== lm-eval (public benchmark suite) ==="
# The notebook's eval stage checks for this at import time and silently skips every public
# benchmark when it is missing -- which you only discover at the end of a multi-day run.
"$PIP" install -q lm-eval

echo "=== flash-linear-attention (gated DeltaNet slots) ==="
# Without this, HAS_FLA is False and every layer falls back to full attention -- the 3:1
# hybrid the config advertises silently does not run.
"$PIP" install -q "flash-linear-attention>=0.4.0" einops

echo "=== transformer engine (FP4) ==="
"$PIP" install --no-build-isolation "transformer_engine[pytorch]" 2>&1 | tail -5

echo "=== capability probe ==="
"$PY" - <<'EOF'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
try:
    import transformer_engine.pytorch as te
    print("transformer_engine", md.version("transformer_engine"))
    try:
        from transformer_engine.pytorch.quantization import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support)
    except Exception:
        from transformer_engine.pytorch.fp8 import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support)
    for name, fn in (("NVFP4", check_nvfp4_support), ("MXFP8", check_mxfp8_support),
                     ("FP8", check_fp8_support)):
        ok, why = fn()
        print(f"{name}: {ok} {'' if ok else why}")
except Exception as exc:
    print("transformer_engine UNAVAILABLE:", type(exc).__name__, exc)
try:
    import triton
    print("triton", triton.__version__, "-> torch.compile available")
except Exception as exc:
    print("triton missing:", exc)
EOF
