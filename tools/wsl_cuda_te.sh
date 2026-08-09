#!/usr/bin/env bash
# Install the CUDA toolkit that Transformer Engine will be built against.
#
# Why a toolkit at all: TE compiles from source, and its build needs `nvcc`. The pip CUDA runtime
# packages torch pulls in carry libraries but no compiler, so the build fails with "Could neither
# find NVCC executable nor CUDA runtime Python package" until a real toolkit is present. The WSL
# GPU driver comes from the Windows host -- only the toolkit is installed here, never a driver.
#
# Why 13.3 specifically: Ubuntu 26.04 ships glibc 2.43, which declares rsqrt/rsqrtf. CUDA 13.0
# declares them incompatibly in crt/math_functions.h, and that toolkit therefore cannot compile
# *any* .cu file on this system -- not even CMake's compiler-identification probe, which fails as
# the thoroughly misleading "CMAKE_CXX_COMPILER not set, after EnableLanguage".
#
# This script no longer builds TE itself. Run tools/wsl_te_source.sh for that: the FP4 kernels
# live in TE's core library, which the prebuilt wheels ship without sm_120a.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
CUDA_VER=13-3
CUDA_DIR=/usr/local/cuda-13.3

if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "=== NVIDIA CUDA repository ==="
    apt-get install -y -qq wget gnupg >/dev/null
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
    apt-get update -qq
    echo "=== cuda toolkit $CUDA_VER (this is the long step, ~3 GB) ==="
    apt-get install -y -qq "cuda-toolkit-$CUDA_VER" >/dev/null
fi

"$CUDA_DIR/bin/nvcc" --version | tail -2

# Verify before anything is built against it -- a toolkit that cannot compile a trivial .cu will
# otherwise surface much later as an unrelated-looking CMake error.
echo "=== can this toolkit compile CUDA here? ==="
bash "$(dirname "$0")/wsl_nvcc_probe.sh" "$CUDA_DIR"

echo
echo "next: bash $(dirname "$0")/wsl_te_source.sh   # builds Transformer Engine for sm_120a"
