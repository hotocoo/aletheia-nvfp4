#!/usr/bin/env bash
# Can this CUDA toolkit compile *any* .cu file against this system's glibc?
#
# Ubuntu 26.04 ships glibc 2.43, which declares rsqrt/rsqrtf. CUDA toolkits older than that
# declare the same names differently in crt/math_functions.h, and nvcc then rejects every
# translation unit -- including CMake's own compiler-identification probe, which fails with the
# misleading "CMAKE_CXX_COMPILER not set, after EnableLanguage".
#
#   bash wsl_nvcc_probe.sh                       # probes /usr/local/cuda
#   bash wsl_nvcc_probe.sh /usr/local/cuda-13.3  # probes a specific toolkit
set -u
CUDA="${1:-/usr/local/cuda}"
echo "int main(){return 0;}" > /tmp/nvcc_probe.cu
"$CUDA/bin/nvcc" --version | tail -2
if "$CUDA/bin/nvcc" /tmp/nvcc_probe.cu -o /tmp/nvcc_probe 2>/tmp/nvcc_probe.err; then
    echo "PASS: $CUDA compiles CUDA on this system"
    exit 0
fi
echo "FAIL: $CUDA cannot compile a trivial .cu here"
head -4 /tmp/nvcc_probe.er
exit 1
