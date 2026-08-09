#!/usr/bin/env bash
# Report which GPU architectures the installed Transformer Engine binaries actually carry.
#
# Worth checking directly rather than trusting NVTE_CUDA_ARCHS: the FP4 conversion PTX is only
# emitted for architecture-specific targets (sm_120a, not sm_120), and a build that silently
# dropped the "a" still loads, still reports NVFP4 support, and still prints
# "FP4 cvt PTX instructions are architecture-specific" from inside the kernel.
set -u
CUDA="${CUDA_HOME:-/usr/local/cuda-13.3}"
PKG=$(/opt/ale/bin/python -c 'import torch, transformer_engine as t, os; print(os.path.dirname(t.__file__))')
echo "package: $PKG"
find "$PKG" -name '*.so' | while read -r so; do
    echo "=== $(basename "$so")"
    ls -l --time-style=+%m-%d_%H:%M "$so" | awk '{print "    installed:", $6}'
    archs=$("$CUDA/bin/cuobjdump" --list-elf "$so" 2>/dev/null | grep -oE 'sm_[0-9]+a?' | sort -u | paste -sd' ')
    echo "    archs: ${archs:-<none found>}"
    case "$archs" in
        *sm_120a*) echo "    -> FP4 stochastic rounding: supported" ;;
        *sm_120*)  echo "    -> sm_120 WITHOUT the 'a' variant: FP4 casts will fail at runtime" ;;
    esac
done
