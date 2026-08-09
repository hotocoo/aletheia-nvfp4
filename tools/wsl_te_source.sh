#!/usr/bin/env bash
# Build Transformer Engine -- core library included -- from source for sm_120a.
#
# Why this exists: NVIDIA's prebuilt `transformer_engine_cu13` wheel carries
#   sm_75 sm_80 sm_89 sm_90 sm_90a sm_100 sm_100a sm_103a sm_120
# The "a" (architecture-specific) variants are shipped for datacenter Blackwell but not fo
# consumer sm_120, and the FP4 conversion PTX -- including
# `mul_cvt_bf16_to_fp4_8x_stochastic_rounding` -- is only emitted for sm_120a. On an RTX 50-series
# card the prebuilt wheel therefore reports NVFP4 support, runs the forward quantization, and then
# aborts in the gradient cast:
#   "FP4 cvt PTX instructions are architecture-specific. Try recompiling with sm_XXXa"
# Rebuilding only the PyTorch extension does not help: the failing kernel is in the core library.
set -uo pipefail
VENV=/opt/ale
# Pinned to 13.3, not the /usr/local/cuda symlink. Ubuntu 26.04's glibc 2.43 declares rsqrt and
# rsqrtf, and CUDA 13.0's crt/math_functions.h declares them incompatibly -- that toolkit cannot
# compile *any* .cu file here, including CMake's own compiler-identification probe, which fails
# as the misleading "CMAKE_CXX_COMPILER not set, after EnableLanguage". Verify with
# `bash wsl_nvcc_probe.sh /usr/local/cuda-13.3` before trusting a build.
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NVTE_CUDA_ARCHS="120a"
export NVTE_FRAMEWORK=pytorch
# TE vendors its own NCCL and builds an expert-parallel "NCCL EP" component by default. It is
# for multi-GPU expert routing, which a single 5070 has no use for, and it does not compile here:
# the vendored source expects a different NCCL struct layout than the headers on this box, and its
# nvcc flags collide with glibc's math headers ("exception specification is incompatible ...
# rsqrt"). Neither failure touches the FP4 kernels we actually need.
export NVTE_WITH_NCCL_EP=0
export MAX_JOBS="${MAX_JOBS:-4}"      # each nvcc job peaks 2-4 GB against a 12 GB WSL cap

# cuDNN comes from torch's pip dependency, and it is handed to CMake as a variable rather than
# through CPATH. CPATH demotes the directories it lists out of "system header" status, which makes
# nvcc parse glibc's <math.h> as user code and collide with CUDA's own declarations:
#   mathcalls.h(206): error: exception specification is incompatible ... "rsqrt"
# That fires during CMake's compiler-identification step, so the build fails before it starts with
# the misleading "CMAKE_CXX_COMPILER not set".
# TE's util/logging.h includes <nccl.h> unconditionally -- even with NCCL EP disabled -- and the
# fused-attention sources need cuDNN. Both headers ship inside torch's pip dependencies. They are
# collected into one directory and handed to CMake as a single -I token per language:
#   * one token, because NVTE_CMAKE_EXTRA_ARGS is split on whitespace
#   * -I rather than CPATH, because CPATH demotes *every* directory it touches out of system-heade
#     status, which makes nvcc parse glibc's <math.h> as user code
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
INC=/opt/ale/te_include
mkdir -p "$INC"
for d in "$SITE"/nvidia/nccl/include "$SITE"/nvidia/cudnn/include; do
    [ -d "$d" ] && ln -sf "$d"/*.h "$INC"/ 2>/dev/null
done
echo "extra headers: $(ls "$INC" | tr '\n' ' ')"

CUDNN="$SITE/nvidia/cudnn"
export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_FLAGS=-I$INC -DCMAKE_CXX_FLAGS=-I$INC"
if [ -f "$CUDNN/lib/libcudnn.so" ]; then
    export NVTE_CMAKE_EXTRA_ARGS="$NVTE_CMAKE_EXTRA_ARGS -DCUDNN_INCLUDE_DIR=$CUDNN/include -DCUDNN_LIBRARY=$CUDNN/lib/libcudnn.so"
    export LD_LIBRARY_PATH="$CUDNN/lib:$LD_LIBRARY_PATH"
fi
export LIBRARY_PATH="$SITE/nvidia/nccl/lib:$CUDNN/lib:${LIBRARY_PATH:-}"
# The build has two halves with two different build systems: the core library goes through CMake
# (which takes the -I flags above), and the PyTorch extension goes through torch's setuptools
# builder, which never sees them. CPATH is what reaches the second half. It is safe here only
# because CUDA 13.3 is in use -- under 13.0 it looked like the cause of the glibc rsqrt clash,
# but that clash happened with or without it.
export CPATH="$INC:${CPATH:-}"

apt-get install -y -qq git cmake ninja-build >/dev/null 2>&1
# --no-build-isolation means pip will not provision TE's build requirements, so they have to be
# present in the venv already. Building in isolation instead is not an option: the build imports
# the installed torch to discover its ABI, and an isolated env would not have it.
"$VENV/bin/pip" install -q pybind11 cmake ninja packaging setuptools wheel

# Remove the prebuilt wheels FIRST. They own `transformer_engine/common/`, so uninstalling them
# after a source build deletes that package out from under it, leaving the freshly compiled .so
# files with no Python tree. Leaving them installed is equally broken: TE's import-time sanity
# check finds the cu13 core package, concludes it must be a PyPI install, and refuses to load
# with "Could not find `transformer-engine` PyPI package."
"$VENV/bin/pip" uninstall -y -q transformer_engine transformer_engine_cu13 transformer_engine_torch 2>/dev/null
rm -rf "$SITE/transformer_engine" "$SITE"/transformer_engine*.dist-info

LOG=/root/te_source_build.log        # not /tmp: it does not survive a distro restart here
echo "arch: $NVTE_CUDA_ARCHS   jobs: $MAX_JOBS   log: $LOG"
"$VENV/bin/pip" install --no-build-isolation --no-cache-dir --force-reinstall --no-deps -v \
    "git+https://github.com/NVIDIA/TransformerEngine.git@v2.17.1#egg=transformer_engine" \
    >"$LOG" 2>&1
rc=$?
echo "exit: $rc  ($(wc -l <"$LOG") lines)"
if [ "$rc" -ne 0 ]; then
    echo "=== first hard error ==="
    grep -n -m 5 -E "error:|fatal error|CMake Error|No such file" "$LOG" || tail -25 "$LOG"
fi
exit "$rc"
