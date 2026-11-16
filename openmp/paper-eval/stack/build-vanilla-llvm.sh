#!/bin/bash
# Build vanilla LLVM (clang, lld, KMP libomp, libomptarget + device runtime) from an
# upstream checkout ($VANILLA_SRC, master), the reference toolchain of the sweep.
# Usage: MACHINE=a100 bash build-vanilla-llvm.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env/${MACHINE:?set MACHINE}.conf" || exit 1
J=${J:-$(nproc)}
export PATH=$CMAKE_BIN:$CUDA_PATH/bin:$PATH LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}
mkdir -p $VANILLA_BUILD && cd $VANILLA_BUILD || exit 1
echo "vanilla src $(git -C $VANILLA_SRC log --oneline -1 | cut -c1-80)"
CC=$BOOT_CC CXX=$BOOT_CXX cmake \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$VANILLA \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES="openmp" \
  -DLLVM_RUNTIME_TARGETS="default;nvptx64-nvidia-cuda" \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
  -DLLVM_HOST_TRIPLE=$HOST_TRIPLE -DLLVM_DEFAULT_TARGET_TRIPLE=$HOST_TRIPLE \
  -DLLVM_PARALLEL_LINK_JOBS=8 -DPython3_EXECUTABLE=$PYTHON \
  $VANILLA_SRC/llvm > cmake.log 2>&1 || { echo "cmake failed"; tail -30 cmake.log; exit 1; }
make install -j $J > make.log 2>&1; rc=$?
echo "vanilla rc=$rc $(date -u +%T)"; [ $rc -eq 0 ] || grep -m5 -B2 -A5 "error" make.log
