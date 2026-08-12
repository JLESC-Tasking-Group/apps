# =============================================================================
# Krylov solvers - build configuration
#
# Build a solver with its target, e.g.:   make cg
#
# CPU vs GPU is chosen HERE via the flag block below (not by a separate target):
# keep the default flags for a host-task build, or uncomment the GPU block for an
# OpenMP target-offload build. One target per solver.
# =============================================================================

# ---- Compiler -------------------------------------------------------------
# clang++ is the supported compiler (the taskgraph construct needs Julian's LLVM
# / XKOMP fork), consistent with the other apps in this repo. Other compilers may
# work for a portable CPU build (USE_TASKGRAPH=0) but need the clang-specific
# flags below adjusted (e.g. drop -fopenmp-version=60 for g++).
CC = clang++

# ---- Common flags (edit me) ----------------------------------------------
CFLAGS  =
CFLAGS += -Icommon
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3 -ffast-math
CFLAGS += -Wall -Wextra
# Compile-time problem/solver defaults (all also settable at runtime, see -h):
CFLAGS += -DGRID_N=64
CFLAGS += -DMAX_ITER=50
# Record/replay the identical per-iteration task graph. This needs a taskgraph-
# enabled clang (it emits the `replayable(1)` clause + `#pragma omp taskgraph`).
# For a portable build on vanilla clang/gcc, set this to 0 (plain OpenMP tasks).
CFLAGS += -DUSE_TASKGRAPH=1
# Use XKOMP's taskgraph API instead of LLVM's (needs C++20):
#CFLAGS += -DUSE_XKOMP=1 -std=c++20

LDFLAGS  =
LDFLAGS += -lm

# ---- GPU (OpenMP target offload) -----------------------------------------
# Uncomment this block to build the GPU backend instead of the CPU one. Adjust
# --offload-arch to your device (sm_80 = A100, sm_90 = H100, gfx942 = MI300X).
#CFLAGS  += -DUSE_TARGET=1
#CFLAGS  += -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80
#CFLAGS  += -fopenmp-offload-mandatory -foffload-lto
#LDFLAGS += -lcudart

# ---- Sources --------------------------------------------------------------
COMMON_SRC = common/spmat.cpp

# =============================================================================
# Solvers (one target each). Add new solvers as: <name>: ... $(CC) ... -o <name>.x
# =============================================================================
cg: $(COMMON_SRC) cg/cg.cpp common/spmat.h common/tasking.h
	$(CC) $(CFLAGS) $(COMMON_SRC) cg/cg.cpp -o cg.x $(LDFLAGS)

clean:
	rm -f cg.x

.PHONY: cg clean
