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
#CC = clang++
CC = xkcxx -DUSE_XKOMP=1

# ---- Common flags (edit me) ----------------------------------------------
CFLAGS  =
CFLAGS += -Icommon
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3
#CFLAGS += -ffast-math
#CFLAGS += -O0 -g
CFLAGS += -Wall -Wextra
# Compile-time problem/solver defaults (all also settable at runtime, see -h):
CFLAGS += -DGRID_N=64
CFLAGS += -DMAX_ITER=50
# Record/replay the identical per-iteration task graph. This needs a taskgraph-
# enabled clang (it emits the `replayable(1)` clause + `#pragma omp taskgraph`).
# For a portable build on vanilla clang/gcc, set this to 0 (plain OpenMP tasks).
CFLAGS += -DUSE_TASKGRAPH=1
CFLAGS += -DUSE_REPLAYABLE=0
# Synchronous execution mode: run each kernel to completion before the next
# (classic blocking `omp target`, no tasks/depend/nowait/taskgraph) -- for
# comparing against the default asynchronous task/taskgraph schedule. When set,
# it overrides USE_TASKGRAPH; on the CPU backend it degenerates to serial.
#CFLAGS += -DUSE_SYNC=1

# XKOMP JIT flags
#CFLAGS += -fopenmp-task-jit-type=none
#CFLAGS += -fopenmp-task-jit-type=pointers
CFLAGS += -fopenmp-task-jit-type=packed

LDFLAGS  =
LDFLAGS += -lm

# ---- GPU (OpenMP target offload) -----------------------------------------
# Uncomment this block to build the GPU backend instead of the CPU one. Adjust
# --offload-arch to your device (sm_80 = A100, sm_90 = H100, gfx942 = MI300X).
# With a standard offload clang (no taskgraph support), also set USE_TASKGRAPH=0
# above for a plain target-offload build.
CFLAGS  += -DUSE_TARGET=1
CFLAGS  += -fopenmp-targets=nvptx64-nvidia-cuda
DETECTED_SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u | tr -d '.')
ifneq ($(DETECTED_SMS),)
  CFLAGS += $(foreach sm,$(DETECTED_SMS),--offload-arch=sm_$(sm))
else
  CFLAGS += --offload-arch=sm_80
endif
CFLAGS += -fopenmp-offload-mandatory
#CFLAGS += -foffload-lto
LDFLAGS += -L/soft/compilers/cuda/cuda-13.3.1/lib64
LDFLAGS += -lcudart

# Pinned host-memory backend for the mapped buffers (see common/kalloc.h). The
# default (nothing set) uses the OpenMP pinned allocator (omp_alloc, which the
# runtime backs with CUDA/HIP/L0); pick a vendor path for an explicit one:
CFLAGS  += -DALLOC_CUDA                    # cudaMallocHost (uses -lcudart above)
#CFLAGS  += -DALLOC_HIP                     # hipHostMalloc
#LDFLAGS += -lamdhip64                      # ... its runtime lib, for ALLOC_HIP

# ---- Sources --------------------------------------------------------------
COMMON_SRC = common/spmat.cpp common/kalloc.cpp common/kernels.cpp common/driver.cpp
COMMON_HDR = common/spmat.h common/kalloc.h common/kernels.h common/tasking.h common/driver.h

# =============================================================================
# Solvers (one target each). Add a new solver <name> as:
#   <name>: $(COMMON_SRC) <name>/<name>.cpp $(COMMON_HDR)
#           $(CC) $(CFLAGS) $(COMMON_SRC) <name>/<name>.cpp -o <name>.x $(LDFLAGS)
# =============================================================================
cg: $(COMMON_SRC) cg/cg.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) cg/cg.cpp -o cg.x $(LDFLAGS)

cr: $(COMMON_SRC) cr/cr.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) cr/cr.cpp -o cr.x $(LDFLAGS)

bicgstab: $(COMMON_SRC) bicgstab/bicgstab.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) bicgstab/bicgstab.cpp -o bicgstab.x $(LDFLAGS)

minres: $(COMMON_SRC) minres/minres.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) minres/minres.cpp -o minres.x $(LDFLAGS)

gmres: $(COMMON_SRC) gmres/gmres.cpp $(COMMON_HDR)
	$(CC) $(CFLAGS) $(COMMON_SRC) gmres/gmres.cpp -o gmres.x $(LDFLAGS)

clean:
	rm -f *.x

.PHONY: cg cr bicgstab minres gmres clean
