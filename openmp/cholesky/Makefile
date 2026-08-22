# =============================================================================
# Tiled Cholesky factorization - build configuration
#
# Build with:   make
#
# CPU vs GPU is chosen HERE via the flag blocks below (not by a separate target):
# keep the default flags for a host-task build, or uncomment the GPU block for an
# OpenMP target-offload build. One source, two backends (see tasking.h).
# =============================================================================

# ---- Compiler -------------------------------------------------------------
# clang is the supported compiler (the taskgraph construct needs Julian's LLVM /
# XKOMP fork), consistent with the other apps in this repo. A portable CPU build
# (USE_TASKGRAPH=0) also works on vanilla clang; drop -fopenmp-version=60 for gcc.
#CC = clang
CC = xkcxx -DUSE_XKOMP=1

# ---- Common flags (edit me) ----------------------------------------------
CFLAGS  =
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3
#CFLAGS += -ffast-math
#CFLAGS += -O0 -g
CFLAGS += -Wall -Wextra
# Compile-time problem defaults (all also settable at runtime, see -h):
CFLAGS += -DDEFAULT_NT=8
CFLAGS += -DDEFAULT_TS=256
CFLAGS += -DDEFAULT_REPS=1
# Record/replay the identical per-repetition task graph. Needs a taskgraph-
# enabled clang (emits replayable(1) + the taskgraph construct). Set to 0 for a
# portable build on vanilla clang/gcc (plain OpenMP tasks).
CFLAGS += -DUSE_TASKGRAPH=1
CFLAGS += -DUSE_REPLAYABLE=0
# Synchronous execution mode: run each kernel to completion before the next
# (no tasks/depend/nowait/taskgraph); overrides USE_TASKGRAPH. On the CPU backend
# it degenerates to a serial run.
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
#CFLAGS  += -DUSE_TARGET=1
#CFLAGS  += -fopenmp-targets=nvptx64-nvidia-cuda
#DETECTED_SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u | tr -d '.')
#ifneq ($(DETECTED_SMS),)
#  CFLAGS += $(foreach sm,$(DETECTED_SMS),--offload-arch=sm_$(sm))
#else
#  CFLAGS += --offload-arch=sm_80
#endif
#CFLAGS += -fopenmp-offload-mandatory
##CFLAGS += -foffload-lto
#LDFLAGS += -lcudart

# Pinned host-memory backend for the mapped matrix (see kalloc.h). The default
# (nothing set) uses the OpenMP pinned allocator (omp_alloc); pick a vendor path
# for an explicit one:
#CFLAGS  += -DALLOC_CUDA                    # cudaMallocHost (uses -lcudart above)
#CFLAGS  += -DALLOC_HIP                     # hipHostMalloc
#LDFLAGS += -lamdhip64                      # ... its runtime lib, for ALLOC_HIP

# ---- Sources --------------------------------------------------------------
SRC = main.c kalloc.c
HDR = tasking.h kalloc.h kernels.h

cholesky: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o cholesky $(LDFLAGS)

clean:
	rm -f cholesky

.PHONY: clean
