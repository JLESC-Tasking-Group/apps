# =============================================================================
# Tiled Cholesky factorization - build configuration
#
# Build with:   make
#
# CPU vs GPU is chosen HERE via the flag blocks below (not by a separate target):
# keep the default flags for a host-task build, or uncomment the GPU block for an
# OpenMP target-offload build. One source, two backends (see tasking.h).
#
# NOTE: Cholesky is intentionally NOT a record/replay taskgraph example (its tiled
# DAG changes shape every step), so there is no USE_TASKGRAPH here. It is a plain
# tasks (CPU) / target (GPU) benchmark and is not part of the shared apps/openmp
# evaluation harness.
# =============================================================================

# ---- Compiler -------------------------------------------------------------
# clang is the supported compiler for the OpenMP target/tasks features; xkcxx
# (the LLVM/XKOMP fork) works too and keeps a single toolchain across the repo.
#CC = clang
CC = xkcxx -DUSE_XKOMP=1

# ---- Common flags (edit me) ----------------------------------------------
CFLAGS  =
# -I.. resolves the shared apps/openmp/alloc.h (the pinned host allocator that
# used to be cholesky's local kalloc.{c,h}). Cholesky keeps its OWN tasking.h:
# a quoted include searches main.c's directory first, so the local copy wins.
CFLAGS += -I..
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3
#CFLAGS += -ffast-math
#CFLAGS += -O0 -g
CFLAGS += -Wall -Wextra
# Compile-time problem defaults (all also settable at runtime, see -h):
CFLAGS += -DDEFAULT_NT=8
CFLAGS += -DDEFAULT_TS=256
CFLAGS += -DDEFAULT_REPS=1
# Synchronous execution mode: run each kernel to completion before the next
# (no tasks/depend/nowait). On the CPU backend it degenerates to a serial run.
#CFLAGS += -DUSE_SYNC=1

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

# Pinned host-memory backend for the mapped matrix (see ../alloc.h). The default
# (nothing set) uses the OpenMP pinned allocator (omp_alloc); pick a vendor path
# for an explicit one:
#CFLAGS  += -DALLOC_CUDA                    # cudaMallocHost (uses -lcudart above)
#CFLAGS  += -DALLOC_HIP                     # hipHostMalloc
#LDFLAGS += -lamdhip64                      # ... its runtime lib, for ALLOC_HIP

# ---- Sources --------------------------------------------------------------
SRC = main.c ../alloc.c
HDR = tasking.h ../alloc.h kernels.h

cholesky: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o cholesky $(LDFLAGS)

clean:
	rm -f cholesky

.PHONY: clean
