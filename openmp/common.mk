# =============================================================================
# common.mk - shared build configuration for the apps/openmp taskgraph apps
# (krylov, lulesh, llm.c). Each app's Makefile does `include ../common.mk` and
# then adds its own sources / defines / targets. The top-level Makefile only
# dispatches (`$(MAKE) -C <app>`).
#
# Backend and schedule are selected by overridable toggles, e.g.:
#     make                       # CPU tasks, taskgraph on   (defaults)
#     make USE_TARGET=1          # GPU target offload
#     make USE_SYNC=1            # synchronous blocking (no tasks/taskgraph)
#     make USE_TASKGRAPH=0       # plain tasks/target, no record/replay
# The evaluation harness (scripts/evaluate.py) drives these to build the
# synchronous / no-taskgraph / taskgraph reference configurations.
# =============================================================================

# ---- Compiler -------------------------------------------------------------
# clang(++) is the supported compiler (the taskgraph construct needs Julian's
# LLVM / XKOMP fork). A portable CPU build (USE_TASKGRAPH=0) also works on
# vanilla clang. Override on the command line with `make CC=clang++`.
CC = xkcxx -DUSE_XKOMP=1

# ---- Backend / schedule toggles (override on the command line) ------------
USE_TARGET     ?= 0     # 0: host CPU tasks        1: GPU target offload
USE_TASKGRAPH  ?= 1     # 1: record/replay graph   0: plain tasks/target
USE_SYNC       ?= 0     # 0: asynchronous tasks     1: synchronous blocking
USE_REPLAYABLE ?= 0     # mark task-generating constructs replayable(1)

# ---- Common flags ---------------------------------------------------------
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3
#CFLAGS += -O0 -g
CFLAGS += -fopenmp-task-jit-type=packed        # XKOMP JIT (none|pointers|packed)
CFLAGS += -DUSE_TARGET=$(USE_TARGET)
CFLAGS += -DUSE_TASKGRAPH=$(USE_TASKGRAPH)
CFLAGS += -DUSE_SYNC=$(USE_SYNC)
CFLAGS += -DUSE_REPLAYABLE=$(USE_REPLAYABLE)

LDFLAGS += -lm

# ---- GPU (OpenMP target offload), active only when USE_TARGET=1 -----------
# Adjust --offload-arch to your device (sm_80=A100, sm_90=H100, gfx942=MI300X);
# it is auto-detected from nvidia-smi when available.
ifeq ($(USE_TARGET),1)
  CFLAGS  += -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-offload-mandatory
  #CFLAGS += -foffload-lto
  DETECTED_SMS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u | tr -d '.')
  ifneq ($(DETECTED_SMS),)
    CFLAGS += $(foreach sm,$(DETECTED_SMS),--offload-arch=sm_$(sm))
  else
    CFLAGS += --offload-arch=sm_80
  endif
  # CUDA runtime for cudaMallocHost + the offload lib. Edit the -L path for your
  # site if libcudart is not on the default linker search path.
  LDFLAGS += -L/soft/compilers/cuda/cuda-13.3.1/lib64
  LDFLAGS += -lcudart
  # Pinned host-memory backend for the mapped buffers (see each app's allocator).
  # Default here is cudaMallocHost; switch to ALLOC_HIP / omit for the OpenMP
  # pinned allocator as needed.
  CFLAGS  += -DALLOC_CUDA
  #CFLAGS  += -DALLOC_HIP
  #LDFLAGS += -lamdhip64
endif
