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
#
# USE_OMPSS=1 targets NODES/OmpSs-2 instead, which is a different toolchain, so
# it takes its compiler from OMPSS_CC (default: the OmpSs-2 clang++). Selecting
# it here rather than at the call site keeps `make USE_OMPSS=1` working on its
# own -- the XKOMP default would otherwise be silently wrong.
USE_OMPSS ?= 0
ifeq ($(USE_OMPSS),1)
  OMPSS_CC ?= clang++
  CC = $(OMPSS_CC)
else
  CC = xkcxx -DUSE_XKOMP=1
endif

# ---- Backend / schedule toggles (override on the command line) ------------
USE_TARGET     ?= 0     # 0: host CPU tasks        1: GPU target offload
USE_TASKGRAPH  ?= 1     # 1: record/replay graph   0: plain tasks/target
USE_TASKGRAPHLOOP ?= 1  # 1: unroll -u iterations into ONE graph instance (taskgraphloop)
                        # 0: one graph instance per iteration (the A/B baseline)
USE_SYNC       ?= 0     # 0: asynchronous tasks     1: synchronous blocking
USE_REPLAYABLE ?= 0     # mark task-generating constructs replayable(1)
                        # (USE_OMPSS is declared above, next to the compiler it selects)

# ---- Common flags ---------------------------------------------------------
# -I.. makes the shared apps/openmp/tasking.h resolvable as #include "tasking.h"
# from each app's build dir (one level below apps/openmp).
CFLAGS += -I..

# ---- Optimization level ----------------------------------------------------
# Override with e.g. `make OPT="-O0 -g"`.
#
# WARNING: -O0 BREAKS GPU OFFLOAD on this toolchain (clang fork + xkomp/xkrt,
# NVPTX sm_90). A `target teams distribute parallel for` whose body makes device
# function calls -- i.e. anything the optimizer would otherwise have inlined --
# faults with CUDA_ERROR_ILLEGAL_ADDRESS (700) once roughly 10^5 threads are
# concurrently inside the call chain. It is NOT a stack-size problem
# (LIBOMPTARGET_STACK_SIZE up to 256 KiB changes nothing) and NOT an application
# bug: MNMGDatalog/tc_repro.cpp reproduces it with no atomics, no tasking and
# provably in-bounds indexing, and the same binary passes at -O3.
# See MNMGDatalog/REPRODUCER.md for the full matrix.
#
# Use `-O0 -g` for CPU debugging (USE_TARGET=0) only.
OPT ?= -O3
CFLAGS += $(OPT)

ifeq ($(USE_TARGET),1)
  ifneq ($(filter -O0,$(OPT)),)
    $(warning *** -O0 with USE_TARGET=1 is known to fault on this toolchain.)
    $(warning *** Device kernels that call non-inlined helpers hit)
    $(warning *** CUDA_ERROR_ILLEGAL_ADDRESS. See MNMGDatalog/REPRODUCER.md.)
  endif
endif

CFLAGS += -DUSE_TARGET=$(USE_TARGET)
CFLAGS += -DUSE_TASKGRAPH=$(USE_TASKGRAPH)
CFLAGS += -DUSE_TASKGRAPHLOOP=$(USE_TASKGRAPHLOOP)
CFLAGS += -DUSE_SYNC=$(USE_SYNC)
CFLAGS += -DUSE_REPLAYABLE=$(USE_REPLAYABLE)
CFLAGS += -DUSE_OMPSS=$(USE_OMPSS)

LDFLAGS += -lm

# ---- Tasking backend flags -------------------------------------------------
# The shared tasking.h switches `omp task` -> `oss task` when USE_OMPSS=1. The two
# backends are different toolchains and their flags do not overlap: -fompss-2
# already implies the OmpSs-2 tasking front end, and -fopenmp-task-jit-abi is an
# XKOMP-fork flag that the OmpSs-2 clang rejects. (Mutually exclusive with
# USE_TARGET; tasking.h enforces that with an #error.)
ifeq ($(USE_OMPSS),1)
  # libnodes both selects the runtime and makes the compiler preserve each task
  # body as LLVM-IR in the NANOS6 convention, which is what CGIR's prog-fuse/jit
  # passes consume.
  CFLAGS  += -fompss-2=libnodes
  LDFLAGS += -lnuma
else
  CFLAGS  += -fopenmp -fopenmp-version=60
  CFLAGS  += -fopenmp-task-jit-abi=packed     # XKOMP JIT (none|pointers|packed)
endif

# ---- GPU (OpenMP target offload), active only when USE_TARGET=1 -----------
# Adjust --offload-arch to your device (sm_80=A100, sm_90=H100, gfx942=MI300X);
# it is auto-detected from nvidia-smi when available.
ifeq ($(USE_TARGET),1)
  CFLAGS  += -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-offload-mandatory
  # REQUIRED for a fair baseline -- do not comment out. Without device LTO the
  # ahead-of-time kernel keeps its calls into the OpenMP DeviceRTL, whereas CGIR's
  # `jit` pass links that runtime in, internalizes everything but the entry and
  # re-optimizes, so it emits PTX with no runtime call at all. The pass then
  # measures 1.27-1.41x on Krylov -- which is this flag, not the pass. With it on,
  # JIT and ahead-of-time land within noise of each other (CG: 0.345 vs 0.346 ms),
  # which is what the paper reports.
  CFLAGS += -foffload-lto
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
