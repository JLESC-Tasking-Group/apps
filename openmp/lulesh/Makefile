# =============================================================================
# LULESH - build configuration
#
# Build with:   make
#
# CPU vs GPU is chosen HERE via the flag blocks below (not by a separate target):
# keep the default flags for a host-task build, or uncomment the GPU block for an
# OpenMP target-offload build. One source, two backends (see tasking.h): every
# per-block loop is a host task on the CPU or an offloaded target task on the GPU.
#
# Task granularity (number of tasks per loop on the CPU backend / number of
# offloaded blocks on the GPU backend) is a runtime knob: -nb <n> (default 32).
# =============================================================================

# ---- Compiler -------------------------------------------------------------
# clang++ is the supported compiler (the taskgraph construct needs Julian's LLVM
# / XKOMP fork), consistent with the other apps in this repo. A portable CPU
# build (USE_TASKGRAPH=0) also works on vanilla clang++.
#CC = clang++
CC = xkcxx -DUSE_XKOMP=1

# ---- Common flags (edit me) ----------------------------------------------
CFLAGS  =
CFLAGS += -fopenmp -fopenmp-version=60
CFLAGS += -O3
#CFLAGS += -O0 -g
# Record/replay the identical per-iteration task graph. Needs a taskgraph-enabled
# clang (emits replayable(1) + the taskgraph construct). Set to 0 for a portable
# build on vanilla clang++ (plain OpenMP tasks / target tasks).
CFLAGS += -DUSE_TASKGRAPH=1
CFLAGS += -DUSE_REPLAYABLE=0
# Synchronous execution mode: run each kernel to completion before the next (no
# tasks/depend/nowait/taskgraph); overrides USE_TASKGRAPH. On the CPU backend it
# degenerates to a serial run.
#CFLAGS += -DUSE_SYNC=1

# XKOMP JIT flags
#CFLAGS += -fopenmp-task-jit-type=none
#CFLAGS += -fopenmp-task-jit-type=pointers
CFLAGS += -fopenmp-task-jit-type=packed

LDFLAGS  =

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
#LDFLAGS += -L/soft/compilers/cuda/cuda-13.3.1/lib64
#LDFLAGS += -lcudart

# Pinned host-memory backend for the mapped domain arrays (see lulesh.h's
# AllocateForTarget). Pick one when building the GPU backend; the default (none)
# on the CPU backend uses plain malloc:
#CFLAGS  += -DALLOC_CUDA                    # cudaMallocHost (needs -lcudart above)
#CFLAGS  += -DALLOC_HIP                     # hipHostMalloc
#LDFLAGS += -lamdhip64                      # ... its runtime lib, for ALLOC_HIP
#CFLAGS  += -DALLOC_OMP                     # omp_alloc pinned allocator

# mi250 / mi300x (ROCm) example:
#CC = clang++
#CFLAGS  += -I $(ROCM_PATH)/include
#CFLAGS  += -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx90a,gfx942
#CFLAGS  += -DALLOC_HIP=1
#LDFLAGS += -L $(ROCM_PATH)/lib/ -lamdhip64

# Optional energy measurement backend for the FOM/watt figure (see energy.cc).
# Leave all off for a plain build (power is then reported as 0):
#CFLAGS  += -DCUDA_ENERGY=1 ; LDFLAGS += -lnvidia-ml     # NVIDIA (NVML)
#CFLAGS  += -DROCM_ENERGY=1 ; LDFLAGS += -lrocm_smi64    # AMD (ROCm SMI)
#CFLAGS  += -DZES_ENERGY=1  ; LDFLAGS += -lze_loader     # Intel (Level Zero)

# ---- Sources --------------------------------------------------------------
program = lulesh

source = lulesh.cc \
        lulesh-viz.cc \
        lulesh-util.cc \
        lulesh-init.cc \
        energy.cc

obj = $(source:.cc=.o)

HDR = lulesh.h energy.h tasking.h

# =============================================================================
# Targets to build
# =============================================================================
$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) mpc-omp-task-trace-* report*

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	./$(program) -i 100 -s 128 -r 11 -b 1 -c 1

.PHONY: clean test run
