CLANG_DIR=$(shell which clang | rev | cut -d '/' -f 3- | rev)
GCC_DIR=$(shell which g++ | rev | cut -d '/' -f 3- | rev)
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
#CC        = clang++
#CC        = nvc++

#===============================================================================
# Program name & source code list
#===============================================================================

program = lulesh

source = lulesh.cc \
        lulesh-viz.cc \
        lulesh-util.cc \
        lulesh-init.cc \
		energy.cc

obj = $(source:.cc=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CC=xkcxx
LDFLAGS=
CFLAGS=-O3
#CFLAGS=-O0 -g

# GPU FLAGS
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

# XKOMP flags
# CFLAGS+=-DUSE_XKOMP_TASKGRAPH=1 -std=c++20
# CFLAGS+=-L$(XKOMP_HOME)/lib -lomp -Wl,-rpath,.
#LDFLAGS+=-L$(XKOMP_HOME)/lib -lomp -Wl,-rpath,.

# mi250, mi300x
# CC=clang++
# CFLAGS+=-I $(ROCM_PATH)/include
# CFLAGS+=-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx90a,gfx942
# CFLAGS+=-DALLOC_HIP=1
# CFLAGS+=-DROCM_ENERGY=1
# LDFLAGS+=-L $(ROCM_PATH)/lib/ -lamdhip64
# LDFLAGS+=-L $(ROCM_PATH)/lib/ -lrocm_smi64

#CFLAGS+=-fopenmp-cuda-mode -fopenmp-assume-no-thread-state -fopenmp-assume-no-nested-parallelism

# pvc
# CFLAGS+=-fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc"
# CC = icpx
# CFLAGS+=-DZES_ENERGY=1
# CFLAGS+=-I /usr/include/level_zero
# LDFLAGS+=-lze_loader

# LDFLAGS=-L$(GCC_DIR)/lib64 -lstdc++
#LDFLAGS+=-L$(CLANG_DIR)/lib -lomptarget

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc lulesh.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) mpc-omp-task-trace-* report*

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	./$(program) -i 100 -s 128 -r 11 -b 1 -c 1

